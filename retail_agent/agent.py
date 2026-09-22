"""The agent loop.

One turn is: build the prompt, let the model call tools, feed results back, stop
when it produces prose. The loop is written out explicitly rather than handed to
a framework's autopilot, because every interesting requirement lives in the
seams between steps:

  * SQL is validated and scoped between the model asking and BigQuery running.
  * A failed query is classified, and only *self-correctable* failures are handed
    back to the model — with the error text — for a bounded number of retries.
    Retrying a permission error, or an exhausted quota, only inflates cost.
  * Deletion tools return a PROPOSAL, never a deletion.
  * Every step is traced, and the turn always ends with something sayable, even
    when the model, the warehouse, or the API is unavailable.

Requirement 5's "self-correct without inflating costs" is the reason for the two
counters here: MAX_SQL_CORRECTIONS bounds the repair loop, and the provider's own
call budget bounds the turn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from retail_agent import catalog
from retail_agent.bigquery_runner import BigQueryRunner, QueryError, QueryErrorKind
from retail_agent.confirmation import ConfirmationBroker, PendingDeletion
from retail_agent.llm import (
    FunctionCall,
    LLMBudgetError,
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMQuotaError,
    LLMResponse,
)
from retail_agent.observability import Tracer
from retail_agent.prompt import (
    build_system_prompt,
    missing_report_sections,
    persona_version,
    section_heading,
)
from retail_agent.reports import ReportStore
from retail_agent.safety.pii import scrub_text
from retail_agent.safety.scope import Scope

# How many times the model may rewrite a query that failed for a reason it could
# plausibly fix. Three attempts converges on real syntax slips; beyond that it is
# looping, and each attempt costs a model call and a dry run.
MAX_SQL_CORRECTIONS = 3

# Turns kept verbatim in context. Older turns are dropped rather than summarised
# — summarising is a model call per turn, and executives ask short follow-ups.
MAX_HISTORY_TURNS = 12


TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "describe_schema",
        "description": (
            "Describe the available tables and columns. Call with no arguments "
            "for the whole data model, or a table name for one table."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "orders, order_items, products or users",
                }
            },
        },
    },
    {
        "name": "run_analysis_sql",
        "description": (
            "Run a read-only BigQuery SELECT against the retail dataset and get "
            "the rows back. The query is automatically restricted to the data "
            "this user may see."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement."},
                "purpose": {
                    "type": "string",
                    "description": "One line on what this query answers.",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "save_report",
        "description": (
            "Save an analysis as a report in the user's library. Use when the "
            "user asks for a report. The body must use every report section "
            "heading listed in your instructions, including action items; a "
            "report missing one is rejected, not saved."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Markdown report body."},
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Brands, categories, states etc. mentioned.",
                },
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "list_reports",
        "description": "List the user's saved reports, optionally filtered by text.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    },
    {
        "name": "propose_delete_reports",
        "description": (
            "Prepare a deletion of saved reports for the user to confirm. This "
            "does NOT delete anything. You MUST pass exactly one selector: "
            "'text' to match report contents, this_conversation=true for reports "
            "made in this conversation, or all_reports=true ONLY when the user "
            "explicitly asked to delete everything."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Match reports mentioning this."},
                "this_conversation": {"type": "boolean"},
                "all_reports": {
                    "type": "boolean",
                    "description": (
                        "Every saved report the user owns. Only when they asked "
                        "for all of them, never as a default."
                    ),
                },
                "criteria": {
                    "type": "string",
                    "description": "Human description, e.g. \"mentioning Acme\".",
                },
            },
            "required": ["criteria"],
        },
    },
]


@dataclass
class TurnResult:
    """What the CLI needs to render one turn."""

    answer: str
    trace_id: str
    pending_deletion: PendingDeletion | None = None
    saved_report_ids: tuple[str, ...] = ()
    sql_executed: tuple[str, ...] = ()
    status: str = "ok"


@dataclass
class Agent:
    """Orchestrates one conversation for one user."""

    provider: LLMProvider
    runner: BigQueryRunner
    store: ReportStore
    broker: ConfirmationBroker
    scope: Scope
    tracer: Tracer
    conversation_id: str
    preferences: str = ""

    history: list[dict[str, Any]] = field(default_factory=list)

    # Report ids removed by the most recent confirmed deletion, so /undo can
    # restore exactly that batch. It lives on the agent rather than on a
    # frontend because every frontend offers /undo, and an attribute invented by
    # whichever one happened to run first is not a contract.
    last_deleted: tuple[str, ...] = ()

    # -- public -----------------------------------------------------------

    def ask(self, question: str) -> TurnResult:
        """Run one user turn to completion. Never raises."""
        trace_id = self.tracer.start_turn(question)
        begin = getattr(self.provider, "begin_turn", None)
        if callable(begin):
            begin()

        self._append("user", question)
        self.tracer.emit("persona.loaded", version=persona_version())

        try:
            result = self._run_loop()
        except LLMQuotaError as err:
            result = self._degrade(
                "The language model quota for this deployment is exhausted, so I "
                "cannot analyse anything right now. It needs more quota or "
                "credits for the model account — retrying will not help.",
                status="quota_exhausted",
                error=err,
            )
        except LLMConfigError as err:
            result = self._degrade(
                "The language model is misconfigured for this deployment, so I "
                "cannot answer right now. An operator needs to check the model "
                "settings — project, API access or key, and model name.",
                status="config_error",
                error=err,
            )
        except LLMBudgetError as err:
            result = self._degrade(
                "I was not able to reach a confident answer within this turn's "
                "budget. Try narrowing the question — a single metric over a "
                "single period usually gets there.",
                status="budget_exhausted",
                error=err,
            )
        except LLMError as err:
            result = self._degrade(
                "The language model is unavailable at the moment. Your data and "
                "reports are unaffected — please try again shortly.",
                status="llm_unavailable",
                error=err,
            )
        except Exception as err:  # noqa: BLE001 - the CLI must never crash
            result = self._degrade(
                "Something went wrong on my side and I could not finish that. "
                f"Quote trace {trace_id} if you report it.",
                status="internal_error",
                error=err,
            )

        answer, redactions = scrub_text(result.answer)
        if redactions:
            # Defence in depth: the model should never have produced these.
            self.tracer.metrics.pii_redactions += len(redactions)
            self.tracer.audit("pii_redacted", where="final_answer", details=redactions)
            result = TurnResult(**{**result.__dict__, "answer": answer})

        self._append("model", result.answer)
        self.tracer.end_turn(status=result.status, answer=result.answer)
        return TurnResult(**{**result.__dict__, "trace_id": trace_id})

    # -- loop -------------------------------------------------------------

    def _run_loop(self) -> TurnResult:
        system = build_system_prompt(self.scope, preferences=self.preferences)
        # The assembled prompt, once per turn: "what was this model actually
        # told?" is the first question any debugging session asks, and it was
        # the one thing the trace could not answer. The digest identifies the
        # exact prompt across turns even when the body is truncated in the
        # trace, so "did this user get the new persona?" is answerable without
        # TRACE_FULL_MESSAGES=1.
        self.tracer.emit(
            "prompt.built",
            system=system,
            system_sha=hashlib.sha256(system.encode()).hexdigest()[:12],
            system_chars=len(system),
            history_entries=len(self.history),
        )
        corrections = 0
        saved: list[str] = []
        executed: list[str] = []

        while True:
            with self.tracer.span("llm.generate"):
                response = self.provider.generate(
                    system=system,
                    contents=list(self.history),
                    tools=TOOL_DECLARATIONS,
                )
            self._record_usage(response)

            if not response.wants_tool:
                return TurnResult(
                    answer=response.text or "I do not have an answer for that.",
                    trace_id=self.tracer.trace_id,
                    saved_report_ids=tuple(saved),
                    sql_executed=tuple(executed),
                )

            # One model turn carrying every call of this step, in the order they
            # were returned. Splitting them into a turn each cost the second call
            # its thought_signature — Gemini only signs the first — and the next
            # request died on "400 Function call is missing a thought_signature",
            # after the queries had already run and been billed.
            self._append_function_calls(response.function_calls)

            results: list[tuple[str, dict[str, Any]]] = []
            pending: PendingDeletion | None = None
            stop: TurnResult | None = None

            for call in response.function_calls:
                self.tracer.metrics.tool_calls += 1
                self.tracer.emit("tool.call", name=call.name, args=call.args)

                # An unexpected failure inside a tool must still answer the
                # call it belongs to. Returning early without a response
                # leaves a malformed history behind that every later turn in
                # this conversation has to carry.
                answered = len(results)
                try:
                    if call.name == "run_analysis_sql":
                        outcome, ok = self._tool_run_sql(call.args)
                        results.append((call.name, outcome))
                        if ok:
                            executed.append(outcome["sql"])
                        elif outcome.get("retriable_by_model"):
                            corrections += 1
                            self.tracer.metrics.sql_self_corrections += 1
                            if corrections > MAX_SQL_CORRECTIONS and stop is None:
                                stop = self._give_up_on_sql(outcome, executed, saved)
                        elif stop is None:
                            # Nothing the model can write will fix a warehouse that
                            # is down, unreachable or refusing our credentials.
                            stop = self._warehouse_unavailable(outcome, executed, saved)

                    elif call.name == "describe_schema":
                        results.append(
                            (call.name,
                             {"schema": catalog.describe_schema(call.args.get("table"))})
                        )

                    elif call.name == "save_report":
                        results.append((call.name, self._tool_save_report(call.args, saved)))

                    elif call.name == "list_reports":
                        reports = self.store.search(
                            actor=self.scope.user_id, text=call.args.get("text")
                        )
                        results.append(
                            (call.name, {"reports": [r.summary() for r in reports]})
                        )

                    elif call.name == "propose_delete_reports":
                        pending, outcome = self._tool_propose_deletion(call.args)
                        results.append((call.name, outcome))

                    else:
                        results.append(
                            (call.name, {"error": f"unknown tool {call.name!r}"})
                        )
                except Exception as err:  # noqa: BLE001
                    self.tracer.emit(
                        "tool.error", name=call.name,
                        error_type=type(err).__name__, error=str(err),
                    )
                    del results[answered:]
                    results.append(
                        (call.name, {"error": f"{type(err).__name__}: {err}"})
                    )

            # Every call answered, in one user turn, before anything returns.
            # A call left without its response is a malformed history that the
            # next turn has to carry.
            self._append_function_results(results)

            if pending is not None:
                # The confirmation prompt IS the turn's answer; the model must
                # not narrate over it.
                return TurnResult(
                    answer=pending.prompt(),
                    trace_id=self.tracer.trace_id,
                    pending_deletion=pending if pending.targets else None,
                    saved_report_ids=tuple(saved),
                    sql_executed=tuple(executed),
                    status="awaiting_confirmation" if pending.targets else "ok",
                )
            if stop is not None:
                return stop

    # -- tools ------------------------------------------------------------

    def _tool_run_sql(self, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        sql = str(args.get("sql", ""))
        try:
            with self.tracer.span("sql.execute", purpose=args.get("purpose", "")):
                result = self.runner.execute(sql, self.scope, tracer=self.tracer)
        except QueryError as err:
            self.tracer.emit("sql.failed", kind=str(err.kind), error=err.message)
            return (
                {
                    "error": err.message,
                    "hint": err.hint or "",
                    "kind": str(err.kind),
                    "self_correctable": err.self_correctable,
                    "retriable_by_model": err.retriable_by_model,
                },
                False,
            )

        return (
            {
                "sql": result.sql,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "rows": result.to_markdown(),
                "note": (
                    "No rows matched. Say so; do not report this as zero revenue."
                    if result.is_empty
                    else ""
                ),
            },
            True,
        )

    def _tool_save_report(
        self, args: dict[str, Any], saved: list[str]
    ) -> dict[str, Any]:
        """Save a report, scrubbing it first.

        The final answer is scrubbed on the way to the screen, but a report is
        written straight to storage and read back days later — so without this
        the library was the one place a leaked email could settle permanently.
        """
        title, title_hits = scrub_text(str(args.get("title", "Untitled report")))
        body, body_hits = scrub_text(str(args.get("body", "")))

        # persona.yaml's report_sections, enforced rather than only requested:
        # a report without its action items is not the report the user asked
        # for, and the model can fix that in one more call.
        missing = missing_report_sections(body)
        if missing:
            headings = ", ".join(f'"{section_heading(s)}"' for s in missing)
            self.tracer.emit("report.rejected", missing=missing)
            return {
                "saved": False,
                "error": f"Not saved — missing required section(s): {headings}. "
                         "Add them as headings and call save_report again.",
            }
        raw_entities = args.get("entities") or []
        if isinstance(raw_entities, str):  # the model sometimes sends one string
            raw_entities = [raw_entities]
        entities = [scrub_text(str(e))[0] for e in raw_entities]

        if title_hits or body_hits:
            self.tracer.metrics.pii_redactions += len(title_hits) + len(body_hits)
            self.tracer.audit(
                "pii_redacted", where="saved_report", details=title_hits + body_hits
            )

        report = self.store.save(
            owner=self.scope.user_id,
            conversation_id=self.conversation_id,
            title=title,
            body=body,
            entities=entities,
        )
        saved.append(report.report_id)
        self.tracer.audit("report_saved", report_id=report.report_id)
        return {"saved": True, "report_id": report.report_id[:8], "title": report.title}

    def _tool_propose_deletion(
        self, args: dict[str, Any]
    ) -> tuple[PendingDeletion, dict[str, Any]]:
        pending = self.broker.propose_deletion(
            actor=self.scope.user_id,
            criteria=str(args.get("criteria", "the reports you named")),
            text=args.get("text"),
            conversation_id=(
                self.conversation_id if args.get("this_conversation") else None
            ),
            all_reports=bool(args.get("all_reports")),
        )
        if pending.selector == "none":
            # A deletion that named nothing to match is refused, not guessed.
            self.tracer.metrics.refusals += 1
        self.tracer.audit(
            "deletion_proposed",
            count=len(pending.targets),
            criteria=pending.criteria,
            selector=pending.selector,
        )
        return pending, {
            "status": "awaiting_user_confirmation",
            "matched": len(pending.targets),
            "criteria": pending.criteria,
            "note": "Nothing is deleted yet. The user confirms or cancels next.",
        }

    def _give_up_on_sql(
        self, outcome: dict[str, Any], executed: list[str], saved: list[str]
    ) -> TurnResult:
        """Stop the repair loop and say so, rather than burning the budget."""
        self.tracer.emit("sql.gave_up", attempts=MAX_SQL_CORRECTIONS)
        return TurnResult(
            answer=(
                "I could not build a working query for that after several "
                "attempts. Could you rephrase it, or narrow it to a single "
                "metric over a specific period? "
                f"(Last problem: {outcome.get('error', 'unknown')[:200]})"
            ),
            trace_id=self.tracer.trace_id,
            sql_executed=tuple(executed),
            saved_report_ids=tuple(saved),
            status="sql_unrecoverable",
        )

    def _warehouse_unavailable(
        self, outcome: dict[str, Any], executed: list[str], saved: list[str]
    ) -> TurnResult:
        """End the turn on a failure no rewrite can fix, and name it honestly.

        These used to go back to the model as just another failed tool call. It
        would rewrite the query, the rewrite would fail the same way, and the
        turn ended on the call budget telling the user to "narrow the question"
        — for an expired credential or a network outage. Eight model calls and
        eight BigQuery attempts to produce advice that could not possibly help.
        """
        kind = outcome.get("kind", "")
        self.tracer.emit("sql.unavailable", kind=kind)
        if kind == str(QueryErrorKind.PERMISSION):
            message = (
                "I cannot reach the data warehouse — this deployment's Google "
                "Cloud credentials are missing or were refused. That needs an "
                "operator, not a different question. Your saved reports are "
                "unaffected."
            )
            status = "warehouse_permission"
        elif kind in {str(QueryErrorKind.TRANSIENT), str(QueryErrorKind.TIMEOUT),
                      str(QueryErrorKind.CIRCUIT_OPEN)}:
            message = (
                "The data warehouse is not responding at the moment, so I "
                "cannot run the analysis. I have stopped retrying rather than "
                "queue up queries against it — please try again shortly."
            )
            status = "warehouse_unavailable"
        else:
            message = (
                "Something went wrong between me and the data warehouse that I "
                "cannot work around. Quote trace "
                f"{self.tracer.trace_id} to an operator."
            )
            status = "warehouse_error"
        return TurnResult(
            answer=message,
            trace_id=self.tracer.trace_id,
            sql_executed=tuple(executed),
            saved_report_ids=tuple(saved),
            status=status,
        )

    # -- helpers ----------------------------------------------------------

    def _degrade(self, message: str, *, status: str, error: Exception) -> TurnResult:
        self.tracer.metrics.llm_errors += 1
        self.tracer.emit(
            "turn.degraded", status=status, error_type=type(error).__name__,
            error=str(error),
        )
        return TurnResult(answer=message, trace_id=self.tracer.trace_id, status=status)

    def _record_usage(self, response: LLMResponse) -> None:
        self.tracer.metrics.llm_calls += 1
        self.tracer.metrics.prompt_tokens += response.prompt_tokens
        self.tracer.metrics.output_tokens += response.output_tokens
        if response.used_fallback:
            self.tracer.metrics.fallback_model_used = True
        self.tracer.emit(
            "llm.response",
            model=response.model,
            used_fallback=response.used_fallback,
            tool_calls=[c.name for c in response.function_calls],
            text=response.text,
        )

    def _append(self, role: str, text: str) -> None:
        if role == "user":
            self._trim()  # before a new turn, never in the middle of one
        self.history.append({"role": role, "parts": [{"text": text}]})

    def _append_function_calls(self, function_calls: Sequence[FunctionCall]) -> None:
        """Echo one model step back as one model turn, parts in order.

        Gemini returns parallel calls as several parts of a SINGLE content and
        signs only the first of them. Appending a turn per call therefore
        produced a second turn whose call had no thought_signature, which the
        API rejects outright.
        """
        parts: list[dict[str, Any]] = []
        for call in function_calls:
            part: dict[str, Any] = {
                "function_call": {"name": call.name, "args": call.args}
            }
            if call.thought_signature is not None:
                # Gemini 3.x rejects history whose functionCall parts have lost
                # their thought_signature, so it is echoed back as received.
                part["thought_signature"] = call.thought_signature
            parts.append(part)
        self.history.append({"role": "model", "parts": parts})

    def _append_function_results(
        self, results: Sequence[tuple[str, dict[str, Any]]]
    ) -> None:
        """One response part per call, in the same order, in one user turn."""
        parts: list[dict[str, Any]] = []
        for name, payload in results:
            clean = json.loads(json.dumps(payload, default=str))
            parts.append({"function_response": {"name": name, "response": clean}})
            # What the model was actually handed back — the other half of the
            # correspondence a trace has to be able to replay.
            self.tracer.emit("tool.result", name=name, response=clean)
        self.history.append({"role": "user", "parts": parts})

    def _trim(self) -> None:
        """Keep the last MAX_HISTORY_TURNS turns, cutting only where one starts.

        Bounded so a long session cannot grow the prompt (and its cost) without
        limit. It used to cut a fixed number of entries, which lands wherever
        the arithmetic says: after a mix of short and tool-heavy turns the
        history opened on a function_response whose call had been cut away —
        a tool result answering nothing.
        """
        starts = [
            index
            for index, content in enumerate(self.history)
            if content["role"] == "user" and "text" in content["parts"][0]
        ]
        keep = MAX_HISTORY_TURNS - 1  # leaving room for the turn about to start
        if len(starts) > keep:
            self.history = self.history[starts[len(starts) - keep]:] if keep else []
