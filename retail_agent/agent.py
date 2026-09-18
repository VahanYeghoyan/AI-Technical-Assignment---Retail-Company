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

import json
from dataclasses import dataclass, field
from typing import Any

from retail_agent import catalog
from retail_agent.bigquery_runner import BigQueryRunner, QueryError, QueryErrorKind
from retail_agent.confirmation import ConfirmationBroker, PendingDeletion
from retail_agent.llm import (
    LLMBudgetError,
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMQuotaError,
    LLMResponse,
)
from retail_agent.observability import Tracer
from retail_agent.prompt import build_system_prompt, persona_version
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
            "user asks for a report. Include action items."
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
            "does NOT delete anything. Use 'text' to match report contents, or "
            "this_conversation=true for reports made in this conversation."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Match reports mentioning this."},
                "this_conversation": {"type": "boolean"},
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
                "cannot analyse anything right now. This needs credits topped up "
                "on the API key — retrying will not help.",
                status="quota_exhausted",
                error=err,
            )
        except LLMConfigError as err:
            result = self._degrade(
                "The language model is misconfigured for this deployment, so I "
                "cannot answer right now. An operator needs to check the API key "
                "and model settings.",
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
        corrections = 0
        saved: list[str] = []
        executed: list[str] = []
        pending: PendingDeletion | None = None

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
                    pending_deletion=pending,
                    saved_report_ids=tuple(saved),
                    sql_executed=tuple(executed),
                )

            for call in response.function_calls:
                self.tracer.metrics.tool_calls += 1
                self.tracer.emit("tool.call", name=call.name, args=call.args)
                self._append_function_call(call.name, call.args)

                if call.name == "run_analysis_sql":
                    outcome, ok = self._tool_run_sql(call.args)
                    if ok:
                        executed.append(outcome["sql"])
                    elif outcome.get("self_correctable"):
                        corrections += 1
                        self.tracer.metrics.sql_self_corrections += 1
                        if corrections > MAX_SQL_CORRECTIONS:
                            return self._give_up_on_sql(outcome, executed, saved)
                    self._append_function_result(call.name, outcome)

                elif call.name == "describe_schema":
                    self._append_function_result(
                        call.name,
                        {"schema": catalog.describe_schema(call.args.get("table"))},
                    )

                elif call.name == "save_report":
                    report = self.store.save(
                        owner=self.scope.user_id,
                        conversation_id=self.conversation_id,
                        title=str(call.args.get("title", "Untitled report")),
                        body=str(call.args.get("body", "")),
                        entities=[str(e) for e in call.args.get("entities", [])],
                    )
                    saved.append(report.report_id)
                    self.tracer.audit("report_saved", report_id=report.report_id)
                    self._append_function_result(
                        call.name,
                        {"saved": True, "report_id": report.report_id[:8],
                         "title": report.title},
                    )

                elif call.name == "list_reports":
                    reports = self.store.search(
                        actor=self.scope.user_id, text=call.args.get("text")
                    )
                    self._append_function_result(
                        call.name, {"reports": [r.summary() for r in reports]}
                    )

                elif call.name == "propose_delete_reports":
                    pending = self.broker.propose_deletion(
                        actor=self.scope.user_id,
                        criteria=str(call.args.get("criteria", "the reports you named")),
                        text=call.args.get("text"),
                        conversation_id=(
                            self.conversation_id
                            if call.args.get("this_conversation")
                            else None
                        ),
                    )
                    self.tracer.audit(
                        "deletion_proposed",
                        count=len(pending.targets),
                        criteria=pending.criteria,
                    )
                    # Returning early: the confirmation prompt IS the turn's
                    # answer, and the model must not narrate over it.
                    return TurnResult(
                        answer=pending.prompt(),
                        trace_id=self.tracer.trace_id,
                        pending_deletion=pending if pending.targets else None,
                        saved_report_ids=tuple(saved),
                        sql_executed=tuple(executed),
                        status="awaiting_confirmation" if pending.targets else "ok",
                    )

                else:
                    self._append_function_result(
                        call.name, {"error": f"unknown tool {call.name!r}"}
                    )

    # -- tools ------------------------------------------------------------

    def _tool_run_sql(self, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        sql = str(args.get("sql", ""))
        try:
            with self.tracer.span("sql.execute", purpose=args.get("purpose", "")):
                result = self.runner.execute(sql, self.scope, tracer=self.tracer)
        except QueryError as err:
            self.tracer.emit("sql.failed", kind=str(err.kind), error=err.message)
            payload = {
                "error": err.message,
                "hint": err.hint or "",
                "self_correctable": err.self_correctable,
            }
            if err.kind is QueryErrorKind.COST:
                payload["hint"] = err.hint
            return payload, False

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
        self.history.append({"role": role, "parts": [{"text": text}]})
        self._trim()

    def _append_function_call(self, name: str, args: dict[str, Any]) -> None:
        self.history.append(
            {"role": "model", "parts": [{"function_call": {"name": name, "args": args}}]}
        )

    def _append_function_result(self, name: str, payload: dict[str, Any]) -> None:
        self.history.append(
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": name,
                            "response": json.loads(json.dumps(payload, default=str)),
                        }
                    }
                ],
            }
        )
        self._trim()

    def _trim(self) -> None:
        # Keep the transcript bounded so a long session cannot grow the prompt
        # (and its cost) without limit.
        limit = MAX_HISTORY_TURNS * 4
        if len(self.history) > limit:
            self.history = self.history[-limit:]
