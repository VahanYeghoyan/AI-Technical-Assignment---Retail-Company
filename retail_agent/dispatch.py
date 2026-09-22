"""Frontend-agnostic message routing.

Every frontend — the CLI REPL and the Streamlit UI — has to make the same three
decisions in the same order, and the order is a safety property rather than a
cosmetic one:

  1. A live deletion confirmation owns the next message. If "yes" were routed to
     the model like any other text, the model would answer it conversationally
     and the pending deletion would be lost — or worse, re-proposed and
     double-confirmed.
  2. Slash commands bypass the model, so /undo, /reports and /trace keep working
     during an LLM outage — which, with an exhausted quota, is exactly the state
     this prototype has to stay usable in.
  3. Everything else is a question for the agent.

This module makes those decisions and reports *what happened*. It renders
nothing and imports no UI library, so the CLI prints Rich panels and the web UI
draws chat bubbles from one identical routing pass. A second frontend that
re-implemented the routing would eventually drift from the first, and the
confirmation state machine is the last thing in this system that should exist in
two copies.

One turn can produce more than one outcome: declining a proposal with a message
that is also a new question ("no, show me Q2 instead") cancels the deletion and
then answers the question, so dispatch returns a list.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, ContextManager

from retail_agent.agent import Agent, TurnResult
from retail_agent.observability import read_events, summarise_turns
from retail_agent.reports import DeleteOutcome, Report

# Replies that end the turn once they have cancelled a pending deletion. A reply
# that is *only* a refusal has said everything it means to say; anything longer
# ("no, show me Q2 instead") is treated as a new question so the user is never
# trapped in a prompt. Deliberately narrower than the broker's negative set: the
# broker decides whether a proposal is cancelled, this decides whether there is
# anything left to answer.
_PURE_NEGATIVES = frozenset({"no", "n", "cancel", "stop", "abort"})

# Turns shown by /trace with no argument.
RECENT_TURNS = 15


class Kind(StrEnum):
    """What a dispatched message turned out to be."""

    EMPTY = "empty"
    DELETED = "deleted"
    REPROMPT = "reprompt"
    CANCELLED = "cancelled"
    QUIT = "quit"
    HELP = "help"
    REPORTS = "reports"
    UNDO = "undo"
    TRACE_TURNS = "trace_turns"
    TRACE_EVENTS = "trace_events"
    WHOAMI = "whoami"
    PERSONA = "persona"
    UNKNOWN_COMMAND = "unknown_command"
    ANSWER = "answer"


@dataclass(frozen=True)
class Outcome:
    """One renderable thing that happened, with the data a frontend needs.

    Deliberately a single record with optional payloads rather than a class per
    kind: frontends switch on `kind` and read the one or two fields that apply,
    which keeps both renderers flat.
    """

    kind: Kind
    text: str = ""
    delete_outcome: DeleteOutcome | None = None
    reports: tuple[Report, ...] = ()
    restored: tuple[Report, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    turns: tuple[dict[str, Any], ...] = ()
    trace_id: str = ""
    result: TurnResult | None = None
    # Snapshot of the turn's counters, taken before the next turn resets them.
    metrics: dict[str, Any] = field(default_factory=dict)


def dispatch(
    agent: Agent,
    text: str,
    *,
    progress: Callable[[], ContextManager[Any]] = nullcontext,
) -> list[Outcome]:
    """Route one user message. Never raises on user input.

    `progress` wraps the model call only — the CLI passes a console spinner, the
    web UI passes st.spinner — so a frontend can show that work is happening
    without this module knowing what a spinner is.
    """
    stripped = text.strip()
    if not stripped:
        return [Outcome(Kind.EMPTY)]

    outcomes: list[Outcome] = []

    # 1. A pending destructive action owns the next message.
    verdict = agent.broker.interpret(agent.scope.user_id, stripped)
    if verdict == "confirm":
        deleted = agent.broker.confirm(agent.scope.user_id)
        agent.tracer.audit(
            "reports_deleted",
            count=deleted.deleted_count,
            report_ids=[r.report_id for r in deleted.deleted],
        )
        agent.last_deleted = tuple(r.report_id for r in deleted.deleted)
        return [Outcome(Kind.DELETED, delete_outcome=deleted)]

    if verdict == "reprompt":
        # Still pending: the reply meant yes, but not in the form a bulk
        # delete requires. Nothing is deleted and nothing reaches the model.
        # (If it expired a moment ago, the reply is an ordinary message.)
        pending = agent.broker.pending_for(agent.scope.user_id)
        if pending is not None:
            return [Outcome(Kind.REPROMPT, text=pending.reprompt())]

    if verdict == "cancel":
        cancelled = agent.broker.cancel(agent.scope.user_id)
        agent.tracer.audit(
            "deletion_cancelled",
            count=len(cancelled.targets) if cancelled else 0,
            criteria=cancelled.criteria if cancelled else "",
        )
        outcomes.append(Outcome(Kind.CANCELLED))
        if stripped.lower() in _PURE_NEGATIVES:
            return outcomes
        # Otherwise fall through: the message was probably a new question.

    # 2. Slash commands work even when the model is unavailable.
    if stripped.startswith("/"):
        outcomes.append(_slash_command(agent, stripped))
        return outcomes

    # 3. Otherwise it is a question for the agent.
    with progress():
        result = agent.ask(stripped)
    outcomes.append(
        Outcome(
            Kind.ANSWER,
            result=result,
            # Read now: start_turn() resets these on the next question.
            metrics=agent.tracer.metrics.as_dict(),
        )
    )
    return outcomes


def _slash_command(agent: Agent, stripped: str) -> Outcome:
    command, _, argument = stripped[1:].partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command in {"quit", "exit", "q"}:
        return Outcome(Kind.QUIT)

    if command == "help":
        return Outcome(Kind.HELP)

    if command == "reports":
        return Outcome(
            Kind.REPORTS, reports=agent.store.list_for_user(agent.scope.user_id)
        )

    if command == "undo":
        restored = agent.store.restore(agent.last_deleted, actor=agent.scope.user_id)
        # A restore changes who can see what, exactly as a delete does, so it
        # belongs in the same audit stream rather than nowhere.
        agent.tracer.audit(
            "reports_restored",
            count=len(restored),
            report_ids=[r.report_id for r in restored],
        )
        return Outcome(Kind.UNDO, restored=restored)

    if command == "trace":
        if argument:
            return Outcome(
                Kind.TRACE_EVENTS,
                trace_id=argument,
                # Scoped to the caller. A trace holds the question, the SQL and
                # the answer, so an unfiltered lookup by id let one executive
                # read another's analysis — including the figures their own
                # entitlements are there to keep from them.
                events=tuple(
                    read_events(trace_id=argument, user_id=agent.scope.user_id)
                ),
            )
        turns = summarise_turns(read_events(conversation_id=agent.conversation_id))
        return Outcome(Kind.TRACE_TURNS, turns=tuple(turns))

    if command == "whoami":
        return Outcome(Kind.WHOAMI)

    if command == "persona":
        return Outcome(Kind.PERSONA)

    return Outcome(Kind.UNKNOWN_COMMAND, text=command)
