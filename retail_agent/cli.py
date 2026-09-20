"""Command-line chat interface.

Run with:  python -m retail_agent.cli --user maya

This module is a renderer. The decision of what a message *means* — a live
deletion confirmation, a slash command, or a question for the model — lives in
retail_agent/dispatch.py, because the Streamlit UI has to make exactly the same
decisions in exactly the same order and a second copy of that state machine
would eventually drift from this one. See that module for why the order matters.

What stays here is Rich: panels, tables, markdown and the input loop.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from retail_agent.agent import Agent
from retail_agent.bigquery_runner import BigQueryRunner
from retail_agent.confirmation import ConfirmationBroker
from retail_agent.dispatch import RECENT_TURNS, Kind, Outcome, dispatch
from retail_agent.llm import LLMConfigError, build_provider
from retail_agent.observability import Tracer
from retail_agent.prompt import persona_version
from retail_agent.reports import ReportStore
from retail_agent.safety.scope import UnknownUserError, get_scope, load_scopes

console = Console()


def _safe(value: object) -> str:
    """Render untrusted text as itself, not as Rich markup.

    Report titles, questions and error strings all reach a Table cell or a
    print(), and Rich parses square brackets in them as style tags. A title like
    "Q1 [/b] review" raised MarkupError out of /reports — and the REPL's own
    handler then printed the same brackets inside the error text, raising again
    from the except block and taking the process down. A renderer must not be
    able to crash on the text it is asked to render.
    """
    return escape(str(value))


BANNER = """\
[bold]Retail analysis assistant[/bold]
Ask about sales, products, customers or performance. Type /help for commands.
"""

HELP = """\
[bold]Commands[/bold]
  /reports            list your saved reports
  /undo               restore the reports deleted most recently
  /trace [id]         recent turns, or the full event trace for one turn
  /whoami             your identity, scope and the active persona version
  /persona            reload and show the current persona (edit config/persona.yaml)
  /help               this message
  /quit               leave

Everything else is a question for the assistant.
"""


def build_agent(user_id: str, *, conversation_id: str) -> Agent:
    """Wire the object graph for one session."""
    scope = get_scope(user_id)
    store = ReportStore()
    tracer = Tracer(user_id=user_id, conversation_id=conversation_id)
    return Agent(
        provider=build_provider(),
        runner=BigQueryRunner(),
        store=store,
        broker=ConfirmationBroker(store=store),
        scope=scope,
        tracer=tracer,
        conversation_id=conversation_id,
    )


# -- rendering ------------------------------------------------------------


def _render_reports(reports: tuple) -> None:
    if not reports:
        console.print("[dim]No saved reports yet.[/dim]")
        return
    table = Table(title="Saved reports", header_style="bold")
    table.add_column("id")
    table.add_column("title")
    table.add_column("created")
    for report in reports:
        table.add_row(
            report.report_id[:8], _safe(report.title), report.created_at[:10]
        )
    console.print(table)


def _render_trace_events(outcome: Outcome) -> None:
    if not outcome.events:
        console.print(f"[yellow]No events for trace {outcome.trace_id}.[/yellow]")
        return
    table = Table(title=f"Trace {outcome.trace_id}", header_style="bold")
    table.add_column("time")
    table.add_column("event")
    table.add_column("detail", overflow="fold")
    for event in outcome.events:
        detail = {
            k: v
            for k, v in event.items()
            if k not in {"ts", "trace_id", "span_id", "parent_span_id", "event",
                         "user_id", "conversation_id"}
        }
        table.add_row(
            event["ts"][11:23], _safe(event["event"]), _safe(str(detail)[:160])
        )
    console.print(table)


def _render_trace_turns(outcome: Outcome) -> None:
    if not outcome.turns:
        console.print("[dim]No turns recorded yet.[/dim]")
        return
    table = Table(title="Recent turns", header_style="bold")
    table.add_column("trace id")
    table.add_column("question", overflow="fold")
    table.add_column("status")
    table.add_column("llm")
    table.add_column("sql")
    for turn in outcome.turns[-RECENT_TURNS:]:
        metrics = turn.get("metrics", {})
        table.add_row(
            turn["trace_id"],
            _safe(str(turn.get("question", ""))[:60]),
            _safe(turn.get("status", "?")),
            str(metrics.get("llm_calls", "")),
            str(metrics.get("sql_attempts", "")),
        )
    console.print(table)
    console.print("[dim]/trace <id> for the full event stream of one turn.[/dim]")


def _render_answer(outcome: Outcome) -> None:
    result = outcome.result
    assert result is not None
    console.print()
    console.print(Markdown(result.answer))
    if result.saved_report_ids:
        console.print(
            f"[green]Saved report {result.saved_report_ids[0][:8]}[/green] "
            "[dim](/reports to list)[/dim]"
        )
    if result.status not in {"ok", "awaiting_confirmation"}:
        console.print(
            f"[dim]status: {_safe(result.status)} · trace {result.trace_id} "
            f"(/trace {result.trace_id})[/dim]"
        )
    console.print()


def render(agent: Agent, outcome: Outcome) -> None:
    """Print one dispatched outcome."""
    if outcome.kind is Kind.DELETED:
        deleted = outcome.delete_outcome
        assert deleted is not None
        console.print(
            f"[green]Deleted {deleted.deleted_count} report(s).[/green] "
            "[dim]/undo restores them.[/dim]"
        )
    elif outcome.kind is Kind.CANCELLED:
        console.print("[yellow]Cancelled — nothing was deleted.[/yellow]")
    elif outcome.kind is Kind.HELP:
        console.print(HELP)
    elif outcome.kind is Kind.REPORTS:
        _render_reports(outcome.reports)
    elif outcome.kind is Kind.UNDO:
        console.print(
            f"[green]Restored {len(outcome.restored)} report(s).[/green]"
            if outcome.restored
            else "[yellow]Nothing to restore.[/yellow]"
        )
    elif outcome.kind is Kind.TRACE_EVENTS:
        _render_trace_events(outcome)
    elif outcome.kind is Kind.TRACE_TURNS:
        _render_trace_turns(outcome)
    elif outcome.kind is Kind.WHOAMI:
        console.print(
            Panel(
                f"[bold]{_safe(agent.scope.display_name or agent.scope.user_id)}"
                "[/bold]"
                f"{f' — {_safe(agent.scope.title)}' if agent.scope.title else ''}\n"
                f"Scope: {_safe(agent.scope.describe())}\n"
                f"Persona version: {persona_version()}\n"
                f"Conversation: {agent.conversation_id}",
                title="whoami",
            )
        )
    elif outcome.kind is Kind.PERSONA:
        console.print(
            f"[bold]Persona v{persona_version()}[/bold] "
            "[dim](re-read from config/persona.yaml on every turn)[/dim]"
        )
    elif outcome.kind is Kind.UNKNOWN_COMMAND:
        console.print(
            f"[yellow]Unknown command /{_safe(outcome.text)}. Try /help.[/yellow]"
        )
    elif outcome.kind is Kind.ANSWER:
        _render_answer(outcome)


def handle_input(agent: Agent, text: str) -> bool:
    """Process one line. Returns False to exit."""
    keep_going = True
    for outcome in dispatch(
        agent, text, progress=lambda: console.status("[dim]analysing…[/dim]")
    ):
        if outcome.kind is Kind.QUIT:
            keep_going = False
            continue
        render(agent, outcome)
    return keep_going


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retail analysis chat assistant")
    parser.add_argument(
        "--user",
        default=os.getenv("AGENT_USER", "maya"),
        help="which executive to sign in as (see config/entitlements.yaml)",
    )
    parser.add_argument("--list-users", action="store_true", help="show known users")
    args = parser.parse_args(argv)

    if args.list_users:
        for user_id, scope in load_scopes().items():
            console.print(
                f"  {user_id:8} {_safe(scope.title or ''):34} {_safe(scope.describe())}"
            )
        return 0

    try:
        agent = build_agent(args.user, conversation_id=uuid.uuid4().hex[:12])
    except UnknownUserError as err:
        console.print(f"[red]{_safe(err)}[/red]")
        return 2
    except LLMConfigError as err:
        console.print(f"[red]{_safe(err)}[/red]")
        return 2

    console.print(Panel(BANNER, border_style="dim"))
    console.print(
        f"[dim]Signed in as {_safe(agent.scope.display_name or args.user)} · "
        f"scope: {_safe(agent.scope.describe())} · "
        f"persona v{_safe(persona_version())}[/dim]\n"
    )

    while True:
        try:
            text = console.input("[bold cyan]›[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return 0
        try:
            if not handle_input(agent, text):
                console.print("[dim]Bye.[/dim]")
                return 0
        except Exception as err:  # noqa: BLE001 - the REPL must survive anything
            console.print(f"[red]Unexpected error:[/red] {_safe(err)}")


if __name__ == "__main__":
    sys.exit(main())
