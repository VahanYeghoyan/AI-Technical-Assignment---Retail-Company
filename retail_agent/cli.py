"""Command-line chat interface.

Run with:  python -m retail_agent.cli --user maya

The one piece of real logic in here is the order of checks in handle_input():
a live deletion confirmation is intercepted BEFORE the model is consulted. If
"yes" were routed to the model like any other message, the model would answer it
conversationally and the pending deletion would be lost — or worse, re-proposed
and double-confirmed. Confirmation is a state machine in code, not a thing the
model is trusted to remember.

Slash commands are deliberately outside the model too: /undo, /reports and
/trace must work even when the LLM is down, which — with an exhausted quota —
is exactly the state this prototype has to stay usable in.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from retail_agent.agent import Agent
from retail_agent.bigquery_runner import BigQueryRunner
from retail_agent.confirmation import ConfirmationBroker
from retail_agent.llm import LLMConfigError, build_provider
from retail_agent.observability import Tracer, read_events, summarise_turns
from retail_agent.prompt import persona_version
from retail_agent.reports import ReportStore
from retail_agent.safety.scope import UnknownUserError, get_scope, load_scopes

console = Console()

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


def _render_reports(agent: Agent) -> None:
    reports = agent.store.list_for_user(agent.scope.user_id)
    if not reports:
        console.print("[dim]No saved reports yet.[/dim]")
        return
    table = Table(title="Saved reports", header_style="bold")
    table.add_column("id")
    table.add_column("title")
    table.add_column("created")
    for report in reports:
        table.add_row(report.report_id[:8], report.title, report.created_at[:10])
    console.print(table)


def _render_trace(agent: Agent, trace_id: str | None) -> None:
    if trace_id:
        events = read_events(trace_id=trace_id)
        if not events:
            console.print(f"[yellow]No events for trace {trace_id}.[/yellow]")
            return
        table = Table(title=f"Trace {trace_id}", header_style="bold")
        table.add_column("time")
        table.add_column("event")
        table.add_column("detail", overflow="fold")
        for event in events:
            detail = {
                k: v
                for k, v in event.items()
                if k not in {"ts", "trace_id", "span_id", "parent_span_id", "event",
                             "user_id", "conversation_id"}
            }
            table.add_row(event["ts"][11:23], event["event"], str(detail)[:160])
        console.print(table)
        return

    turns = summarise_turns(read_events(conversation_id=agent.conversation_id))
    if not turns:
        console.print("[dim]No turns recorded yet.[/dim]")
        return
    table = Table(title="Recent turns", header_style="bold")
    table.add_column("trace id")
    table.add_column("question", overflow="fold")
    table.add_column("status")
    table.add_column("llm")
    table.add_column("sql")
    for turn in turns[-15:]:
        metrics = turn.get("metrics", {})
        table.add_row(
            turn["trace_id"],
            str(turn.get("question", ""))[:60],
            turn.get("status", "?"),
            str(metrics.get("llm_calls", "")),
            str(metrics.get("sql_attempts", "")),
        )
    console.print(table)
    console.print("[dim]/trace <id> for the full event stream of one turn.[/dim]")


def handle_input(agent: Agent, text: str) -> bool:
    """Process one line. Returns False to exit.

    Order matters: confirmation first, then slash commands, then the model.
    """
    stripped = text.strip()
    if not stripped:
        return True

    # 1. A pending destructive action owns the next message.
    verdict = agent.broker.interpret(agent.scope.user_id, stripped)
    if verdict == "confirm":
        outcome = agent.broker.confirm(agent.scope.user_id)
        agent.tracer.audit(
            "reports_deleted",
            count=outcome.deleted_count,
            report_ids=[r.report_id for r in outcome.deleted],
        )
        console.print(
            f"[green]Deleted {outcome.deleted_count} report(s).[/green] "
            "[dim]/undo restores them.[/dim]"
        )
        agent.last_deleted = tuple(r.report_id for r in outcome.deleted)  # type: ignore[attr-defined]
        return True
    if verdict == "cancel":
        agent.broker.cancel(agent.scope.user_id)
        console.print("[yellow]Cancelled — nothing was deleted.[/yellow]")
        # Deliberately fall through: the message was probably a new question.
        if stripped.lower() in {"no", "n", "cancel", "stop", "abort"}:
            return True

    # 2. Slash commands work even when the model is unavailable.
    if stripped.startswith("/"):
        command, _, argument = stripped[1:].partition(" ")
        command = command.lower()
        if command in {"quit", "exit", "q"}:
            return False
        if command == "help":
            console.print(HELP)
        elif command == "reports":
            _render_reports(agent)
        elif command == "undo":
            ids = getattr(agent, "last_deleted", ())
            restored = agent.store.restore(ids, actor=agent.scope.user_id)
            console.print(
                f"[green]Restored {len(restored)} report(s).[/green]"
                if restored
                else "[yellow]Nothing to restore.[/yellow]"
            )
        elif command == "trace":
            _render_trace(agent, argument.strip() or None)
        elif command == "whoami":
            console.print(
                Panel(
                    f"[bold]{agent.scope.display_name or agent.scope.user_id}[/bold]"
                    f"{f' — {agent.scope.title}' if agent.scope.title else ''}\n"
                    f"Scope: {agent.scope.describe()}\n"
                    f"Persona version: {persona_version()}\n"
                    f"Conversation: {agent.conversation_id}",
                    title="whoami",
                )
            )
        elif command == "persona":
            console.print(
                f"[bold]Persona v{persona_version()}[/bold] "
                "[dim](re-read from config/persona.yaml on every turn)[/dim]"
            )
        else:
            console.print(f"[yellow]Unknown command /{command}. Try /help.[/yellow]")
        return True

    # 3. Otherwise it is a question for the agent.
    with console.status("[dim]analysing…[/dim]"):
        result = agent.ask(stripped)

    console.print()
    console.print(Markdown(result.answer))
    if result.saved_report_ids:
        console.print(
            f"[green]Saved report {result.saved_report_ids[0][:8]}[/green] "
            "[dim](/reports to list)[/dim]"
        )
    if result.status not in {"ok", "awaiting_confirmation"}:
        console.print(
            f"[dim]status: {result.status} · trace {result.trace_id} "
            f"(/trace {result.trace_id})[/dim]"
        )
    console.print()
    return True


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
            console.print(f"  {user_id:8} {scope.title or '':34} {scope.describe()}")
        return 0

    try:
        agent = build_agent(args.user, conversation_id=uuid.uuid4().hex[:12])
    except UnknownUserError as err:
        console.print(f"[red]{err}[/red]")
        return 2
    except LLMConfigError as err:
        console.print(f"[red]{err}[/red]")
        return 2

    console.print(Panel(BANNER, border_style="dim"))
    console.print(
        f"[dim]Signed in as {agent.scope.display_name or args.user} · "
        f"scope: {agent.scope.describe()} · persona v{persona_version()}[/dim]\n"
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
            console.print(f"[red]Unexpected error:[/red] {err}")


if __name__ == "__main__":
    sys.exit(main())
