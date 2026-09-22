"""Streamlit front end — the CLI's behaviour, in a browser.

Run with:  streamlit run streamlit_app.py

This is a *renderer*, exactly like retail_agent/cli.py. Both call
retail_agent.dispatch.dispatch() and draw whatever it reports, so the rules that
matter are enforced in one place for both:

  * a pending deletion still intercepts the next message before the model,
  * slash commands still bypass the model, so /reports, /undo and /trace keep
    working during an LLM outage,
  * and the agent still never deletes anything itself.

One deliberate omission: there are no Confirm / Cancel buttons. Above three
reports the confirmation flow requires the user to type the count ("delete 7"),
and a button would turn that back into the reflexive single click the rule
exists to prevent. Confirmation is typed here for the same reason it is typed in
the CLI.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from retail_agent.agent import Agent
from retail_agent.cli import build_agent
from retail_agent.dispatch import RECENT_TURNS, Kind, Outcome, dispatch
from retail_agent.llm import LLMConfigError
from retail_agent.prompt import persona_version
from retail_agent.safety.scope import UnknownUserError, load_scopes

load_dotenv()

st.set_page_config(
    page_title="Retail analysis assistant",
    page_icon="📊",
    layout="wide",
)

HELP_MARKDOWN = """
| Command | Does |
|---|---|
| `/reports` | list your saved reports |
| `/reports deleted` | deleted reports you can still restore (30 days) |
| `/undo [id]` | restore the reports just deleted, or one by its id |
| `/trace [id]` | recent turns, or the full event stream for one turn |
| `/whoami` | identity, data scope, persona version |
| `/persona` | show the live persona version |
| `/help` | this message |

Everything else is a question for the assistant.
"""


# -- session ---------------------------------------------------------------


def get_agent(user_id: str) -> Agent:
    """Build the agent once per signed-in user and keep it across reruns.

    Streamlit re-executes this script top to bottom on every interaction, so the
    agent — which owns the conversation history, the confirmation broker's
    pending state and the tracer — has to live in session_state or every message
    would start a fresh conversation.
    """
    if st.session_state.get("user_id") != user_id or "agent" not in st.session_state:
        st.session_state.agent = build_agent(
            user_id, conversation_id=uuid.uuid4().hex[:12]
        )
        st.session_state.user_id = user_id
        st.session_state.transcript = []
    return st.session_state.agent


def safe_dispatch(agent: Agent, text: str) -> list[Outcome]:
    """Route one message. Like the REPL, the UI must survive anything.

    agent.ask() already absorbs its own failures, so what reaches here is the
    unexpected kind — a confirmation that expired between render and submit, a
    store error — and none of it should replace the page with a traceback.
    """
    return dispatch(agent, text, progress=lambda: st.spinner("Analysing…"))


# -- rendering -------------------------------------------------------------


def render_answer(outcome: Outcome) -> None:
    result = outcome.result
    if result is None:
        return

    st.markdown(result.answer)

    if result.saved_report_ids:
        st.success(
            f"Saved report `{result.saved_report_ids[0][:8]}` — `/reports` to list."
        )

    metrics = outcome.metrics
    summary = (
        f"status **{result.status}** · llm {metrics.get('llm_calls', 0)}"
        f" · sql {metrics.get('sql_attempts', 0)}"
        f" · corrections {metrics.get('sql_self_corrections', 0)}"
        f" · tok {metrics.get('prompt_tokens', 0):,}→{metrics.get('output_tokens', 0):,}"
        + (
            f" · think {metrics['thinking_tokens']:,}"
            if metrics.get("thinking_tokens")
            else ""
        )
        + f" · bytes billed {metrics.get('bq_bytes_billed', 0):,}"
        f" · trace `{result.trace_id}`"
    )
    st.caption(summary)

    if result.sql_executed:
        with st.expander(f"SQL executed ({len(result.sql_executed)})"):
            # The scope-rewritten SQL, i.e. what actually reached BigQuery —
            # not what the model wrote.
            for statement in result.sql_executed:
                st.code(statement, language="sql")

    if metrics:
        with st.expander("Turn metrics"):
            st.json(metrics)

    if result.status not in {"ok", "awaiting_confirmation"}:
        st.info(f"Run `/trace {result.trace_id}` to replay this turn.")


def render_outcome(agent: Agent, outcome: Outcome) -> None:
    """Draw one dispatched outcome."""
    kind = outcome.kind

    if kind is Kind.EMPTY:
        return

    if kind is Kind.DELETED:
        deleted = outcome.delete_outcome
        assert deleted is not None
        st.success(
            f"Deleted {deleted.deleted_count} report(s). `/undo` restores them."
        )
        if deleted.skipped_not_owned:
            st.warning(
                f"Skipped {len(deleted.skipped_not_owned)} report(s) belonging to "
                "someone else."
            )

    elif kind is Kind.REPROMPT:
        st.warning(outcome.text)

    elif kind is Kind.CANCELLED:
        st.warning("Cancelled — nothing was deleted.")

    elif kind is Kind.QUIT:
        st.info(
            "`/quit` ends the CLI session. In the browser, close the tab or start "
            "a new conversation from the sidebar."
        )

    elif kind is Kind.HELP:
        st.markdown(HELP_MARKDOWN)

    elif kind is Kind.REPORTS:
        deleted = outcome.text == "deleted"
        if not outcome.reports:
            st.caption(
                "No deleted reports to restore." if deleted else "No saved reports yet."
            )
        else:
            st.dataframe(
                [
                    {
                        "id": r.report_id[:8],
                        "title": r.title,
                        ("deleted" if deleted else "created"): (
                            r.deleted_at if deleted else r.created_at
                        )[:10],
                    }
                    for r in outcome.reports
                ],
                hide_index=True,
                width="stretch",
            )
            if deleted:
                st.caption("`/undo <id>` restores one.")

    elif kind is Kind.UNDO:
        if outcome.restored:
            st.success(f"Restored {len(outcome.restored)} report(s).")
        elif outcome.text:
            st.warning(
                f"No deleted report of yours matches `{outcome.text}` — "
                "`/reports deleted` lists what can be restored."
            )
        else:
            st.warning(
                "Nothing deleted in this session to restore — `/reports deleted` "
                "lists older deletions."
            )

    elif kind is Kind.TRACE_TURNS:
        if not outcome.turns:
            st.caption("No turns recorded yet.")
        else:
            st.dataframe(
                [
                    {
                        "trace id": t["trace_id"],
                        "question": str(t.get("question", ""))[:80],
                        "status": t.get("status", "?"),
                        "llm": t.get("metrics", {}).get("llm_calls", ""),
                        "sql": t.get("metrics", {}).get("sql_attempts", ""),
                    }
                    for t in outcome.turns[-RECENT_TURNS:]
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption("`/trace <id>` for the full event stream of one turn.")

    elif kind is Kind.TRACE_EVENTS:
        if not outcome.events:
            st.warning(f"No events for trace `{outcome.trace_id}`.")
        else:
            noise = {
                "ts", "trace_id", "span_id", "parent_span_id", "event",
                "user_id", "conversation_id",
            }
            st.dataframe(
                [
                    {
                        "time": e["ts"][11:23],
                        "event": e["event"],
                        "detail": str(
                            {k: v for k, v in e.items() if k not in noise}
                        )[:200],
                    }
                    for e in outcome.events
                ],
                hide_index=True,
                width="stretch",
            )
            with st.expander("Raw events"):
                st.json(list(outcome.events))

    elif kind is Kind.WHOAMI:
        scope = agent.scope
        st.markdown(
            f"**{scope.display_name or scope.user_id}**"
            f"{f' — {scope.title}' if scope.title else ''}\n\n"
            f"- Scope: {scope.describe()}\n"
            f"- Persona version: {persona_version()}\n"
            f"- Conversation: `{agent.conversation_id}`"
        )

    elif kind is Kind.PERSONA:
        st.markdown(
            f"**Persona v{persona_version()}** — re-read from "
            "`config/persona.yaml` on every turn."
        )

    elif kind is Kind.UNKNOWN_COMMAND:
        st.warning(f"Unknown command `/{outcome.text}`. Try `/help`.")

    elif kind is Kind.ANSWER:
        render_answer(outcome)


def render_entry(agent: Agent, entry: dict[str, Any]) -> None:
    """Draw one transcript entry (a user message or an assistant response)."""
    with st.chat_message(entry["role"]):
        if "text" in entry:
            st.markdown(entry["text"])
        if "error" in entry:
            st.error(f"Unexpected error: {entry['error']}")
        for outcome in entry.get("outcomes", ()):
            render_outcome(agent, outcome)


# -- sidebar ---------------------------------------------------------------


def render_sidebar(scopes: dict[str, Any], user_id: str) -> str:
    with st.sidebar:
        st.subheader("Session")
        chosen = st.selectbox(
            "Signed in as",
            options=list(scopes),
            index=list(scopes).index(user_id) if user_id in scopes else 0,
            format_func=lambda uid: (
                f"{scopes[uid].display_name or uid} — {scopes[uid].title}"
                if scopes[uid].title
                else (scopes[uid].display_name or uid)
            ),
            help="There is no real authentication here; this selects an "
            "entitlement persona from config/entitlements.yaml.",
        )

        scope = scopes[chosen]
        st.caption(f"**Data scope:** {scope.describe()}")
        st.caption(
            "Every generated query is rewritten to this scope before it runs."
        )

        st.divider()
        st.subheader("Runtime")
        provider = os.getenv("LLM_PROVIDER", "gemini")
        st.caption(f"**Model backend:** `{provider}`")
        if provider != "stub":
            st.caption(f"**Model:** `{os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')}`")
        # Re-read on every rerun, so editing config/persona.yaml shows up here
        # as immediately as it shows up in the next answer.
        st.caption(f"**Persona:** v{persona_version()}")
        if "agent" in st.session_state:
            st.caption(f"**Conversation:** `{st.session_state.agent.conversation_id}`")

        st.divider()
        if st.button("New conversation", width="stretch"):
            st.session_state.pop("agent", None)
            st.session_state.transcript = []
            st.rerun()

        with st.expander("Commands"):
            st.markdown(HELP_MARKDOWN)

    return chosen


# -- page ------------------------------------------------------------------


def main() -> None:
    st.title("📊 Retail analysis assistant")

    try:
        scopes = load_scopes()
    except Exception as err:  # noqa: BLE001 - config problems must be readable
        st.error(f"Could not load config/entitlements.yaml: {err}")
        return
    if not scopes:
        st.error("No users defined in config/entitlements.yaml.")
        return

    default_user = os.getenv("AGENT_USER", "maya")
    user_id = render_sidebar(
        scopes, st.session_state.get("user_id", default_user)
    )

    try:
        agent = get_agent(user_id)
    except UnknownUserError as err:
        st.error(str(err))
        return
    except LLMConfigError as err:
        st.error(str(err))
        st.info(
            "Set `LLM_PROVIDER=stub` in `.env` to run the whole agent offline, "
            "with no credentials and no quota."
        )
        return

    st.caption(
        f"Signed in as **{agent.scope.display_name or user_id}** · "
        f"scope: {agent.scope.describe()} · persona v{persona_version()}"
    )

    for entry in st.session_state.transcript:
        render_entry(agent, entry)

    # Shown after the transcript so it sits directly above the input box. Calling
    # pending_for() also expires a stale proposal, so a confirmation the user
    # left sitting for five minutes disappears from the UI rather than looking
    # live.
    pending = agent.broker.pending_for(agent.scope.user_id)
    if pending is not None:
        expected = (
            f"`delete {len(pending.targets)}`"
            if pending.requires_count
            else "`yes`"
        )
        st.warning(
            f"**Awaiting confirmation** — {len(pending.targets)} report(s) "
            f"matching *{pending.criteria}*. Type {expected} to confirm; "
            "anything else cancels."
        )

    prompt = st.chat_input("Ask about sales, products, customers or performance…")
    if not prompt:
        return

    st.session_state.transcript.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        outcomes = safe_dispatch(agent, prompt)
        entry: dict[str, Any] = {"role": "assistant", "outcomes": outcomes}
    except Exception as err:  # noqa: BLE001 - the UI must survive anything
        entry = {"role": "assistant", "error": str(err)}

    st.session_state.transcript.append(entry)
    st.rerun()


main()
