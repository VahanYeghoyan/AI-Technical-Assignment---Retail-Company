"""The web UI, driven headlessly.

streamlit_app.py is a renderer over retail_agent.dispatch, so what these tests
check is not layout but that the browser path keeps the guarantees the CLI is
held to: slash commands never reach the model, a pending deletion intercepts the
next message before the model sees it, and nothing here needs credentials or
quota to run.

Streamlit is an optional extra (requirements-ui.txt), so this module skips
rather than fails when it is absent — `pip install -r requirements.txt` followed
by `pytest` must stay green for someone who only wants the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip(
    "streamlit",
    reason="optional web UI — pip install -r requirements-ui.txt",
)

from google.api_core import exceptions as gexc  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from retail_agent.bigquery_runner import BigQueryRunner  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class _NoWarehouse:
    def query(self, *_args, **_kwargs):
        raise gexc.Forbidden("no BigQuery credentials in the test suite")


def _texts(elements) -> list[str]:
    """String values of a group of rendered elements."""
    return [e.value for e in elements if isinstance(getattr(e, "value", None), str)]


@pytest.fixture
def app(tmp_path, monkeypatch):
    """The app rendered as maya, offline, against a throwaway store and sink.

    load_dotenv() does not override variables that are already set, so these win
    over whatever a developer has in .env — the suite must not touch the real
    reports database or reach a real model.
    """
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("REPORTS_DB_PATH", str(tmp_path / "reports.db"))
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("AGENT_USER", "maya")
    # The offline demo answers a data question with a real query, so BigQuery
    # is replaced by a warehouse that refuses — what a machine with no
    # credentials sees — and the suite never touches the network.
    monkeypatch.setattr(BigQueryRunner, "client", property(lambda _self: _NoWarehouse()))

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_renders_with_no_credentials(app):
    assert any("Retail analysis" in t.value for t in app.title)
    assert len(app.chat_input) == 1
    # The signed-in user's data scope is stated on the page, not just enforced.
    assert any("Women" in caption for caption in _texts(app.caption))


def test_slash_commands_never_consult_the_model(app):
    """The CLI's guarantee, in the browser: these must survive an LLM outage."""
    agent = app.session_state["agent"]
    before = len(agent.provider.requests)

    for command in ("/whoami", "/reports", "/trace", "/help", "/persona"):
        app.chat_input[0].set_value(command).run()
        assert not app.exception, f"{command}: {app.exception}"

    assert len(agent.provider.requests) == before


def test_unknown_command_is_reported(app):
    app.chat_input[0].set_value("/bogus").run()
    assert any("Unknown command" in warning for warning in _texts(app.warning))


def test_a_question_reaches_the_model_and_shows_its_telemetry(app):
    agent = app.session_state["agent"]
    app.chat_input[0].set_value("what were my top brands this year?").run()

    assert not app.exception
    assert agent.provider.requests
    # Every answer carries the turn's status and trace id, so a reviewer can go
    # from an answer to its replay without leaving the page.
    assert any(
        "status" in caption and "trace" in caption for caption in _texts(app.caption)
    )


def test_a_pending_deletion_is_shown_and_deletes_nothing_on_its_own(app):
    agent = app.session_state["agent"]
    report = agent.store.save(
        owner="maya",
        conversation_id=agent.conversation_id,
        title="Acme review",
        body="Acme quarterly numbers",
    )
    agent.broker.propose_deletion(
        actor="maya", criteria="mentioning Acme", text="Acme"
    )
    app.run()

    assert any("Awaiting confirmation" in w for w in _texts(app.warning))
    assert agent.store.get(report.report_id).is_deleted is False


def test_confirmation_is_intercepted_before_the_model(app):
    agent = app.session_state["agent"]
    report = agent.store.save(
        owner="maya", conversation_id=agent.conversation_id, title="Acme review",
        body="x",
    )
    agent.broker.propose_deletion(
        actor="maya", criteria="mentioning Acme", text="Acme"
    )
    app.run()
    calls_before = len(agent.provider.requests)

    app.chat_input[0].set_value("yes").run()

    assert agent.store.get(report.report_id).is_deleted is True
    assert any("Deleted 1 report" in s for s in _texts(app.success))
    # "yes" was never routed to the model.
    assert len(agent.provider.requests) == calls_before


def test_undo_restores_after_a_confirmed_delete(app):
    agent = app.session_state["agent"]
    report = agent.store.save(
        owner="maya", conversation_id=agent.conversation_id, title="Acme review",
        body="x",
    )
    agent.broker.propose_deletion(
        actor="maya", criteria="mentioning Acme", text="Acme"
    )
    app.run()
    app.chat_input[0].set_value("yes").run()
    app.chat_input[0].set_value("/undo").run()

    assert agent.store.get(report.report_id).is_deleted is False
    assert any("Restored 1 report" in s for s in _texts(app.success))


def test_switching_user_starts_a_clean_conversation(app):
    app.chat_input[0].set_value("/whoami").run()
    assert app.session_state["transcript"]

    app.selectbox[0].set_value("daniel").run()

    assert not app.exception
    assert app.session_state["transcript"] == []
    # Daniel is scoped to categories, not departments.
    assert any("Jeans" in caption for caption in _texts(app.caption))
