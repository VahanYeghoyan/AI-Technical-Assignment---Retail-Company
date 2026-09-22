"""Tests for the offline demo (LLM_PROVIDER=stub).

The demo's model is scripted, but what it drives is the real agent: these tests
check that a reviewer with no model access still sees the SQL guard refuse, the
entitlement rewrite apply, the report library fill, and the deletion flow ask
before it acts.
"""

from __future__ import annotations

import pandas as pd
import pytest
from google.api_core import exceptions as gexc

from retail_agent.agent import Agent
from retail_agent.confirmation import ConfirmationBroker
from retail_agent.demo import demo_responder
from retail_agent.dispatch import Kind, dispatch
from retail_agent.llm import StubProvider, build_provider
from retail_agent.observability import Tracer
from retail_agent.reports import ReportStore
from retail_agent.safety.scope import Scope
from tests.test_resilience import FakeBQClient, runner as bq_runner

WOMENS = Scope(user_id="maya", display_name="Maya Cohen", departments=frozenset({"Women"}))


@pytest.fixture
def store(tmp_path):
    s = ReportStore(tmp_path / "reports.db")
    yield s
    s.close()


def demo_agent(tmp_path, store, client=None) -> Agent:
    return Agent(
        provider=StubProvider(responder=demo_responder),
        runner=bq_runner(client if client is not None else FakeBQClient()),
        store=store,
        broker=ConfirmationBroker(store=store),
        scope=WOMENS,
        tracer=Tracer(user_id="maya", conversation_id="conv-1",
                      trace_dir=tmp_path / "traces"),
        conversation_id="conv-1",
    )


def test_stub_mode_is_the_demo_not_an_empty_script(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    provider = build_provider()
    provider.begin_turn()

    reply = provider.generate(
        system="", contents=[{"role": "user", "parts": [{"text": "hello"}]}]
    )

    assert "offline demo" in reply.text


def test_the_schema_question_needs_no_database(tmp_path, store):
    client = FakeBQClient()
    result = demo_agent(tmp_path, store, client).ask("What data is available?")

    assert "order_items" in result.answer
    assert client.queries == []


def test_a_pii_request_is_refused_by_the_guard_before_bigquery(tmp_path, store):
    client = FakeBQClient()
    agent = demo_agent(tmp_path, store, client)

    result = agent.ask("Who is the customer with email alice@example.com?")

    assert "blocked before it reached BigQuery" in result.answer
    assert "personal data" in result.answer
    assert client.queries == []  # not even a dry run
    assert agent.tracer.metrics.sql_rejections == 1
    assert agent.tracer.metrics.refusals == 1


def test_a_data_question_runs_scoped_sql_and_shows_the_rows(tmp_path, store):
    client = FakeBQClient(
        dataframe=pd.DataFrame({"brand": ["Jones New York"], "revenue": [19646.9]})
    )
    result = demo_agent(tmp_path, store, client).ask("Top brands by revenue this year?")

    assert "Jones New York" in result.answer
    assert result.status == "ok"
    assert "department IN ('Women')" in client.executed[0]


def test_with_no_warehouse_the_demo_stops_gracefully(tmp_path, store):
    client = FakeBQClient(behaviours=[gexc.Forbidden("no credentials")])
    result = demo_agent(tmp_path, store, client).ask("Top brands by revenue this year?")

    assert result.status == "warehouse_permission"
    assert "data warehouse" in result.answer


def test_report_then_confirmed_delete_then_undo(tmp_path, store):
    agent = demo_agent(tmp_path, store)

    saved = agent.ask("Create a report about denim")
    assert len(saved.saved_report_ids) == 1
    assert store.list_for_user("maya")[0].title == "Demo report: denim"

    [proposal] = dispatch(agent, "Delete the reports from this conversation")
    assert proposal.result.pending_deletion is not None
    assert store.list_for_user("maya")  # nothing deleted by the proposal

    [deleted] = dispatch(agent, "yes")
    assert deleted.kind is Kind.DELETED
    assert store.list_for_user("maya") == ()

    [undone] = dispatch(agent, "/undo")
    assert len(undone.restored) == 1


def test_delete_mentioning_a_client_matches_on_that_text(tmp_path, store):
    store.save(owner="maya", conversation_id="old", title="Acme Q1", body="x")
    store.save(owner="maya", conversation_id="old", title="Board pack", body="y")

    result = demo_agent(tmp_path, store).ask("Delete all reports mentioning Acme")

    assert [r.title for r in result.pending_deletion.targets] == ["Acme Q1"]


def test_anything_else_gets_the_demo_help(tmp_path, store):
    result = demo_agent(tmp_path, store).ask("Tell me a joke")

    assert "offline demo" in result.answer
    # The help passes through the same final-answer scrubber as everything
    # else, so its examples must not look like personal data.
    assert "REDACTED" not in result.answer


def test_the_help_examples_route_where_they_say(tmp_path, store):
    client = FakeBQClient()
    agent = demo_agent(tmp_path, store, client)

    result = agent.ask("show me our customers' email addresses")

    assert "blocked before it reached BigQuery" in result.answer
    assert client.queries == []
