"""Tests for Requirement 5: resilience, self-correction and cost control.

Both external dependencies are injected as fakes, so these run with no
credentials and no quota. Where the real service raises a specific exception
type, the fakes raise that same real type (google.api_core.exceptions,
google.genai.errors) rather than a lookalike — classification logic that is only
ever tested against stand-ins tends to be wrong against the real thing.
"""

from __future__ import annotations

import pandas as pd
import pytest
from google.api_core import exceptions as gexc

from retail_agent.bigquery_runner import (
    BigQueryRunner,
    QueryError,
    QueryErrorKind,
)
from retail_agent.llm import (
    FunctionCall,
    GeminiProvider,
    LLMBudgetError,
    LLMConfigError,
    LLMQuotaError,
    LLMResponse,
    LLMTransientError,
    StubProvider,
)
from retail_agent.observability import Tracer
from retail_agent.safety.scope import Scope

UNRESTRICTED = Scope(user_id="ceo", unrestricted=True)
WOMENS = Scope(user_id="maya", departments=frozenset({"Women"}))

SIMPLE_SQL = "SELECT COUNT(*) AS n FROM orders"


# ---------------------------------------------------------------------------
# BigQuery fakes
# ---------------------------------------------------------------------------


class FakeJob:
    def __init__(self, dataframe: pd.DataFrame, total_bytes_processed: int = 1000):
        self._dataframe = dataframe
        self.total_bytes_processed = total_bytes_processed

    def result(self, timeout=None):  # noqa: ARG002
        return self

    def to_dataframe(self):
        return self._dataframe


class FakeBQClient:
    """Programmable BigQuery client.

    `behaviours` is consumed one entry per query() call: an Exception is raised,
    anything else is returned as a job.
    """

    def __init__(self, behaviours=None, dry_run_bytes: int = 1000, dataframe=None):
        self.behaviours = list(behaviours or [])
        self.dry_run_bytes = dry_run_bytes
        self.dataframe = (
            dataframe if dataframe is not None else pd.DataFrame({"n": [42]})
        )
        self.queries: list[str] = []
        self.executed: list[str] = []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        if job_config is not None and getattr(job_config, "dry_run", False):
            return FakeJob(pd.DataFrame(), total_bytes_processed=self.dry_run_bytes)
        self.executed.append(sql)
        if self.behaviours:
            behaviour = self.behaviours.pop(0)
            if isinstance(behaviour, Exception):
                raise behaviour
        return FakeJob(self.dataframe)


def runner(client, **kwargs):
    kwargs.setdefault("sleep", lambda _s: None)  # no real backoff waits in tests
    return BigQueryRunner(project_id="test-project", client=client, **kwargs)


# ---------------------------------------------------------------------------
# Cost control
# ---------------------------------------------------------------------------


def test_query_over_the_byte_cap_is_rejected_before_it_runs():
    client = FakeBQClient(dry_run_bytes=50_000_000_000)  # 50 GB
    with pytest.raises(QueryError) as err:
        runner(client, max_bytes_billed=2_000_000_000).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.COST
    # The crucial part: nothing was actually executed, so nothing was billed.
    assert client.executed == []
    assert "narrow" in err.value.hint.lower()


def test_real_job_carries_a_maximum_bytes_billed_ceiling():
    client = FakeBQClient()
    runner(client, max_bytes_billed=123_456).execute(SIMPLE_SQL, UNRESTRICTED)
    assert client.executed, "the query should have run"


def test_dry_run_happens_before_every_execution():
    client = FakeBQClient()
    runner(client).execute(SIMPLE_SQL, UNRESTRICTED)
    # Two calls: one dry run, one real.
    assert len(client.queries) == 2
    assert len(client.executed) == 1


# ---------------------------------------------------------------------------
# Error classification and retries
# ---------------------------------------------------------------------------


def test_syntax_error_is_self_correctable_and_not_retried():
    client = FakeBQClient(
        behaviours=[gexc.BadRequest("Syntax error: Unexpected keyword FROM at [1:8]")]
    )
    with pytest.raises(QueryError) as err:
        runner(client).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.SYNTAX
    assert err.value.self_correctable is True
    assert err.value.retryable is False
    assert len(client.executed) == 1  # retrying a syntax error only wastes quota
    assert "Syntax error" in err.value.hint


def test_transient_error_is_retried_and_then_succeeds():
    client = FakeBQClient(behaviours=[gexc.ServiceUnavailable("backend error")])
    tracer = Tracer(user_id="maya", conversation_id="c1")

    result = runner(client).execute(SIMPLE_SQL, UNRESTRICTED, tracer=tracer)

    assert result.row_count == 1
    assert len(client.executed) == 2  # failed once, then succeeded
    assert tracer.metrics.retries == 1


def test_retries_are_bounded():
    client = FakeBQClient(behaviours=[gexc.ServiceUnavailable("down")] * 5)
    with pytest.raises(QueryError) as err:
        runner(client, max_attempts=3).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.TRANSIENT
    assert len(client.executed) == 3  # not unbounded


def test_permission_error_is_not_retried():
    client = FakeBQClient(behaviours=[gexc.Forbidden("no access")])
    with pytest.raises(QueryError) as err:
        runner(client).execute(SIMPLE_SQL, UNRESTRICTED)
    assert err.value.kind is QueryErrorKind.PERMISSION
    assert len(client.executed) == 1


def test_circuit_breaker_opens_after_repeated_failures():
    client = FakeBQClient(behaviours=[gexc.ServiceUnavailable("down")] * 20)
    bq = runner(client, max_attempts=2)

    # Two turns' worth of failures trips the breaker (threshold 4).
    for _ in range(2):
        with pytest.raises(QueryError):
            bq.execute(SIMPLE_SQL, UNRESTRICTED)

    executed_before = len(client.executed)
    with pytest.raises(QueryError) as err:
        bq.execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.CIRCUIT_OPEN
    # Fails fast: the open circuit means no further calls reach BigQuery.
    assert len(client.executed) == executed_before


# ---------------------------------------------------------------------------
# Guard integration and result hygiene
# ---------------------------------------------------------------------------


def test_guard_rejection_surfaces_as_a_self_correctable_error():
    client = FakeBQClient()
    with pytest.raises(QueryError) as err:
        runner(client).execute("SELECT email FROM users", UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.GUARD
    assert err.value.self_correctable is True
    assert client.queries == []  # never reached BigQuery


def test_scope_is_applied_and_audited():
    client = FakeBQClient()
    tracer = Tracer(user_id="maya", conversation_id="c1")

    result = runner(client).execute(SIMPLE_SQL, WOMENS, tracer=tracer)

    assert result.scope_applied is True
    assert "department IN ('Women')" in client.executed[0]


def test_results_are_pii_scrubbed_before_reaching_the_model():
    client = FakeBQClient(
        dataframe=pd.DataFrame({"note": ["reach me at ceo@example.com"], "n": [1]})
    )
    tracer = Tracer(user_id="maya", conversation_id="c1")

    result = runner(client).execute(SIMPLE_SQL, UNRESTRICTED, tracer=tracer)

    assert "ceo@example.com" not in result.dataframe.to_string()
    assert tracer.metrics.pii_redactions > 0


def test_large_result_sets_are_truncated():
    client = FakeBQClient(dataframe=pd.DataFrame({"n": range(500)}))
    result = runner(client, max_rows=50).execute(SIMPLE_SQL, UNRESTRICTED)

    assert result.truncated is True
    assert len(result.dataframe) == 50
    assert result.row_count == 500  # the true count is still reported honestly


def test_results_render_even_without_the_optional_tabulate_package(monkeypatch):
    # pandas treats tabulate as an optional extra, so a missing install shows up
    # as an ImportError mid-turn — which the agent's catch-all would report as a
    # mysterious internal error. Rendering must not depend on it.
    client = FakeBQClient(dataframe=pd.DataFrame({"brand": ["Levi's"], "revenue": [12.5]}))
    result = runner(client).execute(SIMPLE_SQL, UNRESTRICTED)

    def no_tabulate(*args, **kwargs):
        raise ImportError("`Import tabulate` failed.")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", no_tabulate)

    rendered = result.to_markdown()
    assert "brand" in rendered and "Levi's" in rendered and "12.5" in rendered


def test_empty_result_is_flagged_rather_than_treated_as_failure():
    client = FakeBQClient(dataframe=pd.DataFrame({"n": []}))
    tracer = Tracer(user_id="maya", conversation_id="c1")

    result = runner(client).execute(SIMPLE_SQL, UNRESTRICTED, tracer=tracer)

    assert result.is_empty is True
    assert tracer.metrics.empty_results == 1
    assert result.to_markdown() == "(no rows)"


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class FakeGenAIModels:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls: list[str] = []

    def generate_content(self, *, model, contents, config):  # noqa: ARG002
        self.calls.append(model)
        behaviour = (
            self.behaviours.pop(0) if self.behaviours else _fake_genai_response("ok")
        )
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


class FakeGenAIClient:
    def __init__(self, behaviours=()):
        self.models = FakeGenAIModels(behaviours)


class _Part:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args


def _fake_genai_response(text="", call=None, prompt_tokens=10, output_tokens=5):
    part = _Part(text=text) if text else _Part(function_call=call)
    candidate = type("C", (), {"content": type("Ct", (), {"parts": [part]})()})()
    usage = type(
        "U",
        (),
        {"prompt_token_count": prompt_tokens, "candidates_token_count": output_tokens},
    )()
    return type("R", (), {"candidates": [candidate], "usage_metadata": usage})()


def gemini(behaviours, **kwargs):
    kwargs.setdefault("sleep", lambda _s: None)
    return GeminiProvider(client=FakeGenAIClient(behaviours), **kwargs)


def test_gemini_parses_text_and_usage():
    provider = gemini([_fake_genai_response("Revenue was $1.2M")])
    response = provider.generate(system="s", contents=[])

    assert response.text == "Revenue was $1.2M"
    assert response.prompt_tokens == 10
    assert response.output_tokens == 5
    assert response.used_fallback is False


def test_gemini_parses_tool_calls():
    call = _Call("run_analysis_sql", {"sql": "SELECT 1"})
    provider = gemini([_fake_genai_response(call=call)])

    response = provider.generate(system="s", contents=[])

    assert response.wants_tool is True
    assert response.function_calls[0] == FunctionCall(
        name="run_analysis_sql", args={"sql": "SELECT 1"}
    )


def test_gemini_retries_transient_server_errors():
    from google.genai import errors as genai_errors

    boom = genai_errors.ServerError(503, {"error": {"message": "unavailable"}})
    provider = gemini([boom, _fake_genai_response("recovered")])

    assert provider.generate(system="s", contents=[]).text == "recovered"


def test_exhausted_quota_is_fatal_and_not_retried():
    # The real failure observed on this project's key: retrying cannot fix it,
    # so the agent must say so rather than spin.
    from google.genai import errors as genai_errors

    depleted = genai_errors.ClientError(
        429, {"error": {"message": "Your prepayment credits are depleted."}}
    )
    provider = gemini([depleted, _fake_genai_response("never reached")])

    with pytest.raises(LLMQuotaError) as err:
        provider.generate(system="s", contents=[])

    assert "will not help" in str(err.value)
    assert provider.client.models.calls == ["gemini-3.6-flash"]  # tried once only


def test_retired_model_falls_back_to_the_secondary():
    from google.genai import errors as genai_errors

    retired = genai_errors.ClientError(
        404, {"error": {"message": "model is no longer available to new users"}}
    )
    provider = gemini(
        [retired, _fake_genai_response("from fallback")],
        model="gemini-2.5-flash",
        fallback_model="gemini-3.6-flash",
    )

    response = provider.generate(system="s", contents=[])

    assert response.text == "from fallback"
    assert response.used_fallback is True
    assert provider.client.models.calls == ["gemini-2.5-flash", "gemini-3.6-flash"]


def test_call_budget_caps_a_runaway_loop():
    provider = gemini([_fake_genai_response("ok")] * 20, call_budget=3)
    provider.begin_turn()
    for _ in range(3):
        provider.generate(system="s", contents=[])

    with pytest.raises(LLMBudgetError):
        provider.generate(system="s", contents=[])


def test_missing_api_key_is_a_clear_config_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMConfigError) as err:
        GeminiProvider()
    assert "LLM_PROVIDER=stub" in str(err.value)


def test_stub_provider_runs_without_network():
    provider = StubProvider(script=[LLMResponse(text="offline answer")])
    provider.begin_turn()

    assert provider.generate(system="s", contents=[]).text == "offline answer"
    assert provider.requests[0]["system"] == "s"


def test_stub_provider_respects_its_budget():
    provider = StubProvider(script=[LLMResponse(text="x")] * 10, call_budget=2)
    provider.begin_turn()
    provider.generate(system="s", contents=[])
    provider.generate(system="s", contents=[])
    with pytest.raises(LLMBudgetError):
        provider.generate(system="s", contents=[])
