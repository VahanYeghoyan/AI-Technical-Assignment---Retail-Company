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
from retail_agent.gcp import resolve_project
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
        self.max_results_requested = "never called"

    def result(self, timeout=None, max_results=None):  # noqa: ARG002
        self.max_results_requested = max_results
        return FakeRows(self._dataframe, max_results)


class FakeRows:
    """A RowIterator: fetches at most max_results, but knows the full count."""

    def __init__(self, dataframe: pd.DataFrame, max_results: int | None):
        self._dataframe = dataframe
        self._max_results = max_results
        self.total_rows = len(dataframe)

    def to_dataframe(self, create_bqstorage_client=None):  # noqa: ARG002
        if self._max_results is None:
            return self._dataframe
        return self._dataframe.head(self._max_results)


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
        # Job configs are recorded because not doing so hid a bug that broke
        # every real query: maximum_bytes_billed=None serialises to the string
        # "None" and BigQuery rejects the job. A fake that ignores the config
        # cannot catch that class of defect.
        self.dry_run_configs: list[object] = []
        self.executed_configs: list[object] = []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        if job_config is not None and getattr(job_config, "dry_run", False):
            self.dry_run_configs.append(job_config)
            return FakeJob(pd.DataFrame(), total_bytes_processed=self.dry_run_bytes)
        self.executed.append(sql)
        self.executed_configs.append(job_config)
        if self.behaviours:
            behaviour = self.behaviours.pop(0)
            if isinstance(behaviour, Exception):
                raise behaviour
        self.last_job = FakeJob(self.dataframe)
        return self.last_job


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
    assert client.executed_configs[0].maximum_bytes_billed == 123_456


def test_dry_run_config_omits_maximum_bytes_billed():
    # Regression: assigning None does not mean "unset". The client serialises it
    # as the string "None" and BigQuery rejects the job with
    # "Invalid value at 'maximum_bytes_billed' (TYPE_INT64)" — which broke every
    # real query while the suite stayed green, because the fake ignored configs.
    client = FakeBQClient()
    runner(client).execute(SIMPLE_SQL, UNRESTRICTED)

    assert client.dry_run_configs[0].maximum_bytes_billed is None
    assert "maximumBytesBilled" not in client.dry_run_configs[0].to_api_repr().get(
        "query", {}
    )


def test_bad_job_configuration_is_not_blamed_on_the_model():
    # A 400 about the job config is our bug. Classifying it as a syntax error
    # would spend the entire repair budget rewriting SQL that was never wrong.
    client = FakeBQClient(
        behaviours=[
            gexc.BadRequest(
                "Invalid value at 'job.configuration.query.maximum_bytes_billed."
                "value' (TYPE_INT64), \"None\""
            )
        ]
    )
    with pytest.raises(QueryError) as err:
        runner(client).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.CONFIG
    assert err.value.self_correctable is False
    assert err.value.retryable is False
    assert len(client.executed) == 1  # not retried


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


def test_a_dry_run_blip_is_retried_rather_than_failing_the_query():
    # The dry run is the FIRST call of every query, and it had neither retries
    # nor breaker accounting — so one 503 there failed a query that a single
    # retry would have completed.
    class BlipOnFirstCall(FakeBQClient):
        def __init__(self):
            super().__init__()
            self.first = True

        def query(self, sql, job_config=None):
            if self.first:
                self.first = False
                raise gexc.ServiceUnavailable("503 blip")
            return super().query(sql, job_config)

    result = runner(BlipOnFirstCall()).execute(SIMPLE_SQL, UNRESTRICTED)
    assert result.row_count == 1


def test_the_breaker_opens_when_the_outage_hits_the_dry_run():
    # A real outage fails at the dry run, before execution is ever reached. With
    # failures counted only during execution, the breaker could never open no
    # matter how long BigQuery stayed down.
    class AlwaysDown(FakeBQClient):
        def query(self, sql, job_config=None):
            self.queries.append(sql)
            raise gexc.ServiceUnavailable("503 backend error")

    client = AlwaysDown()
    bq = runner(client, max_attempts=2)
    for _ in range(3):
        with pytest.raises(QueryError) as err:
            bq.execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.CIRCUIT_OPEN
    assert len(client.queries) == 4  # threshold reached, then fails fast


def test_a_timed_out_job_is_cancelled_before_the_retry():
    # We stop waiting; BigQuery does not stop working. Retrying without
    # cancelling left the first job running and billing, so one slow answer was
    # paid for up to three times.
    cancels = []

    class NeverFinishes:
        def __init__(self):
            self.job_id = "job-1"
            self.total_bytes_processed = 10

        def result(self, timeout=None, max_results=None):  # noqa: ARG002
            raise TimeoutError("job still running")

        def cancel(self):
            cancels.append(self.job_id)

    class SlowClient(FakeBQClient):
        def query(self, sql, job_config=None):
            if job_config is not None and getattr(job_config, "dry_run", False):
                return super().query(sql, job_config)
            self.executed.append(sql)
            return NeverFinishes()

    client = SlowClient()
    with pytest.raises(QueryError) as err:
        runner(client, max_attempts=3).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.TIMEOUT
    assert len(cancels) == len(client.executed) == 3


def test_lost_credentials_are_a_permission_error_not_a_mystery():
    # google.auth raises these, and they are not google.api_core types — so they
    # fell into UNKNOWN and the agent treated an expired login as odd SQL.
    class RefreshError(Exception):
        pass

    RefreshError.__name__ = "RefreshError"

    class NoCredentials(FakeBQClient):
        def query(self, sql, job_config=None):
            raise RefreshError("could not refresh the access token")

    with pytest.raises(QueryError) as err:
        runner(NoCredentials()).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.PERMISSION
    assert err.value.retriable_by_model is False


def test_a_dropped_network_is_transient_not_unknown():
    class Offline(FakeBQClient):
        def query(self, sql, job_config=None):
            raise ConnectionError("Max retries exceeded: network unreachable")

    with pytest.raises(QueryError) as err:
        runner(Offline(), max_attempts=2).execute(SIMPLE_SQL, UNRESTRICTED)

    assert err.value.kind is QueryErrorKind.TRANSIENT


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


@pytest.mark.parametrize(
    ("sql", "is_refusal"),
    [
        ("SELECT email FROM users", True),                       # PII
        ("SELECT TO_JSON_STRING(u) AS j FROM users u", True),    # whole row
        ("DELETE FROM orders WHERE order_id = 1", True),         # a write
        ("SELECT CURRENT_DATE() AS today", False),               # no table: a fumble
        ("SELECT FROM WHERE", False),                            # syntax
    ],
)
def test_only_policy_rejections_count_as_refusals(sql, is_refusal):
    tracer = Tracer(user_id="ceo", conversation_id="c1")
    with pytest.raises(QueryError):
        runner(FakeBQClient()).execute(sql, UNRESTRICTED, tracer=tracer)

    assert tracer.metrics.sql_rejections == 1
    assert tracer.metrics.refusals == (1 if is_refusal else 0)


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
    # ...and only the capped rows were downloaded, not all 500.
    assert client.last_job.max_results_requested == 50


def test_the_compatibility_api_still_returns_every_row():
    # execute_query() is the supplied runner's interface: callers expect the
    # whole result, so the agent's row cap must not leak into it.
    client = FakeBQClient(dataframe=pd.DataFrame({"n": range(500)}))
    frame = runner(client, max_rows=50).execute_query(SIMPLE_SQL)

    assert len(frame) == 500
    assert client.last_job.max_results_requested is None


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
    def __init__(self, text=None, function_call=None, thought_signature=None):
        self.text = text
        self.function_call = function_call
        # Gemini 3.x attaches this to the PART carrying a function call, and
        # requires it back verbatim in history.
        self.thought_signature = thought_signature


class _Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args


def _fake_genai_response(
    text="", call=None, prompt_tokens=10, output_tokens=5, signature=None
):
    part = (
        _Part(text=text)
        if text
        else _Part(function_call=call, thought_signature=signature)
    )
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


def test_thought_signature_is_captured_from_the_response():
    # Gemini 3.x requires this opaque token back verbatim when the call appears
    # in history. Dropping it made every tool-using turn die on its SECOND model
    # call with "400 Function call is missing a thought_signature".
    call = _Call("run_analysis_sql", {"sql": "SELECT 1"})
    provider = gemini([_fake_genai_response(call=call, signature=b"sig-abc123")])

    response = provider.generate(system="s", contents=[])

    assert response.function_calls[0].thought_signature == b"sig-abc123"


def test_malformed_request_is_fatal_with_no_retry_and_no_fallback():
    # A 400 means our request is wrong. Retrying it, or re-sending it to the
    # fallback model, only pays twice to fail the same way.
    from google.genai import errors as genai_errors

    bad = genai_errors.ClientError(
        400,
        {"error": {"message": "INVALID_ARGUMENT: Function call is missing a "
                              "thought_signature in functionCall parts"}},
    )
    provider = gemini(
        [bad, _fake_genai_response("never reached")],
        model="gemini-3.6-flash",
        fallback_model="gemini-3.1-flash-lite",
    )

    with pytest.raises(LLMConfigError):
        provider.generate(system="s", contents=[])

    assert provider.client.models.calls == ["gemini-3.6-flash"]


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


def test_a_per_minute_rate_limit_is_retried_not_declared_fatal():
    # AI Studio's free tier answers a PER-MINUTE limit with a 429 that mentions
    # both "billing" and "Quota exceeded" and ends "Please retry in 18.5s".
    # Matching those words alone told the user the account was empty and that
    # retrying would not help, for the rate limit this assignment says to expect.
    from google.genai import errors as genai_errors

    rate_limited = genai_errors.ClientError(
        429,
        {"error": {"message":
            "You exceeded your current quota, please check your plan and billing "
            "details. * Quota exceeded for metric: generate_content_free_tier_requests"
            ", limit: 10\nPlease retry in 18.5s."}},
    )
    provider = gemini([rate_limited, _fake_genai_response("recovered")])

    assert provider.generate(system="s", contents=[]).text == "recovered"


def test_a_rate_limited_model_falls_back_to_the_secondary():
    # Documented behaviour that was never implemented: only a 404 fell back, so
    # a rate-limited turn failed outright with a spare model sitting idle.
    from google.genai import errors as genai_errors

    busy = genai_errors.ClientError(
        429, {"error": {"message": "Resource exhausted, please retry in 5s"}}
    )
    provider = gemini(
        [busy, busy, busy, _fake_genai_response("from fallback")],
        model="gemini-3.6-flash",
        fallback_model="gemini-3.1-flash-lite",
        max_attempts=3,
    )

    response = provider.generate(system="s", contents=[])

    assert response.text == "from fallback"
    assert response.used_fallback is True
    assert provider.client.models.calls[-1] == "gemini-3.1-flash-lite"


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


class _Credentials:
    def __init__(self, quota_project_id=None):
        self.quota_project_id = quota_project_id


def _adc(monkeypatch, *, project=None, quota_project=None):
    """Application Default Credentials that resolve as given, with no env override."""
    import google.auth

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(
        google.auth, "default", lambda: (_Credentials(quota_project), project)
    )


def test_vertex_uses_the_gcloud_project_when_none_is_configured(monkeypatch):
    # A reviewer who ran `gcloud config set project` must not also have to
    # repeat it in .env — and must never inherit someone else's project from it.
    import google.genai

    _adc(monkeypatch, project="reviewer-project")
    seen = {}
    monkeypatch.setattr(google.genai, "Client", lambda **kwargs: seen.update(kwargs))

    provider = GeminiProvider(use_vertex=True)

    assert provider.project == "reviewer-project"
    assert seen["project"] == "reviewer-project" and seen["vertexai"] is True


def test_the_project_falls_back_to_the_credentials_quota_project(monkeypatch):
    # google.auth asks the gcloud binary for the configured project, so with
    # gcloud off PATH it finds none — though the credentials file names one.
    _adc(monkeypatch, project=None, quota_project="quota-project")

    assert resolve_project() == "quota-project"


def test_an_explicit_project_wins(monkeypatch):
    _adc(monkeypatch, project="gcloud-project")
    assert resolve_project("explicit") == "explicit"

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-env")
    assert resolve_project() == "from-env"


def test_bigquery_resolves_the_same_project_vertex_does(monkeypatch):
    from google.cloud import bigquery

    _adc(monkeypatch, project=None, quota_project="quota-project")
    seen = {}
    monkeypatch.setattr(bigquery, "Client", lambda project=None: seen.update(project=project))

    BigQueryRunner().client  # noqa: B018 - building the client is the point

    assert seen["project"] == "quota-project"


def test_vertex_with_no_project_anywhere_says_how_to_set_one(monkeypatch):
    _adc(monkeypatch, project=None, quota_project=None)
    with pytest.raises(LLMConfigError) as err:
        GeminiProvider(use_vertex=True)
    assert "gcloud config set project" in str(err.value)


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
