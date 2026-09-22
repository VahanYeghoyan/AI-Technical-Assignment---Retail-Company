"""BigQuery access: cost control, retries and error classification.

This extends the BigQueryRunner supplied with the assignment rather than
replacing it — execute_query() and get_table_schema() keep their signatures —
but a lean client is not safe to put behind an LLM. Four things are added:

  COST CONTROL      Every query is dry-run first. If it would scan more than
                    BQ_MAX_BYTES_BILLED it is rejected before a byte is billed,
                    and maximum_bytes_billed is set on the real job as a second
                    ceiling in case the estimate is wrong. A model that writes a
                    cross join should cost nothing.

  ERROR CLASSIFICATION  The agent needs to know *why* a query failed, because
                    the correct response differs: a syntax error should be handed
                    back to the model to fix (self-correction), a 503 should be
                    retried silently, and a cost violation should ask the user to
                    narrow the question. One generic "query failed" makes all
                    three indistinguishable and produces retry loops that burn
                    quota without ever succeeding (Requirement 5).

  CIRCUIT BREAKER   After repeated transient failures the runner fails fast for a
                    cooldown instead of retrying every call. This is what stops a
                    BigQuery outage from turning into a cost incident.

  BOUNDED LIBRARY RETRIES  google-cloud-bigquery retries on its own, for up to
                    10 minutes per call by default — underneath everything above.
                    Measured against a dead endpoint, a query hung for 19 minutes
                    before this module saw any error at all, and that error was
                    unclassifiable, so the breaker never counted it. Every call
                    now passes the library a short retry deadline (it still
                    absorbs a blip) and no job-level retry (this module owns
                    restarts); when the library gives up, that is an outage.

  RESULT HYGIENE    Results are PII-scrubbed and row-capped before they go
                    anywhere near the model's context.

The client is created lazily so importing this module — and running the test
suite — requires no credentials.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import pandas as pd
from google.api_core import exceptions as gexc

from retail_agent.gcp import resolve_project
from retail_agent.observability import Tracer
from retail_agent.safety.pii import pseudonymize_customer_ids, scrub_dataframe
from retail_agent.safety.scope import Scope
from retail_agent.safety.sql_guard import SqlGuardError, validate_and_rewrite

DEFAULT_DATASET = "bigquery-public-data.thelook_ecommerce"

# 2 GB. The whole thelook dataset is a few hundred MB, so anything approaching
# this ceiling is a runaway query, not a legitimate analysis.
DEFAULT_MAX_BYTES = 2_000_000_000
DEFAULT_TIMEOUT_S = 60
# Rows handed to the LLM. Executives ask for aggregates; a 10k-row answer is a
# bug, and pasting it into the prompt is how context windows and bills explode.
DEFAULT_MAX_ROWS = 200
# How long the client library may keep retrying one API call before giving up
# (its own default is 600 s), and how long any single HTTP request may take
# (its own default is forever).
DEFAULT_API_RETRY_S = 10.0
API_REQUEST_TIMEOUT_S = 30.0


class QueryErrorKind(StrEnum):
    """Why a query failed — drives what the agent does next."""

    GUARD = "guard"              # blocked by policy; tell the model why
    SYNTAX = "syntax"            # model's fault; hand back for self-correction
    CONFIG = "config"            # OUR bug (bad job config); never self-correct
    COST = "cost_exceeded"       # ask the user to narrow the question
    PERMISSION = "permission"    # config problem; do not retry, surface to operator
    QUOTA = "quota_exceeded"     # the PROJECT is out of BigQuery quota; not a login problem
    NOT_FOUND = "not_found"      # table/dataset missing
    TIMEOUT = "timeout"
    TRANSIENT = "transient"      # retry with backoff
    UNAVAILABLE = "unavailable"  # the client library already retried and gave up
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class QueryError(Exception):
    """A classified query failure."""

    def __init__(self, kind: QueryErrorKind, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        # Text shown to the model when it is allowed to try again.
        self.hint = hint

    @property
    def retryable(self) -> bool:
        return self.kind in {QueryErrorKind.TRANSIENT, QueryErrorKind.TIMEOUT}

    @property
    def indicates_outage(self) -> bool:
        """Evidence the warehouse is unwell — what the circuit breaker counts.

        Wider than retryable: UNAVAILABLE is not retried here, because the
        client library has already spent its own retries on it, but it is the
        clearest outage signal there is.
        """
        return self.kind in {
            QueryErrorKind.TRANSIENT,
            QueryErrorKind.TIMEOUT,
            QueryErrorKind.UNAVAILABLE,
        }

    @property
    def self_correctable(self) -> bool:
        return self.kind in {QueryErrorKind.SYNTAX, QueryErrorKind.GUARD}

    @property
    def retriable_by_model(self) -> bool:
        """Could a DIFFERENT query plausibly succeed?

        Wider than self_correctable: a cost rejection is not a mistake in the
        SQL, but narrowing the question does fix it, and a missing table means
        the model named something that is not there. Everything outside this set
        is infrastructure — credentials, outages, our own job config — where
        handing the model another turn only spends the budget to fail again.
        """
        return self.kind in {
            QueryErrorKind.SYNTAX,
            QueryErrorKind.GUARD,
            QueryErrorKind.COST,
            QueryErrorKind.NOT_FOUND,
        }


@dataclass
class QueryResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    bytes_processed: int
    bytes_billed: int = 0
    truncated: bool = False
    redactions: tuple[str, ...] = ()
    scope_applied: bool = False

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0

    def to_markdown(self, max_rows: int | None = None) -> str:
        """Compact rendering for the model's context and the CLI.

        Renders every row the runner kept. It used to default to 20 while the
        runner's own cap was 200, so a 50-row "revenue by state" answer was
        written from the first 20 states — and `truncated` said False, because
        that flag only tracks the 200 cap. Two different limits, one of them
        invisible: the model had no way to know it was reasoning about a
        fraction of the result.

        Falls back to a hand-rolled table if `tabulate` is absent. pandas treats
        it as an optional extra, so a missing install surfaces as an ImportError
        mid-turn — which reads as a mysterious internal error rather than a
        missing package. Rendering a result set is too central to the agent to
        let it depend on an optional import being present.
        """
        if self.is_empty:
            return "(no rows)"
        head = self.dataframe if max_rows is None else self.dataframe.head(max_rows)
        try:
            body = head.to_markdown(index=False)
        except ImportError:
            body = _plain_table(head)
        hidden = self.row_count - len(head)
        if hidden > 0:
            body += (
                f"\n\n… {hidden} more row(s) not shown, of {self.row_count} total. "
                "Aggregate further or add a LIMIT if you need them all."
            )
        return body


def _plain_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table using only the standard library."""
    columns = [str(c) for c in frame.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in record] for record in frame.values]
    widths = [
        max(len(col), *(len(row[i]) for row in rows)) if rows else len(col)
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _reason(err: Exception) -> str:
    """BigQuery's machine-readable error reason, e.g. "quotaExceeded"."""
    try:
        return str(err.errors[0]["reason"])  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _classify(err: Exception) -> QueryError:
    """Map a BigQuery exception onto a kind the agent can act on."""
    message = str(err)
    lowered = message.lower()

    if isinstance(err, gexc.RetryError):
        # The client library retried a transient failure until its deadline.
        # Retrying again here would only multiply the wait.
        return QueryError(
            QueryErrorKind.UNAVAILABLE,
            f"BigQuery did not respond: {message}",
        )

    if isinstance(err, gexc.BadRequest):
        # Not every 400 is the model's fault. A malformed job configuration is
        # OUR bug, and handing it to the model as "fix your SQL" burns the whole
        # repair budget on something no SQL change can resolve. Check this first.
        if "job.configuration" in message or "Invalid value at" in message:
            return QueryError(
                QueryErrorKind.CONFIG,
                message,
                hint="The query was not the problem; the job configuration was "
                "rejected. This needs an operator, not a rewrite.",
            )
        # BigQuery reports genuine SQL mistakes as 400s. Distinguishing a syntax
        # error from other 400s is what makes self-correction converge.
        if any(
            token in lowered
            for token in ("syntax error", "unrecognized name", "not found inside",
                          "no matching signature", "invalid cast", "ambiguous")
        ):
            return QueryError(
                QueryErrorKind.SYNTAX,
                message,
                hint="Fix the SQL and try again. " + message.split("\n")[0],
            )
        if "exceeded" in lowered and "bytes" in lowered:
            return QueryError(QueryErrorKind.COST, message)
        return QueryError(QueryErrorKind.SYNTAX, message, hint=message.split("\n")[0])

    if isinstance(err, gexc.Forbidden):
        # BigQuery answers 403 for three different things, and only one of them
        # is about credentials. A sandbox project that has spent its free
        # 1 TB/month was told its login was refused.
        # The structured reason decides when there is one; the text only when
        # there is not.
        reason = _reason(err)
        if reason == "rateLimitExceeded" or (
            not reason and "exceeded rate limits" in lowered
        ):
            return QueryError(QueryErrorKind.TRANSIENT, message)
        if reason == "quotaExceeded" or (not reason and "quota exceeded" in lowered):
            return QueryError(QueryErrorKind.QUOTA, message)
        return QueryError(QueryErrorKind.PERMISSION, message)

    if isinstance(err, gexc.Unauthorized):
        return QueryError(QueryErrorKind.PERMISSION, message)

    # Credentials that cannot be obtained or refreshed, and raw transport
    # failures, never reach google.api_core's exception types — so they landed
    # in UNKNOWN and the agent treated "the laptop lost its network" the same as
    # "the model wrote odd SQL".
    if type(err).__name__ in {
        "DefaultCredentialsError", "RefreshError", "UserAccessTokenError",
    }:
        return QueryError(
            QueryErrorKind.PERMISSION,
            f"{message}. Run `gcloud auth application-default login`.",
        )
    if isinstance(err, gexc.NotFound):
        return QueryError(QueryErrorKind.NOT_FOUND, message)
    # Ahead of the socket-error test below, because TimeoutError is a subclass
    # of OSError: checking sockets first classified every query timeout as a
    # dropped connection, and stopped the timed-out job from being cancelled.
    if isinstance(err, (gexc.DeadlineExceeded, TimeoutError)):
        return QueryError(QueryErrorKind.TIMEOUT, message)
    if isinstance(err, (ConnectionError, OSError)) or type(err).__name__ in {
        "ConnectionError", "ConnectTimeout", "ReadTimeout", "TransportError",
        "ServerNotFoundError",
    }:
        return QueryError(QueryErrorKind.TRANSIENT, message)
    if isinstance(
        err,
        (
            gexc.TooManyRequests,
            gexc.ServiceUnavailable,
            gexc.InternalServerError,
            gexc.BadGateway,
            gexc.GatewayTimeout,
        ),
    ):
        return QueryError(QueryErrorKind.TRANSIENT, message)

    return QueryError(QueryErrorKind.UNKNOWN, message)


def _cancel_quietly(job: Any, tracer: Tracer | None) -> None:
    """Best-effort cancel of a job we have stopped waiting for."""
    cancel = getattr(job, "cancel", None)
    if not callable(cancel):
        return
    try:
        cancel()
    except Exception as err:  # noqa: BLE001 - cancelling must never mask the timeout
        if tracer:
            tracer.emit("bq.cancel_failed", error=str(err))
    else:
        if tracer:
            tracer.emit("bq.cancelled", job_id=str(getattr(job, "job_id", "")))


@dataclass
class _CircuitBreaker:
    """Fails fast while a dependency is known to be down."""

    threshold: int = 4
    cooldown_s: float = 30.0
    _failures: int = 0
    _opened_at: float | None = None

    def check(self, now: float) -> None:
        if self._opened_at is None:
            return
        if now - self._opened_at < self.cooldown_s:
            raise QueryError(
                QueryErrorKind.CIRCUIT_OPEN,
                "BigQuery is failing repeatedly; pausing queries briefly",
                hint="Tell the user the data warehouse is temporarily unavailable "
                "and that you will not retry right now.",
            )
        self.reset()

    def record_failure(self, now: float) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = now

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None


class BigQueryRunner:
    """Executes scoped, cost-capped, read-only queries."""

    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str = DEFAULT_DATASET,
        *,
        client: Any | None = None,
        max_bytes_billed: int | None = None,
        timeout_s: float | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_attempts: int = 3,
        api_retry_s: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id
        self.max_bytes_billed = int(
            max_bytes_billed or os.getenv("BQ_MAX_BYTES_BILLED", DEFAULT_MAX_BYTES)
        )
        self.timeout_s = float(
            timeout_s or os.getenv("BQ_QUERY_TIMEOUT_S", DEFAULT_TIMEOUT_S)
        )
        self.max_rows = max_rows
        self.max_attempts = max_attempts
        self.api_retry_s = float(
            api_retry_s or os.getenv("BQ_API_RETRY_S") or DEFAULT_API_RETRY_S
        )
        self._sleep = sleep
        self._client = client
        self._breaker = _CircuitBreaker()

    @property
    def client(self) -> Any:
        """Create the BigQuery client on first use.

        Lazy so that importing this module, and running the tests, needs no
        credentials — and so a missing ADC surfaces as a clear message at query
        time rather than an import crash.
        """
        if self._client is None:
            from google.cloud import bigquery

            try:
                # The same resolution Vertex uses, so jobs and model calls can
                # never land in two different projects.
                self._client = bigquery.Client(project=resolve_project(self.project_id))
            except Exception as err:  # noqa: BLE001
                raise QueryError(
                    QueryErrorKind.PERMISSION,
                    f"could not authenticate to BigQuery: {err}. Run "
                    "`gcloud auth application-default login`.",
                ) from err
        return self._client

    # -- public API -------------------------------------------------------

    def execute(
        self,
        sql: str,
        scope: Scope,
        *,
        tracer: Tracer | None = None,
    ) -> QueryResult:
        """Validate, scope, cost-check and run a query."""
        try:
            guarded = validate_and_rewrite(sql, scope)
        except SqlGuardError as err:
            if tracer:
                tracer.metrics.sql_rejections += 1
                if err.is_policy:
                    tracer.metrics.refusals += 1
                tracer.audit(
                    "sql_rejected", reason=err.reason, message=err.message, sql=sql
                )
            raise QueryError(QueryErrorKind.GUARD, err.message, hint=err.message) from err

        if tracer:
            tracer.metrics.sql_attempts += 1
            # The SQL that actually reaches BigQuery, which is not the SQL the
            # model wrote. Without it a trace cannot answer the one question an
            # entitlement incident turns on: what did this user's query run as?
            tracer.emit("sql.rewritten", sql=guarded.sql, scope_applied=guarded.rewritten)
            if guarded.rewritten:
                tracer.audit(
                    "scope_applied", tables=sorted(guarded.tables), user=scope.user_id
                )

        estimated = self._dry_run(guarded.sql, tracer=tracer)
        if estimated > self.max_bytes_billed:
            raise QueryError(
                QueryErrorKind.COST,
                f"this query would scan {estimated / 1e9:.2f} GB, over the "
                f"{self.max_bytes_billed / 1e9:.2f} GB limit",
                hint="Narrow the question: add a date filter, or aggregate rather "
                "than selecting detail rows.",
            )

        dataframe, bytes_processed, bytes_billed, total_rows = self._run_with_retries(
            guarded.sql, tracer=tracer, max_rows=self.max_rows
        )

        cleaned, redactions = scrub_dataframe(dataframe)
        cleaned, pseudonymised = pseudonymize_customer_ids(cleaned)
        if redactions and tracer:
            tracer.metrics.pii_redactions += len(redactions)
            tracer.audit("pii_redacted", where="query_result", details=redactions)
        if pseudonymised and tracer:
            tracer.audit("pseudonymised", where="query_result", columns=pseudonymised)

        cleaned = cleaned.head(self.max_rows)
        row_count = max(total_rows, len(dataframe))
        truncated = row_count > len(cleaned)

        result = QueryResult(
            sql=guarded.sql,
            dataframe=cleaned,
            row_count=row_count,
            bytes_processed=bytes_processed,
            bytes_billed=bytes_billed,
            truncated=truncated,
            redactions=tuple(redactions),
            scope_applied=guarded.rewritten,
        )
        if tracer:
            tracer.metrics.bq_bytes_billed += bytes_billed
            tracer.metrics.bq_bytes_processed += bytes_processed
            if result.is_empty:
                tracer.metrics.empty_results += 1
            tracer.emit(
                "bq.result",
                rows=result.row_count,
                bytes_processed=bytes_processed,
                bytes_billed=bytes_billed,
                truncated=truncated,
                empty=result.is_empty,
            )
        return result

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Unscoped execution, kept for compatibility with the supplied runner.

        Not used by the agent: every agent path goes through execute(), which
        applies entitlements. Retained so existing scripts and notebooks that
        import this class keep working.
        """
        dataframe, _, _, _ = self._run_with_retries(
            sql_query, tracer=None, max_rows=None
        )
        return dataframe

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Column metadata for one table (as in the supplied runner)."""
        table = self.client.get_table(
            f"{self.dataset_id}.{table_name}",
            retry=self._api_retry(),
            timeout=API_REQUEST_TIMEOUT_S,
        )
        return [
            {
                "name": field_.name,
                "type": field_.field_type,
                "mode": field_.mode,
                "description": field_.description or "",
            }
            for field_ in table.schema
        ]

    # -- internals --------------------------------------------------------

    def _api_retry(self) -> Any:
        """The client library's own retry policy, cut to api_retry_s.

        Still worth having — it absorbs a dropped connection mid-poll without
        resubmitting the job, and job ids are generated client-side so a
        retried insert cannot run twice — but bounded, so an outage reaches
        this module's classification and breaker in seconds, not ten minutes.
        """
        from google.cloud.bigquery.retry import DEFAULT_RETRY

        return DEFAULT_RETRY.with_timeout(self.api_retry_s)

    def _submit(self, sql: str, *, dry_run: bool) -> Any:
        """client.query() with every library-level wait bounded.

        job_retry=None: the library would otherwise re-run a failed job for up
        to 40 minutes. Restarting a job is this module's decision — it is what
        the breaker and the cancel-before-retry rule exist to govern.
        """
        return self.client.query(
            sql,
            job_config=self._job_config(dry_run=dry_run),
            retry=self._api_retry(),
            timeout=API_REQUEST_TIMEOUT_S,
            job_retry=None,
        )

    def _record(self, error: QueryError, *, counts_as_healthy: bool) -> None:
        """Feed one failure to the circuit breaker."""
        if error.indicates_outage:
            self._breaker.record_failure(time.monotonic())
        elif counts_as_healthy:
            # BigQuery answered — with a no. The warehouse itself is up.
            self._breaker.reset()

    def _job_config(self, *, dry_run: bool):
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(
            dry_run=dry_run,
            use_query_cache=True,
            labels={"app": "retail-agent"},
        )
        if not dry_run:
            # Second ceiling: even if the dry-run estimate is wrong, BigQuery
            # itself refuses to bill beyond this.
            #
            # Set only when it applies. Assigning None does NOT mean "unset" —
            # the client serialises it into the request as the string "None",
            # and BigQuery rejects the job with
            # "Invalid value at 'maximum_bytes_billed' (TYPE_INT64)".
            config.maximum_bytes_billed = self.max_bytes_billed
        return config

    def _dry_run(self, sql: str, *, tracer: Tracer | None) -> int:
        """Estimate the cost, with the same retries and breaker as a real run.

        This is the FIRST call of every query, so a warehouse outage always
        fails here — and while this path had neither retries nor breaker
        accounting, the breaker could never open no matter how long BigQuery
        was down, and a single 503 blip failed a query that one retry would
        have completed.
        """
        def attempt() -> int:
            job = self._submit(sql, dry_run=True)
            return int(getattr(job, "total_bytes_processed", 0) or 0)

        # A dry run is planning only: it keeps answering while the execution
        # engine is degraded, so its failures are evidence the warehouse is
        # unwell, but its successes are not evidence that it is well. Letting it
        # clear the breaker would erase the record of the real queries failing.
        estimated = self._with_retries(
            attempt, tracer=tracer, what="dry_run", counts_as_healthy=False
        )
        if tracer:
            tracer.emit("bq.dry_run", estimated_bytes=estimated)
        return estimated

    def _with_retries(
        self,
        attempt: Callable[[], Any],
        *,
        tracer: Tracer | None,
        what: str,
        counts_as_healthy: bool = True,
    ) -> Any:
        """Run one BigQuery interaction, retrying only what is worth retrying."""
        last: QueryError | None = None
        for number in range(1, self.max_attempts + 1):
            self._breaker.check(time.monotonic())
            try:
                result = attempt()
            except Exception as err:  # noqa: BLE001
                # A QueryError from here is already classified (the client
                # property raises PERMISSION when credentials are missing);
                # re-classifying it would demote that to UNKNOWN.
                error = err if isinstance(err, QueryError) else _classify(err)
                self._record(error, counts_as_healthy=counts_as_healthy)
                if not error.retryable:
                    raise error from err
                last = error
                if tracer:
                    tracer.metrics.retries += 1
                    tracer.emit(
                        "bq.retry", attempt=number, phase=what, kind=str(error.kind),
                        error=error.message,
                    )
                if number < self.max_attempts:
                    # Exponential backoff with jitter, so concurrent sessions do
                    # not retry in lockstep against a struggling service.
                    self._sleep(min(2 ** (number - 1), 8) * (0.5 + random.random()))
                continue
            else:
                if counts_as_healthy:
                    self._breaker.reset()
                return result

        assert last is not None
        raise last

    def _run_with_retries(
        self, sql: str, *, tracer: Tracer | None, max_rows: int | None
    ) -> tuple[pd.DataFrame, int, int, int]:
        """Run a query; returns (rows, bytes processed, bytes billed, total rows).

        Only `max_rows` rows are downloaded. The cap used to be applied after
        the download, so a detail query pulled every row over REST — 180k
        order_items rows for a 200-row answer — outside the query timeout,
        which covers the job and not the download.
        """
        last: QueryError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._breaker.check(time.monotonic())
            job = None
            try:
                job = self._submit(sql, dry_run=False)
                rows = job.result(
                    timeout=self.timeout_s,
                    max_results=max_rows,
                    retry=self._api_retry(),
                    job_retry=None,
                )
                # REST download, not the Storage API. google-cloud-bigquery-
                # storage is deliberately not a dependency (it pulls in gRPC),
                # and result sets here are row-capped well below the size where
                # it would pay off. Saying so explicitly also stops the client
                # from warning "BigQuery Storage module not found" on every
                # query -- it checks this flag before attempting the import.
                dataframe = rows.to_dataframe(create_bqstorage_client=False)
                total_rows = int(getattr(rows, "total_rows", None) or len(dataframe))
            except Exception as err:  # noqa: BLE001
                error = err if isinstance(err, QueryError) else _classify(err)
                if job is not None and error.indicates_outage:
                    # We stopped waiting; BigQuery did not stop working. Without
                    # this the retry submits a second job while the first one
                    # runs on and bills, so a slow query was charged up to three
                    # times over for one answer — and giving up entirely left
                    # the job running with nobody waiting for it.
                    _cancel_quietly(job, tracer)
                self._record(error, counts_as_healthy=True)
                if not error.retryable:
                    raise error from err
                last = error
                if tracer:
                    tracer.metrics.retries += 1
                    tracer.emit(
                        "bq.retry", attempt=attempt, phase="execute",
                        kind=str(error.kind), error=error.message,
                    )
                if attempt < self.max_attempts:
                    # Exponential backoff with jitter, so concurrent sessions do
                    # not retry in lockstep against a struggling service.
                    self._sleep(min(2 ** (attempt - 1), 8) * (0.5 + random.random()))
                continue
            else:
                self._breaker.reset()
                bytes_processed = int(getattr(job, "total_bytes_processed", 0) or 0)
                # What is actually charged: a 10 MB minimum per table
                # referenced, and 0 for a cache hit. Fakes without the
                # attribute fall back to the processed figure.
                bytes_billed = int(
                    getattr(job, "total_bytes_billed", bytes_processed) or 0
                )
                return dataframe, bytes_processed, bytes_billed, total_rows

        assert last is not None
        raise last
