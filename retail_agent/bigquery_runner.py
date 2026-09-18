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

from retail_agent.observability import Tracer
from retail_agent.safety.pii import scrub_dataframe
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


class QueryErrorKind(StrEnum):
    """Why a query failed — drives what the agent does next."""

    GUARD = "guard"              # blocked by policy; tell the model why
    SYNTAX = "syntax"            # model's fault; hand back for self-correction
    COST = "cost_exceeded"       # ask the user to narrow the question
    PERMISSION = "permission"    # config problem; do not retry, surface to operator
    NOT_FOUND = "not_found"      # table/dataset missing
    TIMEOUT = "timeout"
    TRANSIENT = "transient"      # retry with backoff
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
    def self_correctable(self) -> bool:
        return self.kind in {QueryErrorKind.SYNTAX, QueryErrorKind.GUARD}


@dataclass
class QueryResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    bytes_processed: int
    truncated: bool = False
    redactions: tuple[str, ...] = ()
    scope_applied: bool = False

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0

    def to_markdown(self, max_rows: int = 20) -> str:
        """Compact rendering for the model's context and the CLI.

        Falls back to a hand-rolled table if `tabulate` is absent. pandas treats
        it as an optional extra, so a missing install surfaces as an ImportError
        mid-turn — which reads as a mysterious internal error rather than a
        missing package. Rendering a result set is too central to the agent to
        let it depend on an optional import being present.
        """
        if self.is_empty:
            return "(no rows)"
        head = self.dataframe.head(max_rows)
        try:
            body = head.to_markdown(index=False)
        except ImportError:
            body = _plain_table(head)
        if self.row_count > max_rows:
            body += f"\n\n… {self.row_count - max_rows} more row(s) not shown"
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


def _classify(err: Exception) -> QueryError:
    """Map a BigQuery exception onto a kind the agent can act on."""
    message = str(err)
    lowered = message.lower()

    if isinstance(err, gexc.BadRequest):
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

    if isinstance(err, (gexc.Forbidden, gexc.Unauthorized)):
        return QueryError(QueryErrorKind.PERMISSION, message)
    if isinstance(err, gexc.NotFound):
        return QueryError(QueryErrorKind.NOT_FOUND, message)
    if isinstance(err, (gexc.DeadlineExceeded, TimeoutError)):
        return QueryError(QueryErrorKind.TIMEOUT, message)
    if isinstance(
        err,
        (
            gexc.TooManyRequests,
            gexc.ServiceUnavailable,
            gexc.InternalServerError,
            gexc.GatewayTimeout,
        ),
    ):
        return QueryError(QueryErrorKind.TRANSIENT, message)

    return QueryError(QueryErrorKind.UNKNOWN, message)


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
                self._client = bigquery.Client(project=self.project_id)
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
                tracer.audit(
                    "sql_rejected", reason=err.reason, message=err.message, sql=sql
                )
            raise QueryError(QueryErrorKind.GUARD, err.message, hint=err.message) from err

        if tracer:
            tracer.metrics.sql_attempts += 1
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

        dataframe, bytes_processed = self._run_with_retries(guarded.sql, tracer=tracer)

        cleaned, redactions = scrub_dataframe(dataframe)
        if redactions and tracer:
            tracer.metrics.pii_redactions += len(redactions)
            tracer.audit("pii_redacted", where="query_result", details=redactions)

        truncated = len(cleaned) > self.max_rows
        if truncated:
            cleaned = cleaned.head(self.max_rows)

        result = QueryResult(
            sql=guarded.sql,
            dataframe=cleaned,
            row_count=int(len(dataframe)),
            bytes_processed=bytes_processed,
            truncated=truncated,
            redactions=tuple(redactions),
            scope_applied=guarded.rewritten,
        )
        if tracer:
            tracer.metrics.bq_bytes_billed += bytes_processed
            if result.is_empty:
                tracer.metrics.empty_results += 1
            tracer.emit(
                "bq.result",
                rows=result.row_count,
                bytes=bytes_processed,
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
        dataframe, _ = self._run_with_retries(sql_query, tracer=None)
        return dataframe

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Column metadata for one table (unchanged from the supplied runner)."""
        table = self.client.get_table(f"{self.dataset_id}.{table_name}")
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

    def _job_config(self, *, dry_run: bool):
        from google.cloud import bigquery

        return bigquery.QueryJobConfig(
            dry_run=dry_run,
            use_query_cache=True,
            # Second ceiling: even if the dry-run estimate is wrong, BigQuery
            # itself refuses to bill beyond this.
            maximum_bytes_billed=None if dry_run else self.max_bytes_billed,
            labels={"app": "retail-agent"},
        )

    def _dry_run(self, sql: str, *, tracer: Tracer | None) -> int:
        self._breaker.check(time.monotonic())
        try:
            job = self.client.query(sql, job_config=self._job_config(dry_run=True))
        except Exception as err:  # noqa: BLE001
            raise _classify(err) from err
        estimated = int(getattr(job, "total_bytes_processed", 0) or 0)
        if tracer:
            tracer.emit("bq.dry_run", estimated_bytes=estimated)
        return estimated

    def _run_with_retries(
        self, sql: str, *, tracer: Tracer | None
    ) -> tuple[pd.DataFrame, int]:
        last: QueryError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._breaker.check(time.monotonic())
            try:
                job = self.client.query(sql, job_config=self._job_config(dry_run=False))
                dataframe = job.result(timeout=self.timeout_s).to_dataframe()
            except Exception as err:  # noqa: BLE001
                error = err if isinstance(err, QueryError) else _classify(err)
                if not error.retryable:
                    self._breaker.reset()
                    raise error from err
                last = error
                self._breaker.record_failure(time.monotonic())
                if tracer:
                    tracer.metrics.retries += 1
                    tracer.emit(
                        "bq.retry", attempt=attempt, kind=str(error.kind),
                        error=error.message,
                    )
                if attempt < self.max_attempts:
                    # Exponential backoff with jitter, so concurrent sessions do
                    # not retry in lockstep against a struggling service.
                    self._sleep(min(2 ** (attempt - 1), 8) * (0.5 + random.random()))
                continue
            else:
                self._breaker.reset()
                bytes_processed = int(getattr(job, "total_bytes_processed", 0) or 0)
                return dataframe, bytes_processed

        assert last is not None
        raise last
