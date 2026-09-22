"""Tracing, metrics and audit logging.

Requirement 7 asks two different questions, and they need two different shapes
of data:

  "When is the agent failing?"  -> counters and timings, aggregated per turn.
  "Why did THIS turn fail?"     -> the full message correspondence, replayable.

Both come from one append-only JSONL event stream (var/traces/YYYY-MM-DD.jsonl).
One line per event, each carrying trace_id / span_id / parent_span_id. That is
deliberately the OpenTelemetry span shape without the OpenTelemetry dependency:
in production the same emit() calls become OTel spans exported to Cloud Trace,
and the JSONL sink becomes a Cloud Logging sink into BigQuery. Nothing else in
the code changes.

Two properties worth noting:

  * Traces are PII-scrubbed on the way in. A debug log that quietly becomes the
    one place customer emails are retained is a data breach with extra steps.
  * Writes are best-effort. Observability must never be able to take down the
    chat: if the sink fails, the turn continues and the failure is counted.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from retail_agent.safety.pii import scrub_text

_DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[1] / "var" / "traces"

# How much of a prompt/response body to keep inline. Full bodies are available
# with TRACE_FULL_MESSAGES=1 for local debugging; the default keeps trace files
# small enough to grep and cheap enough to ship.
_SNIPPET_CHARS = 2000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _truncate(value: str) -> str:
    if os.getenv("TRACE_FULL_MESSAGES") == "1" or len(value) <= _SNIPPET_CHARS:
        return value
    return f"{value[:_SNIPPET_CHARS]}… [+{len(value) - _SNIPPET_CHARS} chars]"


def _sanitise(value: Any) -> Any:
    """Scrub PII and truncate long strings, recursively."""
    if isinstance(value, str):
        return _truncate(scrub_text(value)[0])
    if isinstance(value, dict):
        return {k: _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    return value


@dataclass
class TurnMetrics:
    """Counters for one user turn, emitted on turn_end.

    These are the agent-level metrics from Requirement 7: enough to alert on
    without opening a single trace.
    """

    llm_calls: int = 0
    llm_errors: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    sql_attempts: int = 0
    sql_rejections: int = 0
    sql_self_corrections: int = 0
    bq_bytes_billed: int = 0
    empty_results: int = 0
    pii_redactions: int = 0
    # Requests the code refused on policy grounds: a PII, whole-row, write or
    # out-of-dataset query, or a deletion that named nothing to match. Refusals
    # the model makes in prose are not counted — they need an eval to detect.
    refusals: int = 0
    retries: int = 0
    fallback_model_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Tracer:
    """Append-only event sink for one conversation.

    A Tracer is created per CLI session and carries the identifiers that make
    turns correlatable: session, user, conversation.
    """

    user_id: str
    conversation_id: str
    trace_dir: Path = field(default_factory=lambda: Path(
        os.getenv("TRACE_DIR") or _DEFAULT_TRACE_DIR
    ))
    trace_id: str = field(default_factory=_new_id)
    metrics: TurnMetrics = field(default_factory=TurnMetrics)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _span_stack: list[str] = field(default_factory=list, repr=False)
    _sink_failures: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.trace_dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"

    # -- emission ---------------------------------------------------------

    def emit(self, event: str, **attrs: Any) -> None:
        """Write one event. Never raises — observability cannot break the chat."""
        record = {
            "ts": _now(),
            "trace_id": self.trace_id,
            "span_id": self._span_stack[-1] if self._span_stack else None,
            "parent_span_id": self._span_stack[-2] if len(self._span_stack) > 1 else None,
            "event": event,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            **_sanitise(attrs),
        }
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except Exception:  # noqa: BLE001 - deliberately swallowed
            self._sink_failures += 1

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[str]:
        """Time a unit of work and emit start/end events around it."""
        span_id = _new_id()
        self._span_stack.append(span_id)
        started = time.perf_counter()
        self.emit(f"{name}.start", **attrs)
        try:
            yield span_id
        except Exception as err:
            self.emit(
                f"{name}.error",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                error_type=type(err).__name__,
                error=str(err),
            )
            raise
        else:
            self.emit(
                f"{name}.end",
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        finally:
            self._span_stack.pop()

    # -- turn lifecycle ---------------------------------------------------

    def start_turn(self, question: str) -> str:
        """Begin a turn. Returns the turn's trace_id, shown to the user on error."""
        self.trace_id = _new_id()
        self.metrics = TurnMetrics()
        self.emit("turn.start", question=question)
        return self.trace_id

    def end_turn(self, *, status: str, answer: str = "") -> None:
        self.emit(
            "turn.end",
            status=status,
            answer=answer,
            metrics=self.metrics.as_dict(),
            sink_failures=self._sink_failures,
        )

    # -- audit ------------------------------------------------------------

    def audit(self, action: str, **attrs: Any) -> None:
        """Record a security-relevant decision.

        Audit events are ordinary trace events with audit=True so they can be
        routed to a longer-retention sink in production: refusals, PII
        redactions, entitlement rewrites, and every report deletion.
        """
        self.emit(f"audit.{action}", audit=True, **attrs)


# -- reading back ---------------------------------------------------------


def read_events(
    trace_dir: Path | str | None = None,
    *,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load events from the JSONL sink, newest file last.

    This backs the CLI's /trace command: the deep-dive path from Requirement 7.
    Pass `user_id` to see only one user's turns — what a chat frontend should
    always do, since a trace carries the question, the rows and the answer.
    """
    directory = Path(trace_dir or os.getenv("TRACE_DIR") or _DEFAULT_TRACE_DIR)
    if not directory.exists():
        return []
    events: list[dict[str, Any]] = []
    for file in sorted(directory.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written line must not break debugging
            if trace_id and record.get("trace_id") != trace_id:
                continue
            if conversation_id and record.get("conversation_id") != conversation_id:
                continue
            if user_id and record.get("user_id") != user_id:
                continue
            events.append(record)
    return events[-limit:] if limit else events


def summarise_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse an event stream into one row per turn, for `/trace` with no id."""
    turns: dict[str, dict[str, Any]] = {}
    for record in events:
        trace_id = record.get("trace_id")
        if not trace_id:
            continue
        turn = turns.setdefault(
            trace_id,
            {"trace_id": trace_id, "started": record["ts"], "status": "incomplete"},
        )
        if record["event"] == "turn.start":
            turn["question"] = record.get("question", "")
        elif record["event"] == "turn.end":
            turn["status"] = record.get("status", "unknown")
            turn["metrics"] = record.get("metrics", {})
            turn["ended"] = record["ts"]
        elif record["event"].endswith(".error"):
            turn.setdefault("errors", []).append(record.get("error_type", "error"))
    return list(turns.values())
