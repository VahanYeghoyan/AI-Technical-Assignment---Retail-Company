"""The Saved Reports library.

Requirement 3's destructive-ops flow needs somewhere to be destructive about.
This is that store: SQLite locally, one row per saved report.

Three decisions worth stating, because they are what make "delete all reports
mentioning Client X" safe to support at all:

  * Deletes are SOFT. delete() stamps deleted_at; the row survives. That turns
    the scariest possible instruction ("delete all reports mentioning X") from
    irreversible into a 30-day undo. purge() is the only hard delete, and no
    agent tool is wired to it.

  * Ownership is enforced here, not in the prompt. delete() silently refuses
    rows the actor does not own and REPORTS them as skipped, so the user is told
    "3 deleted, 2 skipped (not yours)" rather than the agent either failing
    entirely or quietly over-deleting.

  * Search is substring matching over title + body + entity tags. "Mentioning
    Client X" has no structured answer in a B2C dataset, so the match set is
    always shown to the user before anything is deleted — the confirmation step
    is what makes an imprecise search safe.

In production this becomes Firestore (per-user documents, TTL policy on
deleted_at) or Cloud SQL; the interface below is what the agent tools call, so
swapping the backend touches only this file.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "var" / "reports.db"

# How long a soft-deleted report can still be restored.
RESTORE_WINDOW_DAYS = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id       TEXT PRIMARY KEY,
    owner           TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    sql_used        TEXT,
    entities        TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL,
    deleted_at      TEXT,
    deleted_by      TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_owner ON reports (owner, deleted_at);
CREATE INDEX IF NOT EXISTS idx_reports_conversation ON reports (conversation_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Report:
    report_id: str
    owner: str
    conversation_id: str
    title: str
    body: str
    sql_used: str
    entities: tuple[str, ...]
    created_at: str
    deleted_at: str | None = None
    deleted_by: str | None = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def summary(self) -> str:
        """One-line form, used in confirmation prompts and listings."""
        marker = " (deleted)" if self.is_deleted else ""
        return f"[{self.report_id[:8]}] {self.title}{marker} — {self.created_at[:10]}"


@dataclass(frozen=True)
class DeleteOutcome:
    """What actually happened, so the agent can report it honestly."""

    deleted: tuple[Report, ...]
    skipped_not_owned: tuple[Report, ...]
    already_deleted: tuple[Report, ...]

    @property
    def deleted_count(self) -> int:
        return len(self.deleted)


class ReportStore:
    """SQLite-backed saved reports."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("REPORTS_DB_PATH") or _DEFAULT_DB)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes -----------------------------------------------------------

    def save(
        self,
        *,
        owner: str,
        conversation_id: str,
        title: str,
        body: str,
        sql_used: str = "",
        entities: Sequence[str] = (),
    ) -> Report:
        report = Report(
            report_id=uuid.uuid4().hex,
            owner=owner,
            conversation_id=conversation_id,
            title=title,
            body=body,
            sql_used=sql_used,
            entities=tuple(entities),
            created_at=_now(),
        )
        self._conn.execute(
            "INSERT INTO reports (report_id, owner, conversation_id, title, body,"
            " sql_used, entities, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report.report_id,
                report.owner,
                report.conversation_id,
                report.title,
                report.body,
                report.sql_used,
                json.dumps(list(report.entities)),
                report.created_at,
            ),
        )
        self._conn.commit()
        return report

    def delete(self, report_ids: Iterable[str], *, actor: str) -> DeleteOutcome:
        """Soft-delete reports, refusing any the actor does not own.

        Never raises on a partial match: the caller is expected to tell the user
        exactly what was and was not removed.
        """
        deleted: list[Report] = []
        skipped: list[Report] = []
        already: list[Report] = []

        for report_id in dict.fromkeys(report_ids):  # de-duplicate, keep order
            report = self.get(report_id)
            if report is None:
                continue
            if report.owner != actor:
                skipped.append(report)
                continue
            if report.is_deleted:
                already.append(report)
                continue
            stamp = _now()
            self._conn.execute(
                "UPDATE reports SET deleted_at = ?, deleted_by = ? WHERE report_id = ?",
                (stamp, actor, report_id),
            )
            deleted.append(
                Report(**{**report.__dict__, "deleted_at": stamp, "deleted_by": actor})
            )
        self._conn.commit()
        return DeleteOutcome(tuple(deleted), tuple(skipped), tuple(already))

    def restore(self, report_ids: Iterable[str], *, actor: str) -> tuple[Report, ...]:
        """Undo a soft delete, within the restore window."""
        cutoff = datetime.now(UTC) - timedelta(days=RESTORE_WINDOW_DAYS)
        restored: list[Report] = []
        for report_id in dict.fromkeys(report_ids):
            report = self.get(report_id)
            if report is None or not report.is_deleted or report.owner != actor:
                continue
            if datetime.fromisoformat(report.deleted_at) < cutoff:
                continue
            self._conn.execute(
                "UPDATE reports SET deleted_at = NULL, deleted_by = NULL"
                " WHERE report_id = ?",
                (report_id,),
            )
            restored.append(
                Report(**{**report.__dict__, "deleted_at": None, "deleted_by": None})
            )
        self._conn.commit()
        return tuple(restored)

    def purge(self, *, older_than_days: int = RESTORE_WINDOW_DAYS) -> int:
        """Hard-delete expired soft-deletes. No agent tool calls this."""
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM reports WHERE deleted_at IS NOT NULL AND deleted_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount

    # -- reads ------------------------------------------------------------

    def get(self, report_id: str) -> Report | None:
        row = self._conn.execute(
            "SELECT * FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return self._row_to_report(row) if row else None

    def resolve_id(self, prefix: str) -> Report | None:
        """Look up by the short id shown in listings ([1a2b3c4d])."""
        row = self._conn.execute(
            "SELECT * FROM reports WHERE report_id LIKE ? || '%'", (prefix,)
        ).fetchone()
        return self._row_to_report(row) if row else None

    def list_for_user(
        self, owner: str, *, include_deleted: bool = False
    ) -> tuple[Report, ...]:
        sql = "SELECT * FROM reports WHERE owner = ?"
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql, (owner,)).fetchall()
        return tuple(self._row_to_report(r) for r in rows)

    def search(
        self,
        *,
        actor: str,
        text: str | None = None,
        conversation_id: str | None = None,
        owned_only: bool = True,
        include_deleted: bool = False,
    ) -> tuple[Report, ...]:
        """Find reports for a deletion proposal or a listing.

        owned_only defaults to True: a delete proposal should surface only what
        the actor can actually act on. Pass False to show the wider match set
        (used to tell the user "2 more match but belong to someone else").
        """
        clauses: list[str] = []
        params: list[object] = []

        if owned_only:
            clauses.append("owner = ?")
            params.append(actor)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if text:
            # ESCAPE so a user searching for a literal % or _ gets what they asked for.
            needle = f"%{_escape_like(text)}%"
            clauses.append(
                "(title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\'"
                " OR entities LIKE ? ESCAPE '\\')"
            )
            params.extend([needle, needle, needle])

        sql = "SELECT * FROM reports"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(self._row_to_report(r) for r in rows)

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> Report:
        return Report(
            report_id=row["report_id"],
            owner=row["owner"],
            conversation_id=row["conversation_id"],
            title=row["title"],
            body=row["body"],
            sql_used=row["sql_used"] or "",
            entities=tuple(json.loads(row["entities"] or "[]")),
            created_at=row["created_at"],
            deleted_at=row["deleted_at"],
            deleted_by=row["deleted_by"],
        )


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so a search for '100%' is not a search for everything."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
