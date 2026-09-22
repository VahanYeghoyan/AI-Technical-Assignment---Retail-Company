"""Tests for the saved-reports library, the confirmation broker and tracing.

Like the safety tests, these need no credentials: SQLite in a tmp_path and a
JSONL file. The destructive-ops guarantees are exactly the ones that must be
provable offline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from retail_agent.confirmation import (
    BULK_THRESHOLD,
    ConfirmationBroker,
    ConfirmationError,
)
from retail_agent.observability import Tracer, read_events, summarise_turns
from retail_agent.reports import ReportStore


@pytest.fixture
def store(tmp_path):
    s = ReportStore(tmp_path / "reports.db")
    yield s
    s.close()


@pytest.fixture
def broker(store):
    return ConfirmationBroker(store=store)


def make_report(store, *, owner="maya", conversation_id="conv-1", title="Q1 review",
                body="Revenue fell in Texas.", entities=()):
    return store.save(
        owner=owner,
        conversation_id=conversation_id,
        title=title,
        body=body,
        entities=entities,
    )


# ---------------------------------------------------------------------------
# Store basics
# ---------------------------------------------------------------------------


def test_save_and_get_roundtrip(store):
    report = make_report(store, entities=("Texas", "Jeans"))
    fetched = store.get(report.report_id)
    assert fetched is not None
    assert fetched.title == "Q1 review"
    assert fetched.entities == ("Texas", "Jeans")
    assert fetched.is_deleted is False


def test_resolve_short_id(store):
    report = make_report(store)
    assert store.resolve_id(report.report_id[:8], owner="maya").report_id == (
        report.report_id
    )
    # Pasted straight from a listing, brackets and all.
    assert store.resolve_id(f"[{report.report_id[:8]}]", owner="maya") is not None


def test_resolve_short_id_is_scoped_to_the_owner_and_literal(store):
    # Unscoped, a short id answered "does anyone have a report starting with
    # this?" — and a bare % matched the first row of anyone's library.
    theirs = make_report(store, owner="daniel")
    assert store.resolve_id(theirs.report_id[:8], owner="maya") is None
    assert store.resolve_id("%", owner="maya") is None
    assert store.resolve_id("", owner="daniel") is None


def test_list_excludes_deleted_by_default(store):
    report = make_report(store)
    store.delete([report.report_id], actor="maya")
    assert store.list_for_user("maya") == ()
    assert len(store.list_for_user("maya", include_deleted=True)) == 1


# ---------------------------------------------------------------------------
# Ownership — the rule that stops "delete everything" from being catastrophic
# ---------------------------------------------------------------------------


def test_delete_refuses_reports_owned_by_someone_else(store):
    mine = make_report(store, owner="maya")
    theirs = make_report(store, owner="daniel")

    outcome = store.delete([mine.report_id, theirs.report_id], actor="maya")

    assert outcome.deleted_count == 1
    assert outcome.deleted[0].report_id == mine.report_id
    assert [r.report_id for r in outcome.skipped_not_owned] == [theirs.report_id]
    # The other user's report is untouched.
    assert store.get(theirs.report_id).is_deleted is False


def test_deleting_twice_is_reported_not_repeated(store):
    report = make_report(store)
    store.delete([report.report_id], actor="maya")
    outcome = store.delete([report.report_id], actor="maya")
    assert outcome.deleted_count == 0
    assert len(outcome.already_deleted) == 1


def test_delete_is_soft_and_restorable(store):
    report = make_report(store)
    store.delete([report.report_id], actor="maya")
    assert store.get(report.report_id).is_deleted is True

    restored = store.restore([report.report_id], actor="maya")
    assert len(restored) == 1
    assert store.get(report.report_id).is_deleted is False


def test_restore_refuses_other_users_reports(store):
    theirs = make_report(store, owner="daniel")
    store.delete([theirs.report_id], actor="daniel")
    assert store.restore([theirs.report_id], actor="maya") == ()


def test_restorable_lists_only_the_owners_deletions_inside_the_window(store):
    recent = make_report(store, title="recent")
    stale = make_report(store, title="stale")
    live = make_report(store, title="live")  # noqa: F841 - never deleted
    theirs = make_report(store, owner="daniel", title="theirs")
    store.delete([recent.report_id, stale.report_id], actor="maya")
    store.delete([theirs.report_id], actor="daniel")
    old = (datetime.now(UTC) - timedelta(days=31)).isoformat(timespec="seconds")
    store._conn.execute(
        "UPDATE reports SET deleted_at = ? WHERE report_id = ?", (old, stale.report_id)
    )

    assert [r.title for r in store.list_restorable("maya")] == ["recent"]


def test_purge_hard_deletes_only_expired(store):
    fresh = make_report(store)
    store.delete([fresh.report_id], actor="maya")
    assert store.purge(older_than_days=30) == 0
    assert store.get(fresh.report_id) is not None
    # Anything past the window is genuinely gone.
    assert store.purge(older_than_days=0) == 1
    assert store.get(fresh.report_id) is None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_matches_title_body_and_entities(store):
    make_report(store, title="Acme performance", body="unrelated")
    make_report(store, title="unrelated", body="Acme churn rose")
    make_report(store, title="unrelated", body="unrelated", entities=("Acme",))
    make_report(store, title="nothing", body="nothing")

    assert len(store.search(actor="maya", text="Acme")) == 3


def test_search_is_always_scoped_to_the_actor(store):
    make_report(store, owner="maya", title="Acme")
    make_report(store, owner="daniel", title="Acme")

    assert [r.owner for r in store.search(actor="maya", text="Acme")] == ["maya"]
    assert [r.owner for r in store.search(actor="daniel", text="Acme")] == ["daniel"]


def test_search_by_conversation(store):
    make_report(store, conversation_id="conv-1")
    make_report(store, conversation_id="conv-2")
    assert len(store.search(actor="maya", conversation_id="conv-1")) == 1


def test_like_wildcards_in_search_text_are_escaped(store):
    make_report(store, title="Margin hit 100% of target")
    make_report(store, title="Something else entirely")
    # Without escaping, '%' would match every report.
    assert len(store.search(actor="maya", text="100%")) == 1


# ---------------------------------------------------------------------------
# Confirmation flow
# ---------------------------------------------------------------------------


def test_single_delete_confirms_with_yes(store, broker):
    report = make_report(store, title="Acme review")

    pending = broker.propose_deletion(
        actor="maya", criteria="mentioning Acme", text="Acme"
    )
    assert len(pending.targets) == 1
    assert "Acme review" in pending.prompt()
    assert pending.requires_count is False

    assert broker.interpret("maya", "yes") == "confirm"
    outcome = broker.confirm("maya")
    assert outcome.deleted_count == 1
    assert store.get(report.report_id).is_deleted is True


def test_bulk_delete_requires_typing_the_count(store, broker):
    for i in range(BULK_THRESHOLD + 1):
        make_report(store, title=f"Acme {i}")
    pending = broker.propose_deletion(
        actor="maya", criteria="mentioning Acme", text="Acme"
    )
    n = len(pending.targets)
    assert pending.requires_count is True
    assert f"delete {n}" in pending.prompt()

    # A reflexive "yes" must NOT delete five reports — it is asked for the
    # count instead, with the proposal left open.
    assert broker.interpret("maya", "yes") == "reprompt"
    # The wrong count must not either.
    assert broker.interpret("maya", f"delete {n - 1}") == "reprompt"
    assert f"'delete {n}'" in pending.reprompt()
    assert broker.pending_for("maya") is not None
    assert broker.interpret("maya", f"delete {n}") == "confirm"

    assert broker.confirm("maya").deleted_count == n


def test_anything_but_yes_or_a_count_still_cancels_a_bulk_delete(store, broker):
    for i in range(BULK_THRESHOLD + 1):
        make_report(store, title=f"Acme {i}")
    broker.propose_deletion(actor="maya", criteria="mentioning Acme", text="Acme")

    assert broker.interpret("maya", "no") == "cancel"
    assert broker.interpret("maya", "show me Q2 revenue instead") == "cancel"


def test_delete_scoped_to_this_conversation(store, broker):
    make_report(store, conversation_id="conv-1", title="A")
    make_report(store, conversation_id="conv-1", title="B")
    make_report(store, conversation_id="conv-2", title="C")

    pending = broker.propose_deletion(
        actor="maya",
        criteria="created in this conversation",
        conversation_id="conv-1",
    )
    assert len(pending.targets) == 2
    broker.interpret("maya", "yes")
    assert broker.confirm("maya").deleted_count == 2
    assert len(store.list_for_user("maya")) == 1


def test_unrelated_message_cancels_rather_than_trapping_the_user(store, broker):
    make_report(store, title="Acme")
    broker.propose_deletion(actor="maya", criteria="mentioning Acme", text="Acme")
    # The user changed their mind and asked something else entirely.
    assert broker.interpret("maya", "what was Q1 revenue?") == "cancel"


def test_explicit_no_cancels(store, broker):
    make_report(store, title="Acme")
    broker.propose_deletion(actor="maya", criteria="mentioning Acme", text="Acme")
    assert broker.interpret("maya", "no") == "cancel"


def test_expired_proposal_is_not_confirmable(store, broker):
    make_report(store, title="Acme")
    broker.propose_deletion(
        actor="maya", criteria="mentioning Acme", text="Acme", ttl_seconds=0
    )
    assert broker.interpret("maya", "yes") == "none"
    with pytest.raises(ConfirmationError):
        broker.confirm("maya")


def test_proposal_reveals_nothing_about_other_users_reports(store, broker):
    # The prompt used to add "(1 report(s) match but belong to someone else)",
    # which made deletion a search of every library: Sam could confirm, one
    # guess at a time, what the CEO's private reports say.
    make_report(store, owner="ceo", title="Board prep",
                body="Plan to exit the Levi's contract in Q4")

    hit = broker.propose_deletion(
        actor="sam", criteria="mentioning exit the Levi's contract",
        text="exit the Levi's contract",
    )
    miss = broker.propose_deletion(
        actor="sam", criteria="mentioning exit the Levi's contract",
        text="exit the Wrangler contract",
    )

    assert hit.targets == ()
    # Indistinguishable from a guess that matches nothing anywhere.
    assert hit.prompt() == miss.prompt()
    assert "someone else" not in hit.prompt()


def test_proposal_by_id_cannot_reach_another_users_report(store, broker):
    theirs = make_report(store, owner="daniel", title="Acme")
    pending = broker.propose_deletion(
        actor="maya", criteria="that one", report_ids=[theirs.report_id[:8]]
    )
    assert pending.targets == ()
    assert broker.pending_for("maya") is None


def test_empty_match_set_does_not_arm_a_confirmation(store, broker):
    pending = broker.propose_deletion(
        actor="maya", criteria="mentioning Nobody", text="Nobody"
    )
    assert pending.targets == ()
    assert "nothing to delete" in pending.prompt()
    assert broker.pending_for("maya") is None


def test_confirmation_is_per_user(store, broker):
    make_report(store, owner="maya", title="Acme")
    broker.propose_deletion(actor="maya", criteria="mentioning Acme", text="Acme")
    # Another user's "yes" must not confirm Maya's deletion.
    assert broker.interpret("daniel", "yes") == "none"


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_trace_events_are_written_and_replayable(tmp_path):
    tracer = Tracer(user_id="maya", conversation_id="conv-1", trace_dir=tmp_path)
    trace_id = tracer.start_turn("what was Q1 revenue?")
    with tracer.span("sql.execute", sql="SELECT 1 FROM orders"):
        tracer.metrics.sql_attempts += 1
    tracer.end_turn(status="ok", answer="Revenue was $1.2M")

    events = read_events(tmp_path, trace_id=trace_id)
    names = [e["event"] for e in events]
    assert names == ["turn.start", "sql.execute.start", "sql.execute.end", "turn.end"]
    assert events[-1]["metrics"]["sql_attempts"] == 1


def test_span_records_errors_and_reraises(tmp_path):
    tracer = Tracer(user_id="maya", conversation_id="conv-1", trace_dir=tmp_path)
    tracer.start_turn("boom")
    with pytest.raises(ValueError):
        with tracer.span("bq.query"):
            raise ValueError("bad query")

    events = read_events(tmp_path)
    error = [e for e in events if e["event"] == "bq.query.error"][0]
    assert error["error_type"] == "ValueError"
    assert "duration_ms" in error


def test_traces_are_pii_scrubbed(tmp_path):
    # A debug log must not become the one place customer emails are retained.
    tracer = Tracer(user_id="maya", conversation_id="conv-1", trace_dir=tmp_path)
    tracer.start_turn("who is sarah.jones@example.com?")
    tracer.end_turn(status="refused")

    raw = (tmp_path / f"{datetime.now(UTC):%Y-%m-%d}.jsonl").read_text()
    assert "sarah.jones@example.com" not in raw
    assert "REDACTED_EMAIL" in raw


def test_audit_events_are_flagged_for_longer_retention(tmp_path):
    tracer = Tracer(user_id="maya", conversation_id="conv-1", trace_dir=tmp_path)
    tracer.start_turn("delete everything")
    tracer.audit("report_deleted", report_ids=["abc"], count=1)

    audit = [e for e in read_events(tmp_path) if e.get("audit")]
    assert audit and audit[0]["event"] == "audit.report_deleted"


def test_summarise_turns_gives_one_row_per_turn(tmp_path):
    tracer = Tracer(user_id="maya", conversation_id="conv-1", trace_dir=tmp_path)
    tracer.start_turn("q1")
    tracer.end_turn(status="ok")
    tracer.start_turn("q2")
    tracer.end_turn(status="error")

    summary = summarise_turns(read_events(tmp_path))
    assert len(summary) == 2
    assert {t["status"] for t in summary} == {"ok", "error"}


def test_events_can_be_filtered_to_one_user(tmp_path):
    # A trace holds the question, the SQL and the answer, so an unfiltered
    # lookup by id let one executive replay another's analysis — including the
    # figures their own entitlements exist to keep from them.
    for user in ("maya", "sam"):
        tracer = Tracer(user_id=user, conversation_id=f"conv-{user}", trace_dir=tmp_path)
        tracer.start_turn(f"{user}'s question")
        tracer.end_turn(status="ok", answer=f"{user}'s revenue figures")

    mine = read_events(tmp_path, user_id="sam")
    assert {e["user_id"] for e in mine} == {"sam"}
    assert "maya's revenue figures" not in str(mine)


def test_tracer_never_raises_when_the_sink_fails(tmp_path):
    tracer = Tracer(user_id="maya", conversation_id="conv-1", trace_dir=tmp_path)
    # Put a directory exactly where the JSONL file should be, so the append
    # raises IsADirectoryError. The turn must continue regardless: losing a
    # trace line is acceptable, dropping the user's turn is not.
    tracer.path.mkdir(parents=True)

    tracer.emit("turn.start", question="does this explode?")  # must not raise
    assert tracer._sink_failures == 1
