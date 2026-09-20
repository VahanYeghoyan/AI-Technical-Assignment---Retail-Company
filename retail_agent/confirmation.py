"""Confirmation broker for destructive operations.

Requirement 3: deleting saved reports needs "a strict confirmation flow before
execution, without breaking UX". Those two pull in opposite directions, so the
design states exactly where the line is.

THE MODEL CANNOT DELETE ANYTHING. The only tool it is given is
propose_delete_reports, which resolves the match set and parks it here. Execution
happens in this module, after a human confirmation, in code the model never
touches. A jailbreak that makes the model *want* to delete everything still
cannot: there is no tool call that does it.

Strictness is proportional to blast radius, which is what keeps it usable:

    1 report      "yes" confirms.           Trivial to undo, trivial to confirm.
    2-3 reports   "yes" confirms.
    4+ reports    the user must type the count ("delete 7").
                  A blind "yes" to a runaway match set is the failure mode that
                  actually hurts, and it is the one thing a reflexive yes cannot do.

Plus three rules regardless of size:

  * The match set is always shown first, itemised, before any confirmation.
  * Proposals expire (default 5 minutes). Confirming a stale proposal re-runs
    the search rather than deleting whatever matched minutes ago.
  * Anything that is not a confirmation cancels the proposal and is handled as
    a normal turn. The user never has to fight their way out of a prompt.

Deletes are soft (see reports.py), so even a confirmed mistake is recoverable
for 30 days — which is what makes the "yes" tier defensible.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Sequence

from retail_agent.reports import DeleteOutcome, Report, ReportStore

# Above this many reports, a bare "yes" is not enough.
BULK_THRESHOLD = 4
DEFAULT_TTL_SECONDS = 300

_AFFIRMATIVE = frozenset(
    {"yes", "y", "yeah", "yep", "confirm", "confirmed", "do it", "go ahead", "delete"}
)
_NEGATIVE = frozenset({"no", "n", "cancel", "stop", "abort", "nevermind", "never mind"})


class ConfirmationError(RuntimeError):
    """Raised when a confirmation is attempted with no live proposal."""


@dataclass(frozen=True)
class PendingDeletion:
    """A delete the user has been shown but not yet approved."""

    action_id: str
    actor: str
    targets: tuple[Report, ...]
    not_owned: tuple[Report, ...]
    criteria: str
    created_at: datetime
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    # How the match set was resolved: "text", "conversation", "ids", "all", or
    # "none" when the request named nothing to match on.
    selector: str = "text"

    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.ttl_seconds)

    def is_expired(self, *, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at

    @property
    def requires_count(self) -> bool:
        return len(self.targets) >= BULK_THRESHOLD

    def prompt(self) -> str:
        """The exact text shown to the user before they confirm."""
        if self.selector == "none":
            return (
                "I need to know which reports you mean before I can delete "
                "anything — name the client or topic they mention, say \"the "
                "ones from this conversation\", or say \"all my reports\" if "
                "you really do mean every one of them."
            )
        if not self.targets:
            base = f"Nothing matches {self.criteria!r}, so there is nothing to delete."
            if self.not_owned:
                base += (
                    f" ({len(self.not_owned)} report(s) match but belong to someone"
                    " else, so they are not yours to delete.)"
                )
            return base

        lines = [
            f"This will delete {len(self.targets)} saved report(s) matching "
            f"{self.criteria!r}:",
            "",
            *(f"  • {r.summary()}" for r in self.targets),
        ]
        if self.not_owned:
            lines += [
                "",
                f"Skipping {len(self.not_owned)} report(s) that match but belong to"
                " someone else.",
            ]
        lines += [
            "",
            f"Deletes are reversible for 30 days (/undo).",
            (
                f"Type 'delete {len(self.targets)}' to confirm."
                if self.requires_count
                else "Type 'yes' to confirm."
            ),
            "Anything else cancels.",
        ]
        return "\n".join(lines)


@dataclass
class ConfirmationBroker:
    """Holds at most one pending destructive action per user."""

    store: ReportStore
    _pending: dict[str, PendingDeletion] = field(default_factory=dict)

    # -- propose ----------------------------------------------------------

    def propose_deletion(
        self,
        *,
        actor: str,
        criteria: str,
        text: str | None = None,
        conversation_id: str | None = None,
        report_ids: Sequence[str] | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        all_reports: bool = False,
    ) -> PendingDeletion:
        """Resolve a deletion request to a concrete, itemised match set.

        `criteria` is the human description echoed back to the user ("mentioning
        Client X", "created in this conversation"), so the confirmation prompt
        always says what it is about to act on.

        A request with no selector at all is refused rather than treated as
        "everything". An empty `text` used to fall through to an unfiltered
        search, so a model that described the criteria but forgot to pass the
        text it was matching on armed a delete-all — labelled, in the user's own
        words, as "mentioning Client X". Deleting everything has to be asked for,
        with `all_reports`, not arrived at by omission.
        """
        selector = (
            "ids" if report_ids
            else "text" if text
            else "conversation" if conversation_id
            else "all" if all_reports
            else "none"
        )
        if selector == "none":
            self._pending.pop(actor, None)
            return PendingDeletion(
                action_id=uuid.uuid4().hex[:12],
                actor=actor,
                targets=(),
                not_owned=(),
                criteria=criteria,
                created_at=datetime.now(UTC),
                ttl_seconds=ttl_seconds,
                selector=selector,
            )
        if selector == "all":
            # Say what it is, not what the model called it.
            criteria = "ALL of your saved reports"

        if report_ids:
            resolved = [self.store.resolve_id(rid) for rid in report_ids]
            targets = tuple(r for r in resolved if r and not r.is_deleted)
            not_owned = tuple(r for r in targets if r.owner != actor)
            targets = tuple(r for r in targets if r.owner == actor)
        else:
            targets = self.store.search(
                actor=actor, text=text, conversation_id=conversation_id
            )
            # Everything matching the same criteria that the actor may not touch,
            # so the user is told rather than silently under-served.
            everyone = self.store.search(
                actor=actor,
                text=text,
                conversation_id=conversation_id,
                owned_only=False,
            )
            not_owned = tuple(r for r in everyone if r.owner != actor)

        pending = PendingDeletion(
            action_id=uuid.uuid4().hex[:12],
            actor=actor,
            targets=targets,
            not_owned=not_owned,
            criteria=criteria,
            created_at=datetime.now(UTC),
            ttl_seconds=ttl_seconds,
            selector=selector,
        )
        if targets:
            self._pending[actor] = pending
        else:
            self._pending.pop(actor, None)
        return pending

    # -- confirm ----------------------------------------------------------

    def pending_for(self, actor: str) -> PendingDeletion | None:
        pending = self._pending.get(actor)
        if pending and pending.is_expired():
            del self._pending[actor]
            return None
        return pending

    def interpret(self, actor: str, message: str) -> str:
        """Classify a reply to a live proposal.

        Returns "confirm", "cancel", or "none" (no live proposal — handle the
        message as an ordinary turn).
        """
        pending = self.pending_for(actor)
        if pending is None:
            return "none"

        normalised = message.strip().lower().rstrip("!. ")
        if normalised in _NEGATIVE:
            return "cancel"

        if pending.requires_count:
            # Accept "delete 7" / "7" / "confirm 7" — the number is the point.
            match = re.fullmatch(r"(?:delete\s+|confirm\s+)?(\d+)", normalised)
            if match and int(match.group(1)) == len(pending.targets):
                return "confirm"
            return "cancel"

        return "confirm" if normalised in _AFFIRMATIVE else "cancel"

    def confirm(self, actor: str) -> DeleteOutcome:
        """Execute the pending deletion. Only reachable via interpret() == confirm."""
        pending = self.pending_for(actor)
        if pending is None:
            raise ConfirmationError(
                "there is no deletion waiting for confirmation (it may have expired)"
            )
        del self._pending[actor]
        return self.store.delete(
            [r.report_id for r in pending.targets], actor=pending.actor
        )

    def cancel(self, actor: str) -> PendingDeletion | None:
        return self._pending.pop(actor, None)
