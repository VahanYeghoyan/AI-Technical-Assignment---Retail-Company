"""Per-user data entitlements.

Loads config/entitlements.yaml into a Scope: a predicate over the products
table describing which products a given executive may analyse. The SQL guard
turns that predicate into SQL; nothing else in the system decides access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "entitlements.yaml"

# Dimensions of the products table a scope may constrain. Order matters only for
# stable error messages.
SCOPE_DIMENSIONS: tuple[str, ...] = ("departments", "categories", "brands")


class UnknownUserError(KeyError):
    """Raised when a CLI login names a user with no entitlement entry.

    Deliberately fatal: an unknown user must never fall back to an open scope.
    """


@dataclass(frozen=True)
class Scope:
    """Which products a user may analyse.

    Dimensions combine with AND; values within a dimension combine with OR.
    An empty dimension means "no constraint on that dimension".
    """

    user_id: str
    display_name: str = ""
    title: str = ""
    unrestricted: bool = False
    departments: frozenset[str] = field(default_factory=frozenset)
    categories: frozenset[str] = field(default_factory=frozenset)
    brands: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.unrestricted and not any(
            getattr(self, dim) for dim in SCOPE_DIMENSIONS
        ):
            # A scope that constrains nothing but is not explicitly marked
            # unrestricted is almost certainly a typo in the YAML. Fail loudly
            # rather than silently granting the whole catalogue.
            raise ValueError(
                f"user {self.user_id!r} has no scope dimensions and is not "
                f"marked 'unrestricted: true' — refusing to default to open access"
            )

    def describe(self) -> str:
        """Human-readable scope, shown at CLI login and in refusal messages."""
        if self.unrestricted:
            return "the full product catalogue"
        # Singular forms are spelled out: trimming the trailing "s" turns
        # "categories" into "categorie", and this string is user-facing.
        singular = {"departments": "department", "categories": "category",
                    "brands": "brand"}
        parts = [
            f"{singular[dim]} in {sorted(getattr(self, dim))}"
            for dim in SCOPE_DIMENSIONS
            if getattr(self, dim)
        ]
        return " and ".join(parts)


def _as_frozenset(raw: object) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, (list, tuple, set)):
        return frozenset(str(v) for v in raw)
    raise ValueError(f"expected a list of strings, got {type(raw).__name__}")


def _as_flag(raw: object, *, user_id: str) -> bool:
    """A YAML boolean, strictly.

    bool() of any non-empty string is True, so `unrestricted: "false"` — quoted,
    as people do — granted the full catalogue: the one typo this file must not
    turn into access. Anything but a real boolean is refused.
    """
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    raise ValueError(
        f"user {user_id!r}: 'unrestricted' must be true or false (unquoted), "
        f"got {raw!r} — refusing to guess at an access grant"
    )


def load_scopes(config_path: Path | str | None = None) -> dict[str, Scope]:
    """Parse entitlements.yaml into Scope objects, keyed by user id."""
    path = Path(config_path or os.getenv("ENTITLEMENTS_PATH") or _DEFAULT_CONFIG)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scopes: dict[str, Scope] = {}
    for user_id, entry in (doc.get("users") or {}).items():
        raw_scope = (entry or {}).get("scope") or {}
        scopes[user_id] = Scope(
            user_id=user_id,
            display_name=(entry or {}).get("display_name", ""),
            title=(entry or {}).get("title", ""),
            unrestricted=_as_flag(raw_scope.get("unrestricted"), user_id=user_id),
            departments=_as_frozenset(raw_scope.get("departments")),
            categories=_as_frozenset(raw_scope.get("categories")),
            brands=_as_frozenset(raw_scope.get("brands")),
        )
    return scopes


@lru_cache(maxsize=1)
def _cached_scopes(path: str | None) -> dict[str, Scope]:
    return load_scopes(path)


def get_scope(user_id: str, config_path: Path | str | None = None) -> Scope:
    """Look up one user's scope.

    Raises UnknownUserError if the user has no entry — callers must not fall
    back to an unrestricted scope.
    """
    scopes = _cached_scopes(str(config_path) if config_path else None)
    try:
        return scopes[user_id]
    except KeyError:
        raise UnknownUserError(
            f"no entitlement entry for user {user_id!r}; "
            f"known users: {sorted(scopes)}"
        ) from None
