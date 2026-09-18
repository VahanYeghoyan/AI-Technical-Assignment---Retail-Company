"""PII registry, output scrubbing and customer pseudonymisation.

Three layers, because any one of them alone is defeatable:

  1. Column level  — sql_guard rejects any query that references a PII column
                     anywhere (projection, WHERE, JOIN, ORDER BY). This blocks
                     both display and oracle attacks such as
                     "SELECT COUNT(*) ... WHERE email = 'x@y.com'".
  2. Result level  — scrub_dataframe() runs over every result set before it is
                     shown OR sent to the LLM. Raw PII never enters the model's
                     context, which matters because the AI Studio free tier may
                     retain prompts.
  3. Text level    — scrub_text() runs over the model's final answer, catching
                     anything reconstructed or hallucinated into prose.

Column lists are the real thelook_ecommerce schema, verified 2026-09-18.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Final

import pandas as pd

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# Columns that identify a natural person. Forbidden everywhere, always.
# users.user_geom is GEOGRAPHY — a precise home location, so it is PII even
# though it carries no name.
PII_COLUMNS: Final[dict[str, frozenset[str]]] = {
    "users": frozenset(
        {
            "first_name",
            "last_name",
            "email",
            "street_address",
            "postal_code",
            "latitude",
            "longitude",
            "user_geom",
        }
    ),
    "orders": frozenset(),
    "order_items": frozenset(),
    "products": frozenset(),
}

ALL_PII_COLUMNS: Final[frozenset[str]] = frozenset().union(*PII_COLUMNS.values())

# Not identifying on their own, but identifying in combination on a small enough
# group. Permitted, but only in aggregates of at least metrics.yaml
# privacy.min_group_size customers.
QUASI_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    {"city", "state", "country", "age", "gender", "traffic_source"}
)

# --------------------------------------------------------------------------
# Text scrubbing
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# US-style street address, which is the shape thelook generates.
_STREET_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl|Terrace|Trail|Parkway|Pkwy)\b\.?",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
# Lat/long pair, e.g. "40.7128, -74.0060" — enough to locate a household.
_LATLONG_RE = re.compile(r"\b-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}\b")

_REDACTIONS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (_EMAIL_RE, "[REDACTED_EMAIL]"),
    (_LATLONG_RE, "[REDACTED_LOCATION]"),
    (_STREET_RE, "[REDACTED_ADDRESS]"),
    (_PHONE_RE, "[REDACTED_PHONE]"),
)


def scrub_text(text: str) -> tuple[str, list[str]]:
    """Redact PII patterns from free text.

    Returns the cleaned text and the list of redaction kinds applied, so the
    caller can log a pii_redactions metric (Requirement 7) rather than silently
    swallowing the event.
    """
    if not text:
        return text, []
    applied: list[str] = []
    for pattern, placeholder in _REDACTIONS:
        text, n = pattern.subn(placeholder, text)
        if n:
            applied.append(placeholder.strip("[]").lower())
    return text, applied


def scrub_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop PII columns and redact PII values from a result set.

    Columns whose name is a known PII column are dropped outright — a query
    should never have produced them (sql_guard rejects that), so reaching here
    means a defence-in-depth catch worth logging.
    """
    if df.empty:
        return df, []
    applied: list[str] = []
    out = df.copy()

    offending = [c for c in out.columns if c.lower() in ALL_PII_COLUMNS]
    if offending:
        out = out.drop(columns=offending)
        applied.append(f"dropped_columns:{','.join(sorted(offending))}")

    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(
            out[col]
        ):
            cleaned = out[col].map(
                lambda v: scrub_text(v)[0] if isinstance(v, str) else v
            )
            if not cleaned.equals(out[col]):
                applied.append(f"redacted_values:{col}")
            out[col] = cleaned
    return out, applied


# --------------------------------------------------------------------------
# Pseudonymisation
# --------------------------------------------------------------------------


def pseudonymize(user_id: object, salt: str | None = None) -> str:
    """Map a raw customer id to a stable, non-reversible handle.

    "Top customers" is a required capability, but names and emails are
    forbidden — so customers surface as CUST-xxxxxx. The mapping is stable
    within a deployment (same salt, same id, same handle), which is what makes
    follow-up questions like "tell me more about the third one" work, and is not
    reversible without the salt.
    """
    salt = salt if salt is not None else os.getenv("PSEUDONYM_SALT", "dev-salt")
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return f"CUST-{digest[:6].upper()}"


def pseudonymize_column(df: pd.DataFrame, column: str = "user_id") -> pd.DataFrame:
    """Replace a raw customer id column with pseudonymous handles."""
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = out[column].map(pseudonymize)
    return out.rename(columns={column: "customer"})
