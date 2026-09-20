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
import json
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

# Result columns holding a raw customer id. Pseudonymised before the rows reach
# the model — see pseudonymize_customer_ids().
CUSTOMER_ID_COLUMNS: Final[frozenset[str]] = frozenset(
    {"user_id", "customer_id", "userid", "customerid"}
)

# --------------------------------------------------------------------------
# Text scrubbing
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

# Street suffixes thelook actually generates (Faker's US address list), plus the
# common abbreviations. Matched case-SENSITIVELY as part of the pattern below.
_STREET_SUFFIXES = (
    "Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Ways|"
    "Place|Pl|Terrace|Trail|Trace|Track|Parkway|Pkwy|Circle|Cir|Square|Sq|Highway|Hwy|"
    "Crossing|Crossroad|Junction|Station|Mission|Point|Port|Ridge|Ridges|River|Run|Row|"
    "Shoal|Shoals|Shore|Shores|Spring|Springs|Summit|Union|Valley|Valleys|Via|Viaduct|"
    "View|Views|Village|Ville|Vista|Walk|Wall|Well|Wells|Park|Parks|Pass|Path|Pike|"
    "Pine|Pines|Plain|Plains|Plaza|Prairie|Rapids|Rest|Field|Fields|Fall|Falls|Ferry|"
    "Flat|Flats|Ford|Forest|Forge|Fork|Forks|Fort|Garden|Gardens|Gateway|Glen|Glens|"
    "Green|Grove|Harbor|Haven|Heights|Hill|Hills|Hollow|Inlet|Island|Islands|Isle|Key|"
    "Keys|Knoll|Knolls|Lake|Lakes|Land|Landing|Light|Lights|Loaf|Lock|Locks|Lodge|Loop|"
    "Manor|Manors|Meadow|Meadows|Mews|Mill|Mills|Motorway|Mount|Mountain|Neck|Orchard|"
    "Overpass|Rue|Skyway|Spur|Stravenue|Stream|Throughway|Tunnel|Turnpike|Underpass|"
    "Branch|Bridge|Brook|Brooks|Burg|Bypass|Camp|Canyon|Cape|Causeway|Center|Cliff|"
    "Cliffs|Club|Common|Corner|Corners|Course|Cove|Creek|Crescent|Crest|Dale|Dam|Divide|"
    "Estate|Estates|Expressway|Extension|Route|Passage|Ramp|Rapid|Radial|Freeway|Turn"
)
# A house number, up to four CAPITALISED words, then a suffix — also capitalised.
#
# The case-insensitive version of this redacted ordinary executive prose, because
# every phrase of the shape "<number> <words> <suffix-word>" matched: "3 levers to
# drive repeat purchases" and "in 2025 revenue grew in every way" both came back
# as [REDACTED_ADDRESS]. An answer mangled by its own safety layer is a bug the
# user sees on every report, so the pattern now requires address-shaped casing.
_STREET_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w'-]*\.?\s+){0,4}"
    # Trailing "s?" because thelook generates plurals of every suffix — Points,
    # Spurs, Mountains, Walks — and listing only the singulars left one address
    # in six untouched.
    rf"(?:{_STREET_SUFFIXES})s?\b\.?(?:\s+(?:Apt|Suite|Ste|Unit)\.?\s*[\w-]+)?"
)
# Requires real phone punctuation: parentheses, or separators between the groups.
# A bare run of ten digits is far more often a byte count or an id than a number.
_PHONE_RE = re.compile(
    # (?<!\d) rather than \b in front: a "(" is not a word character, so a \b
    # there can never match and "(312) 555-0134" slipped through untouched.
    r"(?<!\d)(?:\+?1[-.\s])?\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b"
    r"|\b(?:\+?1[-.\s])?\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
)
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
            cleaned = out[col].map(_scrub_value)
            if not cleaned.equals(out[col]):
                applied.append(f"redacted_values:{col}")
            out[col] = cleaned
    return out, applied


def _scrub_value(value: object) -> object:
    """Scrub one cell, including STRUCT and ARRAY cells.

    A struct arrives as a dict and an array as a list, and neither is a str — so
    scrubbing only strings let anything nested through untouched. They are
    rendered to JSON, scrubbed, and kept as text: the model reads them the same
    way, and there is no longer a container the redaction cannot see into.
    """
    if isinstance(value, str):
        return scrub_text(value)[0]
    if isinstance(value, (dict, list, tuple)) or hasattr(value, "tolist"):
        try:
            as_json = json.dumps(value, default=str)
        except (TypeError, ValueError):
            as_json = str(value)
        return scrub_text(as_json)[0]
    return value


def pseudonymize_customer_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Replace raw customer ids with stable pseudonyms, in place, by column name.

    "Top customers" is a required capability and the prompt promises the answer
    names customers as CUST-xxxxxx — but nothing applied that, so raw user_ids
    reached the model and the model had no choice but to invent the pseudonyms
    it had been told to use. Applying it here makes the promise true for every
    query, without the model having to cooperate.

    Column names rather than values, because only the query knows which integer
    is a customer: `id` alone could be a product or an order.
    """
    targets = [c for c in df.columns if str(c).lower() in CUSTOMER_ID_COLUMNS]
    if not targets:
        return df, []
    out = df.copy()
    for column in targets:
        out[column] = out[column].map(
            lambda v: v if pd.isna(v) else pseudonymize(int(v) if isinstance(v, float) else v)
        )
    return out, [str(c) for c in targets]


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
