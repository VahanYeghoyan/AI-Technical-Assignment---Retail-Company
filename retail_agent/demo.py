"""A scripted model for the offline demo (LLM_PROVIDER=stub).

It reads the question for a few keywords, makes the tool call a real model
would make, and then narrates what the tool sent back. It is not an analyst —
every data question gets the same query — but everything it sets in motion is
the real system: the SQL guard, entitlement rewriting, the cost gate, BigQuery
when credentials exist, the report library, the confirmation broker and the
traces. That is what makes the four prototype requirements checkable on a
machine with no model access at all.

What to try, and what it shows:

  "what data is available?"         describe_schema, no database needed
  "show me our customers' email     the SQL guard refusing a PII query before
   addresses"                       it reaches BigQuery
  "top brands this year"            scope rewriting and the cost gate; real rows
                                    with credentials, a graceful stop without
  "create a report on Q3"           save_report, then /reports
  "delete the reports from this     the proposal, the typed confirmation and
   conversation"                    /undo
"""

from __future__ import annotations

import json
import re
from typing import Any

from retail_agent.llm import FunctionCall, LLMResponse
from retail_agent.prompt import report_sections, section_heading

MODEL_NAME = "stub-demo"

DEMO_SQL = """\
SELECT p.brand, ROUND(SUM(oi.sale_price), 2) AS revenue
FROM `bigquery-public-data.thelook_ecommerce.order_items` AS oi
JOIN `bigquery-public-data.thelook_ecommerce.products` AS p ON p.id = oi.product_id
WHERE oi.status NOT IN ('Cancelled', 'Returned')
  AND oi.created_at >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), YEAR)
  AND oi.created_at < CURRENT_TIMESTAMP()
GROUP BY p.brand
ORDER BY revenue DESC
LIMIT 5"""

# What a curious or hostile user might ask for. The query written for it names
# the columns outright, so it is the guard — not this script — that refuses.
PII_PROBE_SQL = (
    "SELECT first_name, last_name, email "
    "FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 10"
)

HELP = """\
This is the **offline demo**: the model is scripted, so it only understands a
few kinds of request — but everything it triggers is the real system. Try:

- *what data is available?*
- *show me our customers' email addresses* — the SQL guard refuses
- *top brands by revenue this year* — scoped to your entitlements
- *create a report on this quarter* — then `/reports`
- *delete the reports from this conversation* — then `/undo`

Set `LLM_PROVIDER=vertex` (or `gemini`) for the real model."""

_SCHEMA_WORDS = ("what data", "schema", "tables", "columns", "available", "structure")
_PII_WORDS = ("email", "address", "phone", "postcode", "postal", "zip code",
              "first name", "last name", "full name")
_DATA_WORDS = ("revenue", "brand", "sales", "sold", "top", "spend", "customer",
               "order", "product", "margin", "churn", "month", "year", "quarter")
_REPORT_VERBS = ("create", "write", "make", "save", "build", "draft", "generate")
_SELECTOR_RE = re.compile(
    r"\b(?:mentioning|mention|about|containing|that mention|named)\s+"
    r"[\"'“]?(?P<text>[^\"'”?.!]+)",
    re.IGNORECASE,
)
# What a report is about: "a report on denim", "a report for Q3 with actions".
# The README's own example is "create a report on denim", which the selector
# pattern above does not match, so it was saved as "this quarter" and a
# follow-up "delete reports mentioning denim" found nothing.
_TOPIC_RE = re.compile(
    r"\breport\s+(?:on|about|for|covering|mentioning)\s+[\"'“]?"
    r"(?P<text>[^\"'”?.!,]+?)(?=\s+(?:with|including|and)\b|[\"'”?.!,]|$)",
    re.IGNORECASE,
)


def demo_responder(system: str, contents: list[dict[str, Any]]) -> LLMResponse:  # noqa: ARG001
    """StubProvider responder: one tool call per question, then a narration."""
    if contents and _is_tool_results(contents[-1]):
        return _narrate(contents[-1]["parts"])
    return _plan(_latest_question(contents))


# -- planning --------------------------------------------------------------


def _plan(question: str) -> LLMResponse:
    q = question.lower()

    if any(word in q for word in ("delete", "remove")) and "report" in q:
        return _call("propose_delete_reports", **_deletion_args(question))
    if "report" in q and any(verb in q for verb in _REPORT_VERBS):
        return _call("save_report", **_report_args(question))
    if "report" in q and any(word in q for word in ("list", "show", "my reports")):
        return _call("list_reports")
    if any(word in q for word in _PII_WORDS):
        return _call("run_analysis_sql", sql=PII_PROBE_SQL, purpose="look up a customer")
    if any(word in q for word in _SCHEMA_WORDS):
        return _call("describe_schema")
    if any(word in q for word in _DATA_WORDS):
        return _call(
            "run_analysis_sql", sql=DEMO_SQL,
            purpose="top brands by realised revenue, year to date",
        )
    return _text(HELP)


def _deletion_args(question: str) -> dict[str, Any]:
    q = question.lower()
    if "conversation" in q or "this chat" in q or "this session" in q:
        return {"this_conversation": True, "criteria": "made in this conversation"}
    if match := _SELECTOR_RE.search(question):
        text = match.group("text").strip()
        return {"text": text, "criteria": f"mentioning {text}"}
    if re.search(r"\ball\b|\bevery\b", q):
        return {"all_reports": True, "criteria": "all of your reports"}
    # No selector: the broker refuses rather than guessing "everything".
    return {"criteria": "the reports you mentioned"}


def _report_args(question: str) -> dict[str, Any]:
    topic = _TOPIC_RE.search(question) or _SELECTOR_RE.search(question)
    subject = topic.group("text").strip() if topic else "this quarter"
    sections = report_sections() or ("summary", "action_items")
    placeholder = {
        "action_items": "- Re-run this request with a live model to get "
                        "action items tied to measured numbers.",
    }
    body = "\n\n".join(
        f"## {section_heading(s)}\n"
        + placeholder.get(s, "Scripted placeholder — the offline demo writes "
                             "no analysis.")
        for s in sections
    )
    return {
        "title": f"Demo report: {subject}",
        "body": body,
        "entities": [subject],
    }


# -- narration -------------------------------------------------------------


def _narrate(parts: list[dict[str, Any]]) -> LLMResponse:
    lines = [_narrate_one(p["function_response"]) for p in parts]
    return _text("\n\n".join(lines))


def _narrate_one(result: dict[str, Any]) -> str:
    name, payload = result.get("name"), result.get("response") or {}

    if name == "describe_schema":
        return f"Here is the data I can analyse:\n\n```\n{payload.get('schema', '')}\n```"

    if name == "run_analysis_sql":
        if payload.get("error"):
            return (
                "That query was **blocked before it reached BigQuery**:\n\n"
                f"> {payload['error']}\n\n"
                "Customer names, emails and addresses cannot be selected, filtered "
                "on or displayed — ask for an aggregate instead (spend by state, "
                "by age band, by acquisition channel). A live model would "
                "rewrite its query here, up to 3 times."
            )
        if payload.get("row_count", 0) == 0:
            return "The query ran but matched no rows — that is *no data*, not zero revenue."
        return (
            "Top brands by realised revenue (excluding cancelled and returned "
            "items), year to date — within your data scope:\n\n"
            f"{payload.get('rows', '')}\n\n"
            "*Offline demo: every data question gets this one query. `/trace` "
            "shows the scope-rewritten SQL that actually ran.*"
        )

    if name == "save_report":
        if not payload.get("saved"):
            return f"The report was not saved: {payload.get('error', 'unknown reason')}"
        return (
            f"Saved **{payload.get('title')}** ({payload.get('report_id')}). "
            "`/reports` lists your library."
        )

    if name == "list_reports":
        reports = payload.get("reports") or []
        if not reports:
            return "You have no saved reports yet — try *create a report on this quarter*."
        return "Your saved reports:\n\n" + "\n".join(f"- {r}" for r in reports)

    return f"`{name}` returned: {json.dumps(payload, default=str)[:500]}"


# -- helpers ---------------------------------------------------------------


def _latest_question(contents: list[dict[str, Any]]) -> str:
    for content in reversed(contents):
        if content.get("role") == "user":
            for part in content.get("parts", []):
                if "text" in part:
                    return str(part["text"])
    return ""


def _is_tool_results(content: dict[str, Any]) -> bool:
    parts = content.get("parts") or []
    return bool(parts) and all("function_response" in p for p in parts)


def _call(name: str, **args: Any) -> LLMResponse:
    return LLMResponse(function_calls=(FunctionCall(name=name, args=args),), model=MODEL_NAME)


def _text(body: str) -> LLMResponse:
    return LLMResponse(text=body, model=MODEL_NAME)
