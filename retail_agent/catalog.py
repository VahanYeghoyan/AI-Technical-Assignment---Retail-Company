"""Schema catalog and metric glossary.

Two jobs:

  * Answer "what data is available?" without a round trip to BigQuery. The four
    tables are fixed and small, so the catalog is a static, hand-annotated
    description — which is better than a live INFORMATION_SCHEMA dump anyway,
    because it carries the join keys, the grain, and the traps (status values,
    future-dated rows) that raw column types do not.

  * Feed the prompt. The model cannot write correct SQL without knowing that
    revenue lives on order_items.sale_price rather than orders, or that a
    quarter of order_items rows are cancelled/returned.

PII columns are listed here as FORBIDDEN rather than omitted. Hiding them makes
the model invent them; naming them as off-limits makes it route around them.
Column names and types were verified against the live schema on 2026-09-18.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from retail_agent.safety.pii import PII_COLUMNS

_DEFAULT_METRICS = Path(__file__).resolve().parents[1] / "config" / "metrics.yaml"

DATASET = "bigquery-public-data.thelook_ecommerce"


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    note: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    grain: str
    columns: tuple[Column, ...]
    joins: tuple[str, ...] = ()

    def render(self) -> str:
        forbidden = PII_COLUMNS.get(self.name, frozenset())
        lines = [f"{self.name} — one row per {self.grain}"]
        for column in self.columns:
            if column.name in forbidden:
                continue
            suffix = f"  # {column.note}" if column.note else ""
            lines.append(f"    {column.name}: {column.type}{suffix}")
        if forbidden:
            lines.append(
                f"    FORBIDDEN (personal data, never query): "
                f"{', '.join(sorted(forbidden))}"
            )
        for join in self.joins:
            lines.append(f"    join: {join}")
        return "\n".join(lines)


TABLES: tuple[Table, ...] = (
    Table(
        name="orders",
        grain="customer order",
        columns=(
            Column("order_id", "INTEGER", "primary key"),
            Column("user_id", "INTEGER", "-> users.id"),
            Column("status", "STRING", "Shipped/Complete/Processing/Cancelled/Returned"),
            Column("gender", "STRING"),
            Column("created_at", "TIMESTAMP", "order placed"),
            Column("returned_at", "TIMESTAMP"),
            Column("shipped_at", "TIMESTAMP"),
            Column("delivered_at", "TIMESTAMP"),
            Column("num_of_item", "INTEGER"),
        ),
        joins=("orders.order_id = order_items.order_id",),
    ),
    Table(
        name="order_items",
        grain="item within an order — THE revenue table",
        columns=(
            Column("id", "INTEGER", "primary key"),
            Column("order_id", "INTEGER", "-> orders.order_id"),
            Column("user_id", "INTEGER", "-> users.id"),
            Column("product_id", "INTEGER", "-> products.id"),
            Column("inventory_item_id", "INTEGER"),
            Column("status", "STRING", "same values as orders.status"),
            Column("created_at", "TIMESTAMP"),
            Column("shipped_at", "TIMESTAMP"),
            Column("delivered_at", "TIMESTAMP"),
            Column("returned_at", "TIMESTAMP"),
            Column("sale_price", "FLOAT", "revenue per unit — SUM this"),
        ),
        joins=(
            "order_items.product_id = products.id",
            "order_items.user_id = users.id",
        ),
    ),
    Table(
        name="products",
        grain="catalogue product",
        columns=(
            Column("id", "INTEGER", "primary key"),
            Column("cost", "FLOAT", "unit cost — margin = sale_price - cost"),
            Column("category", "STRING", "e.g. Jeans, Swim, Intimates"),
            Column("name", "STRING"),
            Column("brand", "STRING"),
            Column("retail_price", "FLOAT", "list price, NOT what sold"),
            Column("department", "STRING", "Men or Women"),
            Column("sku", "STRING"),
            Column("distribution_center_id", "INTEGER"),
        ),
    ),
    Table(
        name="users",
        grain="customer",
        columns=(
            Column("id", "INTEGER", "primary key"),
            Column("age", "INTEGER", "aggregate only"),
            Column("gender", "STRING", "aggregate only"),
            Column("state", "STRING", "aggregate only"),
            Column("city", "STRING", "aggregate only"),
            Column("country", "STRING", "aggregate only"),
            Column("traffic_source", "STRING", "acquisition channel"),
            Column("created_at", "TIMESTAMP", "signup date"),
        ),
    ),
)

TABLES_BY_NAME = {table.name: table for table in TABLES}


@lru_cache(maxsize=1)
def _metrics_doc(path: str | None = None) -> dict[str, Any]:
    target = Path(path or os.getenv("METRICS_PATH") or _DEFAULT_METRICS)
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def render_schema() -> str:
    """The schema block for the system prompt."""
    header = (
        f"Dataset: `{DATASET}` (BigQuery standard SQL, read-only).\n"
        "Always fully qualify tables as "
        f"`{DATASET}.<table>`.\n"
    )
    return header + "\n" + "\n\n".join(table.render() for table in TABLES)


def render_metrics() -> str:
    """The metric-definition block for the system prompt."""
    doc = _metrics_doc()
    lines: list[str] = ["Metric definitions — use these exactly, and name the one you used:"]

    for metric, body in (doc.get("metrics") or {}).items():
        definition = str(body.get("definition", "")).strip()
        lines.append(f"  {metric}: {definition}")
        if body.get("filters"):
            lines.append(f"      filters: {body['filters']}")

    notes = doc.get("dataset_notes") or {}
    if notes.get("clamp_rule"):
        lines += ["", f"CRITICAL: {str(notes['clamp_rule']).strip()}"]

    privacy = doc.get("privacy") or {}
    if privacy.get("min_group_size"):
        lines.append(
            f"Suppress any customer-level group smaller than "
            f"{privacy['min_group_size']} (report it as suppressed)."
        )
    return "\n".join(lines)


def min_group_size() -> int:
    return int((_metrics_doc().get("privacy") or {}).get("min_group_size", 5))


def describe_schema(table: str | None = None) -> str:
    """Tool implementation: answer questions about the data model."""
    if table is None:
        return render_schema()
    found = TABLES_BY_NAME.get(table.lower())
    if found is None:
        return (
            f"There is no table called {table!r}. Available tables: "
            f"{', '.join(sorted(TABLES_BY_NAME))}."
        )
    return found.render()
