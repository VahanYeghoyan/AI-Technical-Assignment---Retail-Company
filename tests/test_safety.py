"""Tests for the deterministic safety layers.

These run with no credentials: no BigQuery, no Gemini quota. That is deliberate
— the safety guarantees must be verifiable on any machine, and in CI, without
reaching a third party.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from retail_agent.safety import pii
from retail_agent.safety.scope import Scope, UnknownUserError, get_scope, load_scopes
from retail_agent.safety.sql_guard import SqlGuardError, validate_and_rewrite

UNRESTRICTED = Scope(user_id="ceo", unrestricted=True)
WOMENS = Scope(user_id="maya", departments=frozenset({"Women"}))
DENIM = Scope(user_id="daniel", categories=frozenset({"Jeans", "Swim"}))


def guard(sql: str, scope: Scope = UNRESTRICTED):
    return validate_and_rewrite(sql, scope)


def reason(sql: str, scope: Scope = UNRESTRICTED) -> str:
    with pytest.raises(SqlGuardError) as err:
        validate_and_rewrite(sql, scope)
    return err.value.reason


# ---------------------------------------------------------------------------
# Read-only enforcement
# ---------------------------------------------------------------------------


def test_plain_select_is_allowed():
    result = guard("SELECT COUNT(*) AS n FROM orders")
    assert "orders" in result.tables
    assert result.rewritten is False  # unrestricted scope needs no rewrite


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders WHERE order_id = 1",
        "UPDATE orders SET status = 'Complete'",
        "INSERT INTO orders (order_id) VALUES (1)",
        "DROP TABLE orders",
        "CREATE TABLE evil AS SELECT * FROM orders",
    ],
)
def test_write_statements_are_rejected(sql):
    assert reason(sql) == "not_read_only"


def test_stacked_statements_are_rejected():
    assert reason("SELECT 1 FROM orders; DROP TABLE orders") == "multiple_statements"


def test_unparseable_sql_is_rejected_with_a_fixable_reason():
    # The self-correction loop keys off this reason code.
    assert reason("SELECT FROM WHERE orders") == "syntax_error"


# ---------------------------------------------------------------------------
# Table allowlist
# ---------------------------------------------------------------------------


def test_unknown_table_is_rejected():
    assert reason("SELECT * FROM inventory_items") == "table_not_allowed"


def test_other_project_is_rejected():
    sql = "SELECT id FROM `some-other-project.thelook_ecommerce.products`"
    assert reason(sql) == "table_not_allowed"


def test_other_dataset_is_rejected():
    sql = "SELECT id FROM `bigquery-public-data.other_dataset.products`"
    assert reason(sql) == "table_not_allowed"


def test_query_touching_no_table_is_rejected():
    assert reason("SELECT 1") == "no_tables"


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


def test_pii_column_in_projection_is_rejected():
    assert reason("SELECT email FROM users") == "pii_column"


def test_pii_column_in_where_is_rejected():
    # The membership-oracle attack: no PII is displayed, but the count leaks
    # whether a specific person exists in the data.
    sql = "SELECT COUNT(*) AS n FROM users WHERE email = 'someone@example.com'"
    assert reason(sql) == "pii_column"


def test_pii_column_in_join_or_order_by_is_rejected():
    sql = "SELECT u.id FROM users AS u ORDER BY u.last_name"
    assert reason(sql) == "pii_column"


def test_geography_column_is_treated_as_pii():
    assert reason("SELECT user_geom FROM users") == "pii_column"


def test_star_over_users_is_rejected():
    assert reason("SELECT * FROM users") == "star_over_users"


def test_star_over_non_pii_table_is_allowed():
    assert guard("SELECT * FROM products").tables == frozenset({"products"})


def test_non_pii_user_columns_are_allowed():
    result = guard("SELECT state, COUNT(*) AS n FROM users GROUP BY state")
    assert "users" in result.tables


# ---------------------------------------------------------------------------
# Entitlement rewriting
# ---------------------------------------------------------------------------


def test_scope_is_injected_for_restricted_user():
    result = guard("SELECT COUNT(*) AS n FROM products", WOMENS)
    assert result.rewritten is True
    assert "department IN ('Women')" in result.sql


def test_unrestricted_user_sql_is_untouched():
    assert guard("SELECT COUNT(*) AS n FROM products", UNRESTRICTED).rewritten is False


def test_scope_propagates_to_orders_via_exists():
    result = guard("SELECT COUNT(*) AS n FROM orders", WOMENS)
    assert "EXISTS" in result.sql.upper()
    assert "department IN ('Women')" in result.sql


def test_scope_propagates_to_users():
    result = guard("SELECT state FROM users GROUP BY state", DENIM)
    assert "EXISTS" in result.sql.upper()
    assert "category IN ('Jeans', 'Swim')" in result.sql


def test_aliases_are_preserved_so_outer_references_still_resolve():
    sql = "SELECT p.brand, COUNT(*) AS n FROM products AS p GROUP BY p.brand"
    result = guard(sql, WOMENS)
    assert "AS p" in result.sql
    assert result.rewritten is True


def test_joins_scope_every_table():
    sql = """
        SELECT p.brand, SUM(oi.sale_price) AS revenue
        FROM order_items AS oi
        JOIN products AS p ON p.id = oi.product_id
        GROUP BY p.brand
    """
    result = guard(sql, WOMENS)
    # Both the order_items and the products reference must be scoped.
    assert result.sql.count("department IN ('Women')") >= 2


def test_cte_references_are_scoped_too():
    sql = """
        WITH monthly AS (
            SELECT DATE_TRUNC(DATE(created_at), MONTH) AS m, SUM(sale_price) AS rev
            FROM order_items
            GROUP BY m
        )
        SELECT * FROM monthly ORDER BY m
    """
    result = guard(sql, WOMENS)
    assert result.rewritten is True
    assert "department IN ('Women')" in result.sql


def test_literal_quoting_is_escaped():
    # Levi's contains an apostrophe: naive string building would produce invalid
    # or injectable SQL here.
    scope = Scope(user_id="priya", brands=frozenset({"Levi's"}))
    result = guard("SELECT COUNT(*) AS n FROM products", scope)
    assert "Levi\\'s" in result.sql or "Levi''s" in result.sql


def test_subquery_in_where_is_scoped():
    sql = """
        SELECT COUNT(*) AS n FROM orders
        WHERE user_id IN (SELECT id FROM users WHERE state = 'Texas')
    """
    result = guard(sql, WOMENS)
    assert result.rewritten is True
    assert result.sql.upper().count("EXISTS") >= 2


# ---------------------------------------------------------------------------
# Scope config
# ---------------------------------------------------------------------------


def test_entitlements_config_loads():
    scopes = load_scopes()
    assert {"ceo", "maya", "daniel", "priya", "sam"} <= set(scopes)
    assert scopes["ceo"].unrestricted is True
    assert scopes["maya"].departments == frozenset({"Women"})


def test_unknown_user_is_fatal_rather_than_open():
    with pytest.raises(UnknownUserError):
        get_scope("not-a-real-user")


def test_scope_with_no_dimensions_is_rejected():
    # Guards against a YAML typo silently granting the full catalogue.
    with pytest.raises(ValueError):
        Scope(user_id="typo")


def test_scope_description_is_human_readable():
    assert "Women" in WOMENS.describe()
    assert UNRESTRICTED.describe() == "the full product catalogue"


# ---------------------------------------------------------------------------
# PII scrubbing
# ---------------------------------------------------------------------------


def test_scrub_text_redacts_email():
    cleaned, applied = pii.scrub_text("Contact sarah.jones@example.com about it")
    assert "sarah.jones@example.com" not in cleaned
    assert "redacted_email" in applied


def test_scrub_text_redacts_address_and_location():
    cleaned, _ = pii.scrub_text("She lives at 742 Evergreen Terrace")
    assert "Evergreen" not in cleaned
    cleaned, _ = pii.scrub_text("Pinned at 40.71280, -74.00600")
    assert "REDACTED_LOCATION" in cleaned


def test_scrub_text_leaves_clean_analysis_alone():
    text = "Revenue in Texas fell 12% to $1.2M in Q1."
    assert pii.scrub_text(text)[0] == text


def test_scrub_dataframe_drops_pii_columns():
    df = pd.DataFrame({"email": ["a@b.com"], "state": ["Texas"], "revenue": [10.0]})
    cleaned, applied = pii.scrub_dataframe(df)
    assert "email" not in cleaned.columns
    assert "state" in cleaned.columns
    assert any(a.startswith("dropped_columns") for a in applied)


def test_scrub_dataframe_redacts_values_in_free_text_columns():
    df = pd.DataFrame({"note": ["ping bob@example.com"]})
    cleaned, applied = pii.scrub_dataframe(df)
    assert "bob@example.com" not in cleaned["note"].iloc[0]
    assert any(a.startswith("redacted_values") for a in applied)


def test_pseudonymize_is_stable_and_salted():
    a = pii.pseudonymize(42, salt="s")
    assert a == pii.pseudonymize(42, salt="s")  # stable: follow-ups still resolve
    assert a != pii.pseudonymize(43, salt="s")  # distinct per customer
    assert a != pii.pseudonymize(42, salt="different")  # salt actually binds
    # Format, not substring: a hex digest can contain the id's digits by chance
    # (CUST-842560 for id 42), which says nothing about reversibility.
    assert re.fullmatch(r"CUST-[0-9A-F]{6}", a)


def test_pseudonymize_column_renames_and_masks():
    df = pd.DataFrame({"user_id": [1, 2], "revenue": [10.0, 20.0]})
    out = pii.pseudonymize_column(df)
    assert "customer" in out.columns and "user_id" not in out.columns
    assert out["customer"].iloc[0].startswith("CUST-")
