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


def test_bare_table_names_are_qualified_for_unrestricted_users():
    # Regression: BigQuery rejects `FROM orders` with "must be qualified with a
    # dataset". Scoped users were saved by the rewrite (its subqueries are
    # qualified), so the UNRESTRICTED user was the one that failed — backwards.
    #
    # The identifiers must come out QUOTED. This assertion used to normalise the
    # unquoted form into the quoted one before comparing, which made it pass
    # against SQL BigQuery rejects: bare `bigquery-public-data` is read as
    # arithmetic on three identifiers, not as a project id.
    result = guard("SELECT COUNT(*) AS n FROM orders", UNRESTRICTED)
    assert "`bigquery-public-data`.`thelook_ecommerce`.`orders`" in result.sql
    assert "FROM orders" not in result.sql


def test_two_part_names_get_the_project_added():
    # `thelook_ecommerce.orders` has no project, so BigQuery resolves the dataset
    # inside the BILLING project, where it does not exist — a 404 for exactly the
    # unrestricted users the qualification above was meant to stop failing.
    result = guard("SELECT COUNT(*) AS n FROM thelook_ecommerce.orders", UNRESTRICTED)
    assert "`bigquery-public-data`.`thelook_ecommerce`.`orders`" in result.sql


# ---------------------------------------------------------------------------
# CTE names must not buy a table reference a free pass
# ---------------------------------------------------------------------------


def test_a_cte_name_does_not_unlock_a_qualified_table_of_the_same_name():
    # The CTE is never used. It exists only so that the qualified reference
    # underneath it matches a known CTE name and skips the allowlist — which is
    # how `events.ip_address`, and any table in any project, became readable.
    sql = """
        WITH events AS (SELECT 1 AS x FROM orders)
        SELECT user_id, ip_address
        FROM `bigquery-public-data.thelook_ecommerce.events`
    """
    assert reason(sql) == "table_not_allowed"


def test_a_cte_name_does_not_unlock_another_project():
    sql = """
        WITH secret AS (SELECT 1 AS x FROM orders)
        SELECT * FROM `some-other-project.hr.secret`
    """
    assert reason(sql) == "table_not_allowed"


def test_wildcard_table_names_are_rejected():
    # `order_item*` is a BigQuery wildcard table that resolves to order_items
    # itself. Quoted as a CTE name it skipped the entitlement rewrite, and a
    # Women's-division user could read the whole company's revenue through it.
    sql = """
        WITH `order_item*` AS (SELECT 1 AS x FROM orders)
        SELECT SUM(sale_price) AS rev
        FROM `bigquery-public-data.thelook_ecommerce.order_item*`
    """
    assert reason(sql, WOMENS) == "table_not_allowed"


def test_a_real_cte_reference_still_works():
    # The bare reference is what a CTE legitimately looks like; only qualified
    # ones are suspect.
    result = guard(
        "WITH monthly AS (SELECT order_id FROM orders) SELECT COUNT(*) AS n FROM monthly",
        WOMENS,
    )
    assert result.rewritten is True


# ---------------------------------------------------------------------------
# Whole-row references (a column-name check cannot see these)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TO_JSON_STRING(u) AS j FROM users u",
        "SELECT u AS person FROM users u",
        "SELECT ARRAY_AGG(u LIMIT 3) AS a FROM users u",
        "SELECT FORMAT('%T', u) AS s FROM users u",
        "SELECT JSON_VALUE(TO_JSON_STRING(u), '$.first_name') AS fn FROM users u",
        # The membership oracle again, rebuilt without naming a PII column.
        "SELECT COUNT(*) AS n FROM users u "
        "WHERE STRPOS(TO_JSON_STRING(u), 'alice@example.com') > 0",
        # No alias: the table's own name is the row.
        "SELECT TO_JSON_STRING(users) AS j FROM users",
        # A parenthesised join is parsed as a Subquery wrapping the tables, not
        # as plain FROM/JOIN sources. BigQuery accepts it, and this shape once
        # returned first/last names and postcodes that no other rule saw.
        "SELECT TO_JSON_STRING(u) AS j "
        "FROM (users AS u JOIN orders AS o ON o.user_id = u.id)",
        # ...and nested one level down, in the JOIN position.
        "SELECT TO_JSON_STRING(u) AS j FROM order_items AS oi "
        "JOIN (users AS u JOIN orders AS o ON o.user_id = u.id) "
        "ON oi.order_id = o.order_id",
    ],
)
def test_whole_row_references_are_rejected(sql):
    assert reason(sql) == "row_reference"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TO_JSON_STRING(u) AS j FROM users u",
        "SELECT TO_JSON_STRING(u) AS j "
        "FROM (users AS u JOIN orders AS o ON o.user_id = u.id)",
    ],
)
def test_row_reference_check_survives_scoping(sql):
    # Restricted users get the same answer as unrestricted ones: the rewrite
    # happens after validation, so it cannot be used to sneak one past.
    assert reason(sql, WOMENS) == "row_reference"


def test_a_parenthesised_join_of_named_columns_is_allowed_and_scoped():
    sql = (
        "SELECT u.state, COUNT(*) AS n "
        "FROM (users AS u JOIN orders AS o ON o.user_id = u.id) GROUP BY u.state"
    )
    result = guard(sql, WOMENS)
    assert result.tables == frozenset({"users", "orders"})
    assert result.rewritten is True
    assert "department IN ('Women')" in result.sql


def test_a_column_sharing_a_table_name_is_not_a_row_reference():
    # `orders` here is a count, not a row. The check is scoped to the SELECT that
    # actually has the table in its FROM, so this must still be allowed.
    sql = """
        WITH t AS (SELECT COUNT(*) AS orders FROM orders)
        SELECT orders FROM t
    """
    assert guard(sql).tables == frozenset({"orders"})


def test_a_row_reference_to_a_cte_is_allowed():
    # A CTE's projection has already passed every rule here, so its rows hold no
    # column the user could not have selected directly.
    sql = """
        WITH t AS (SELECT id, age FROM users)
        SELECT COUNT(*) AS n FROM t
    """
    assert "users" in guard(sql).tables


def test_already_qualified_tables_are_left_alone():
    sql = "SELECT COUNT(*) AS n FROM `bigquery-public-data.thelook_ecommerce.orders`"
    result = guard(sql, UNRESTRICTED)
    assert result.sql.count("thelook_ecommerce") == 1


def test_qualification_preserves_aliases():
    sql = "SELECT o.order_id FROM orders AS o WHERE o.status = 'Complete'"
    result = guard(sql, UNRESTRICTED)
    assert "AS o" in result.sql
    assert "thelook_ecommerce" in result.sql


def test_cte_names_are_not_qualified():
    sql = """
        WITH recent AS (SELECT order_id FROM orders)
        SELECT COUNT(*) AS n FROM recent
    """
    result = guard(sql, UNRESTRICTED)
    # The CTE reference must stay bare; only the real table gets qualified.
    assert "thelook_ecommerce.recent" not in result.sql
    assert "thelook_ecommerce" in result.sql


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


@pytest.mark.parametrize(
    "prose",
    [
        # Every one of these came back mutilated when the address pattern was
        # case-insensitive and the phone pattern accepted ten bare digits:
        # "<number> <any words> <suffix word>" describes most business writing.
        "Q2 action items: focus on 3 levers to drive repeat purchases.",
        "In 2025 revenue grew in every way we measure it.",
        "We sold 279 units at the St. Louis store.",
        "Total bytes scanned: 1005593512 for the full year.",
        "Top 3 brands by revenue, 2026 to date, excluding 44,769 cancelled rows.",
        "Action: review the 5 worst-performing SKUs by margin this quarter.",
    ],
)
def test_scrubbing_leaves_ordinary_executive_prose_alone(prose):
    assert pii.scrub_text(prose) == (prose, [])


@pytest.mark.parametrize(
    "text,marker",
    [
        # …while still catching the real thing, in the shapes thelook generates.
        ("Ship to 6389 Pine Drive, apartment 4.", "[REDACTED_ADDRESS]"),
        ("Home is 8395 Hall Points Apt. 320.", "[REDACTED_ADDRESS]"),
        ("Mail 46177 Bell Crossing Suite 12 today.", "[REDACTED_ADDRESS]"),
        ("Call the vendor on (312) 555-0134.", "[REDACTED_PHONE]"),
        ("Reach ops at 312-555-0134.", "[REDACTED_PHONE]"),
        ("Contact jane.doe@example.com now.", "[REDACTED_EMAIL]"),
        ("Seen at 40.7128, -74.0060 last week.", "[REDACTED_LOCATION]"),
    ],
)
def test_real_personal_data_is_still_redacted(text, marker):
    assert marker in pii.scrub_text(text)[0]


def test_struct_and_array_cells_are_scrubbed_too():
    # A STRUCT arrives as a dict and an ARRAY as a list. Scrubbing only str cells
    # meant anything nested inside one was passed through untouched.
    df = pd.DataFrame(
        {"profile": [{"note": "mail bob@example.com"}], "tags": [["a@b.com"]]}
    )
    cleaned, applied = pii.scrub_dataframe(df)
    assert "bob@example.com" not in str(cleaned["profile"].iloc[0])
    assert "a@b.com" not in str(cleaned["tags"].iloc[0])
    assert len(applied) == 2


def test_customer_id_columns_are_pseudonymised_in_place():
    df = pd.DataFrame({"user_id": [1, 2], "spend": [10.0, 20.0]})
    out, columns = pii.pseudonymize_customer_ids(df)

    assert columns == ["user_id"]
    assert out["user_id"].tolist() == [pii.pseudonymize(1), pii.pseudonymize(2)]
    # The column keeps its name, so two id columns in one result cannot collide.
    assert list(out.columns) == ["user_id", "spend"]


def test_pseudonymisation_leaves_other_id_columns_alone():
    # Only the query knows which integer is a customer; `id` here is a product.
    df = pd.DataFrame({"id": [7], "brand": ["Levi's"]})
    out, columns = pii.pseudonymize_customer_ids(df)
    assert columns == [] and out["id"].iloc[0] == 7
