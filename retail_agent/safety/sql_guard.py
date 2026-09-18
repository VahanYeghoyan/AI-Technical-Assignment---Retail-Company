"""SQL validation and entitlement rewriting.

Every query the model writes passes through validate_and_rewrite() before it
reaches BigQuery. The model is never trusted to scope or sanitise its own SQL:
it is asked for a plain query over the four tables, and this module decides what
actually runs.

Enforced here, in order:

  1. Exactly one statement          — blocks "SELECT 1; DROP TABLE x"
  2. Read-only                      — only SELECT / WITH / set operations
  3. Table allowlist                — only the four thelook tables, and only in
                                      the bigquery-public-data project
  4. No PII columns anywhere        — projection, WHERE, JOIN, ORDER BY alike,
                                      so membership oracles are blocked too
  5. No SELECT * touching users     — a star would expand to PII columns
  6. Entitlement rewriting          — each table reference is replaced by a
                                      subquery filtered to the caller's scope

Step 6 is what makes Requirement 2's "only data on products related to him"
real. It is AST rewriting via sqlglot rather than string munging, so it survives
aliases, CTEs, subqueries and set operations.

Production note: the same predicate belongs in BigQuery authorized views plus
row-level access policies, so the database enforces it even if this process is
bypassed. That is not possible against a public dataset that we do not own,
which is why it is enforced in-process here.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from retail_agent.safety.pii import ALL_PII_COLUMNS
from retail_agent.safety.scope import Scope

DIALECT = "bigquery"

PROJECT = "bigquery-public-data"
DATASET = "thelook_ecommerce"

ALLOWED_TABLES: frozenset[str] = frozenset(
    {"orders", "order_items", "products", "users"}
)

# Statement types that must never appear, even nested. Top-level SELECT-only
# already blocks these in valid BigQuery, but scanning for them means a parser
# quirk cannot smuggle one through.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
)

_ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.Subquery,
)


class SqlGuardError(ValueError):
    """A query was rejected.

    `reason` is a stable machine code for metrics and for the self-correction
    loop; `message` is what the model is shown so it can fix its query.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a successful guard pass."""

    sql: str
    tables: frozenset[str]
    rewritten: bool


def _qualified(table: str) -> str:
    return f"`{PROJECT}.{DATASET}.{table}`"


def _literal(value: str) -> str:
    """Render a string as a SQL literal with dialect-correct escaping."""
    return exp.Literal.string(value).sql(dialect=DIALECT)


def _scope_predicate(scope: Scope, alias: str) -> str:
    """Build the products-table predicate for a scope, e.g. `p.department IN ('Women')`."""
    clauses: list[str] = []
    for dimension, column in (
        ("departments", "department"),
        ("categories", "category"),
        ("brands", "brand"),
    ):
        values = getattr(scope, dimension)
        if values:
            rendered = ", ".join(_literal(v) for v in sorted(values))
            clauses.append(f"{alias}.{column} IN ({rendered})")
    return " AND ".join(clauses) if clauses else "TRUE"


def _scoped_subquery_sql(table: str, scope: Scope) -> str:
    """SQL for one table restricted to the caller's products.

    Scoping is defined on products and propagated outward: an order is visible
    if it contains at least one in-scope item, and a customer is visible if they
    bought at least one in-scope item. EXISTS rather than JOIN, so that scoping
    never duplicates order or customer rows.
    """
    predicate = _scope_predicate(scope, "scope_p")
    products = _qualified("products")
    order_items = _qualified("order_items")

    if table == "products":
        return f"SELECT scope_p.* FROM {products} AS scope_p WHERE {predicate}"

    if table == "order_items":
        return (
            f"SELECT scope_oi.* FROM {order_items} AS scope_oi "
            f"JOIN {products} AS scope_p ON scope_p.id = scope_oi.product_id "
            f"WHERE {predicate}"
        )

    if table == "orders":
        return (
            f"SELECT scope_o.* FROM {_qualified('orders')} AS scope_o "
            f"WHERE EXISTS (SELECT 1 FROM {order_items} AS scope_oi "
            f"JOIN {products} AS scope_p ON scope_p.id = scope_oi.product_id "
            f"WHERE scope_oi.order_id = scope_o.order_id AND {predicate})"
        )

    if table == "users":
        return (
            f"SELECT scope_u.* FROM {_qualified('users')} AS scope_u "
            f"WHERE EXISTS (SELECT 1 FROM {order_items} AS scope_oi "
            f"JOIN {products} AS scope_p ON scope_p.id = scope_oi.product_id "
            f"WHERE scope_oi.user_id = scope_u.id AND {predicate})"
        )

    raise SqlGuardError("unknown_table", f"table {table!r} is not analysable")


def _parse(sql: str) -> exp.Expression:
    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except sqlglot.ParseError as err:
        raise SqlGuardError("syntax_error", f"the query does not parse: {err}") from err

    statements = [s for s in statements if s is not None]
    if not statements:
        raise SqlGuardError("empty_query", "no SQL statement was provided")
    if len(statements) > 1:
        raise SqlGuardError(
            "multiple_statements",
            "only a single SELECT statement may be run; found "
            f"{len(statements)} statements",
        )
    return statements[0]


def _check_read_only(tree: exp.Expression) -> None:
    if not isinstance(tree, _ALLOWED_ROOTS):
        raise SqlGuardError(
            "not_read_only",
            f"only read-only SELECT queries are allowed, got "
            f"{type(tree).__name__.upper()}",
        )
    for node_type in _FORBIDDEN_NODES:
        if tree.find(node_type):
            raise SqlGuardError(
                "not_read_only",
                f"{node_type.__name__.upper()} is not permitted; the database "
                "connection is read-only",
            )


def _cte_names(tree: exp.Expression) -> frozenset[str]:
    """Names introduced by WITH clauses.

    sqlglot parses a reference to a CTE as a Table node, so without this the
    allowlist would reject the model's own `WITH monthly AS (...)` scaffolding.
    A CTE's definition still gets scoped — it is the reference that is skipped.
    """
    return frozenset(cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE))


def _check_tables(tree: exp.Expression, cte_names: frozenset[str]) -> frozenset[str]:
    """Validate every table reference and return the set of real tables used."""
    for name in sorted(cte_names):
        # A CTE named after a real table makes "is this scoped?" ambiguous to a
        # reader. Cheaper to forbid the shadowing than to reason about it.
        if name in ALLOWED_TABLES:
            raise SqlGuardError(
                "cte_shadows_table",
                f"a WITH clause may not reuse the table name {name!r}; "
                "give the intermediate result a distinct name",
            )

    seen: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        if name not in ALLOWED_TABLES:
            raise SqlGuardError(
                "table_not_allowed",
                f"table {name!r} is not available; this assistant can query only "
                f"{', '.join(sorted(ALLOWED_TABLES))}",
            )
        # Reject a qualified name pointing somewhere other than the public dataset,
        # which would otherwise be a route out of the sandbox.
        if table.db and table.db.lower() != DATASET:
            raise SqlGuardError(
                "table_not_allowed",
                f"dataset {table.db!r} is not available; only {DATASET} is",
            )
        if table.catalog and table.catalog.lower() != PROJECT:
            raise SqlGuardError(
                "table_not_allowed",
                f"project {table.catalog!r} is not available; only {PROJECT} is",
            )
        seen.add(name)
    if not seen:
        raise SqlGuardError(
            "no_tables",
            "the query reads no known table; query one of "
            f"{', '.join(sorted(ALLOWED_TABLES))}",
        )
    return frozenset(seen)


def _has_projection_star(tree: exp.Expression) -> bool:
    """True if any SELECT projects `*` or `alias.*`.

    Deliberately ignores stars nested inside functions: COUNT(*) exposes no
    columns, and treating it as a star made every aggregate over users fail.
    """
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            if isinstance(projection, exp.Star):
                return True
            if isinstance(projection, exp.Column) and isinstance(
                projection.this, exp.Star
            ):
                return True
    return False


def _check_pii(tree: exp.Expression, tables: frozenset[str]) -> None:
    for column in tree.find_all(exp.Column):
        if column.name.lower() in ALL_PII_COLUMNS:
            raise SqlGuardError(
                "pii_column",
                f"column {column.name!r} contains personal data and cannot be "
                "queried, displayed or filtered on. Use aggregates, or "
                "pseudonymous customer ids, instead.",
            )
    # A star over users would expand into the PII columns above.
    if "users" in tables and _has_projection_star(tree):
        raise SqlGuardError(
            "star_over_users",
            "SELECT * is not allowed on the users table because it would expose "
            "personal columns. List the non-personal columns explicitly.",
        )


def _qualify_tables(tree: exp.Expression, cte_names: frozenset[str]) -> None:
    """Expand bare table names to fully-qualified ones, in place.

    The model routinely writes `FROM orders`, which BigQuery rejects with
    "Table must be qualified with a dataset". For a scoped user the rewrite
    below happens to fix that as a side effect, since the injected subqueries
    are fully qualified — which meant unrestricted users were MORE likely to hit
    the error than restricted ones. Qualifying here removes that asymmetry and
    saves a self-correction round trip for everyone.
    """
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names or name not in ALLOWED_TABLES:
            continue
        if table.db:
            continue  # already qualified
        alias = table.args.get("alias")
        qualified = exp.to_table(f"{PROJECT}.{DATASET}.{name}", dialect=DIALECT)
        if alias is not None:
            qualified.set("alias", alias)
        table.replace(qualified)


def _rewrite_for_scope(
    tree: exp.Expression, scope: Scope, cte_names: frozenset[str]
) -> bool:
    """Replace each table reference with a scope-filtered subquery. Returns True if rewritten."""
    if scope.unrestricted:
        return False

    # Materialise the node list first: the replacements themselves contain table
    # references, and re-visiting those would recurse forever.
    original_tables = [
        t for t in tree.find_all(exp.Table) if t.name.lower() not in cte_names
    ]
    if not original_tables:
        return False

    for table in original_tables:
        name = table.name.lower()
        alias = table.alias or name
        inner = sqlglot.parse_one(_scoped_subquery_sql(name, scope), dialect=DIALECT)
        table.replace(
            exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias)))
        )
    return True


def validate_and_rewrite(sql: str, scope: Scope) -> GuardResult:
    """Validate a model-written query and scope it to the caller.

    Raises SqlGuardError with a model-readable message on any violation; the
    agent feeds that message back for self-correction (Requirement 5).
    """
    tree = _parse(sql)
    _check_read_only(tree)
    cte_names = _cte_names(tree)
    tables = _check_tables(tree, cte_names)
    _check_pii(tree, tables)
    _qualify_tables(tree, cte_names)
    rewritten = _rewrite_for_scope(tree, scope, cte_names)
    return GuardResult(
        sql=tree.sql(dialect=DIALECT, pretty=True),
        tables=tables,
        rewritten=rewritten,
    )
