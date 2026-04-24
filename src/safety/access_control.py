"""
Column-Level Access Control - AST-based restricted column enforcement.

Loads restricted column definitions from config/access_control.yaml and
checks LLM-generated SQL (as a parsed AST) for references to any of them.

This module enforces DATA GOVERNANCE - answers the question "does this
query touch data it shouldn't?" As opposed to the SQL validator - answers
"is this a safe type of query?"

The check covers column references in all SQL clauses: SELECT, WHERE,
JOIN ON, GROUP BY, ORDER BY, and HAVING.

Usage:
    from src.safety.access_control import check_access_control
 
    result = check_access_control("SELECT customer_zip_code_prefix FROM customers")
    if not result.is_valid:
        print(result.error)  # includes the reason from the YAML config
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default path to the access control config file.
# Relative to the project root (where app is ran from)
_DEFAULT_CONFIG_PATH = Path("config/access_control.yaml")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AccessControlResult:
    """Outcome of access control validation.
    
    Attributes:
        is_valid: True if the query references no restricted columns.
        violations: List of human-readable violation descriptions.
                    Empty if valid.
        error: Combined error message summarizing all violations.
                None if valid.
    """
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RestrictedColumn:
    """A single restricted column entry from the YAML config.
    
    frozen=True makes instances immutable and hashable - meaning
    they can be stored in sets for fast lookup
    """
    table: str
    column: str
    reason: str


def load_restricted_columns(
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> list[RestrictedColumn]:
    """Load restricted column definitions from the YAML config file.
 
    Args:
        config_path: Path to access_control.yaml.
 
    Returns:
        List of RestrictedColumn entries.
 
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML structure is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Access control config not found: {config_path}"
        )
 
    with open(config_path) as f:
        config = yaml.safe_load(f)
 
    if not config or "restricted_columns" not in config:
        raise ValueError(
            f"Invalid access control config: missing 'restricted_columns' key "
            f"in {config_path}"
        )
 
    restricted = []
    for entry in config["restricted_columns"]:
        # Validate that each entry has the required fields
        for required_key in ("table", "column", "reason"):
            if required_key not in entry:
                raise ValueError(
                    f"Invalid entry in access control config: missing "
                    f"'{required_key}' in {entry}"
                )
 
        restricted.append(
            RestrictedColumn(
                table=entry["table"].lower(),
                column=entry["column"].lower(),
                reason=entry["reason"],
            )
        )
 
    return restricted


# ---------------------------------------------------------------------------
# AST column reference extraction
# ---------------------------------------------------------------------------

def _build_lookup(
        restricted: list[RestrictedColumn],
) -> dict[str, RestrictedColumn]:
    """Build a fast lookup dict keyed by 'table.column'.
    
    Returns:
        Dict mapping "table.column" strings to their RestrictedColumn entry.
    """
    return {f"{rc.table}.{rc.column}": rc for rc in restricted}


def _build_table_to_restricted_columns(
        restricted: list[RestrictedColumn],
) -> dict[str, list[RestrictedColumn]]:
    """Group restricted column by their parent table.

    Used by the SELECT * check: given a starred table, we need every
    restricted column that belongs to it, not just a single lookup.

    Returns:
        Dict mapping table_name -> list of RestrictedColumn entries.
    """
    table_lookup: dict[str, list[RestrictedColumn]] = {}
    for rc in restricted:
        table_lookup.setdefault(rc.table, []).append(rc)
    return table_lookup


def _build_alias_map(tree: exp.Expression) -> dict [str, str]:
    """Map table aliases AND real names to their canonical real name.

    Example - for 'FROM customers c':
        {'c': 'customers', 'customers': 'customers'}

    Both ELECT t.* and SELECT c.col tests depend on this to resolve an
    alias back to the real table name before looking it up in the
    restricted columns config.
    """
    alias_map: dict[str, str] = {}
    for tbl in tree.find_all(exp.Table):
        real_name = tbl.name.lower()
        # Self-map the real name so unaliased references resolve too
        alias_map[real_name] = real_name
        # If the table has an alias, map it to the real name
        if tbl.alias:
            alias_map[tbl.alias.lower()] = real_name
    return alias_map


def _tables_in_select_scope(select: exp.Select) -> set[str]:
    """Collect tables references in THIS select's FROM/JOIN clauses.

    Scopes to from_/joins subtrees so we don't accidentally attribute
    tables from unrelated Select siblings. We DO recurse into any
    subqueries that appear inside FROM - if a subquery there exposes
    customer data via SELECT *, the outer SELECT * inherits it.
    """
    tables: set[str] = set()
    from_node = select.args.get("from_")
    if from_node is not None:
        for tbl in from_node.find_all(exp.Table):
            tables.add(tbl.name.lower())
    for join in select.args.get("joins") or []:
        for tbl in join.find_all(exp.Table):
            tables.add(tbl.name.lower())
    return tables


def _extract_starred_tables(
        tree: exp.Expression,
        alias_map: dict[str, str],
) -> set[str]:
    """Find real table names exposed by SELECT * or SELECT t.*.

    Inspects ONLY the top-level entries of each Select's expressions.
    This deliberately skips Stars nested inside function calls like
    COUNT(*), which count rows but don't expose column data
    Two shapes are handled:
        - exp.Star at top level           -> SELECT *     (all scope tables)
        - exp.Column with exp.Star inside -> SELECT t.*   (one aliased table)

    Returns:
        Set of real (alias-resolved, lowercased) table names.
    """
    starred_tables: set[str] = set()

    for select in tree.find_all(exp.Select):
        for expr in select.expressions:
            # Case 1: bare * in the select list
            if isinstance(expr, exp.Star):
                starred_tables.update(_tables_in_select_scope(select))
            # Case 2: t.* - a Column node whose inner .this is a Star
            elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
                alias_or_name = (expr.table or "").lower()
                if alias_or_name:
                    # Fall back to the raw name if alias isn't in the map
                    # (conservative: treat unknown qualifiers as real names)
                    starred_tables.add(alias_map.get(alias_or_name, alias_or_name))

    return starred_tables


def _extract_column_references(tree: exp.Expression) -> list[tuple[str, str]]:
    """Extract all (table, column) pairs referenced in the AST.
    
    Walks the entire AST looking for Column nodes. For each column,
    attempts to resolve which table it belongs to.

    Returns:
        List of (table_name, column_name) tuples, both lowercased.
        If a column has no table qualifier, table_name will be an
        empty string.

    NOTE: Does NOT handle SELECT * / SELECT t.* - those are Star nodes,
    not column nodes. See _extract_starred_tables() for that case.
    """
    references = []

    for col_node in tree.find_all(exp.Column):
        # Skip Column nodes that wrap a Star (t.*) - handled separately
        if isinstance(col_node.this, exp.Star):
            continue

        column_name = col_node.name.lower()

        # The table the column belongs to — this is set when the SQL
        # uses explicit table references like "customers.customer_id"
        # or table aliases.
        table_name = col_node.table.lower() if col_node.table else ""
 
        references.append((table_name, column_name))
 
    return references


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_access_control(
    sql: str,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> AccessControlResult:
    """Check whether SQL references any restricted columns.
    
    Parses the SQL into an AST, and runs two independent checks:
    1. Explicit column references (SELECT col, SELECT t.col, WHERE col = ...)
        are looked up against the restricted columns config.
    2. Wildcard references (SELECT *, SELECT t.*) are expanded to the
        full set of source tables, and any restricted column belonging
        to those tables triggers a violation.

    A column is flagged as a violation if:
        - Both table AND column match a restricted entry (explicit match)
        - OR the column name matches and no table qualifier is present
          (conservative match - when we can't determine which table a
           column belongs to, prefer to be cautious)
        - OR the column belongs to a table that is being selected via *

    Args:
        sql: SQL string to check (should already be validated by
             sql_validator.validate_sql).
        config_path: Path to access_control.yaml.
 
    Returns:
        AccessControlResult with is_valid status and any violations.
    """
    # Load the restricted columns config
    restricted = load_restricted_columns(config_path)
    if not restricted:
        # No restriction defined - everything is allowed
        return AccessControlResult(is_valid=True)
    
    lookup = _build_lookup(restricted)
    table_lookup = _build_table_to_restricted_columns(restricted)
    restricted_column_names = {rc.column for rc in restricted}

    try:
        tree = sqlglot.parse_one(sql, dialect = "sqlite")
    except sqlglot.errors.ParseError:
        # If parsing fails, the SQL validator should have caught it already
        # Return valid to avoid double-rejecting.
        return AccessControlResult(is_valid=True)
    
    alias_map = _build_alias_map(tree)

    violations: list[str] = []

    # --- Check 1: explicit column references ---
    for table_name, column_name in _extract_column_references(tree):
        # Resolve alias to real table name. Empty string (unqualified) or
        # an unknown alias both yield None, which routes to the conservative
        # column-name-only match in the else branch.
        resolved_table = alias_map.get(table_name) if table_name else None

        if resolved_table:
            # Qualified and resolvable - strict table.column lookup
            key = f"{resolved_table}.{column_name}"
            if key in lookup:
                rc = lookup[key]
                violations.append(
                    f"Access denied for column '{rc.table}.{rc.column}': "
                    f"{rc.reason}"
                )
        else:
            # Unqualified or unresolvable - conservative name-only match
            if column_name in restricted_column_names:
                matching = [rc for rc in restricted if rc.column == column_name]
                for rc in matching:
                    violations.append(
                        f"Access denied for column '{rc.column}' "
                        f"(potentially from '{rc.table}'): {rc.reason}"
                    )

    # --- Check 2: SELECT * / SELECT t.* ---
    starred_tables = _extract_starred_tables(tree, alias_map)
    for table in starred_tables:
        for rc in table_lookup.get(table, []):
            violations.append(
                f"Access denied: SELECT * on '{table}' would expose "
                f"restricted column '{rc.column}' — {rc.reason}"
            )

    if violations:
        unique_violations = list(dict.fromkeys(violations))
        return AccessControlResult(
            is_valid=False,
            violations=unique_violations,
            error=" | ".join(unique_violations),
        )

    return AccessControlResult(is_valid=True)
