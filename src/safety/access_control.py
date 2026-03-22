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


def _extract_column_references(tree: exp.Expression) -> list[tuple[str, str]]:
    """Extract all (table, column) pairs referenced in the AST.
    
    Walks the entire AST looking for Column nodes. For each column,
    attempts to resolve which table it belongs to.

    Returns:
        List of (table_name, column_name) tuples, both lowercased.
        If a column has no table qualifier, table_name will be an
        empty string.
    """
    references = []

    for col_node in tree.find_all(exp.Column):
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
    
    Parses the SQL into an AST, extracts all column references, and
    checks each one against the restricted columns config.

    A column is flagged as a violation if:
        - Both table AND column match a restricted entry (explicit match)
        - OR the column name matches and no table qualifier is present
          (conservative match - when we can't determine which table a
           column belongs to, prefer to be cautious)

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

    # Collect just the restricted column names (without table prefix)
    # for matching unqualified column references
    restricted_column_names = {rc.column for rc in restricted}
 
    # Parse the SQL into an AST
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        # If parsing fails here, the SQL validator should have caught it
        # already. Return valid to avoid double-rejecting.
        return AccessControlResult(is_valid=True)
 
    # Extract all column references from the AST
    references = _extract_column_references(tree)
 
    # Check each reference against the restricted list
    violations = []
 
    for table_name, column_name in references:
        if table_name:
            # Explicit table.column reference — check exact match
            key = f"{table_name}.{column_name}"
            if key in lookup:
                rc = lookup[key]
                violations.append(
                    f"Access denied for column '{rc.table}.{rc.column}': "
                    f"{rc.reason}"
                )
        else:
            # No table qualifier — check if the column name alone matches
            # any restricted column. This is the conservative approach:
            # if someone writes "SELECT customer_zip_code_prefix FROM ..."
            # without specifying the table, we still block it.
            if column_name in restricted_column_names:
                # Find the matching restricted entry for the error message
                matching = [
                    rc for rc in restricted if rc.column == column_name
                ]
                for rc in matching:
                    violations.append(
                        f"Access denied for column '{rc.column}' "
                        f"(potentially from '{rc.table}'): {rc.reason}"
                    )
 
    if violations:
        # Deduplicate in case the same column is referenced multiple times
        unique_violations = list(dict.fromkeys(violations))
        return AccessControlResult(
            is_valid=False,
            violations=unique_violations,
            error=" | ".join(unique_violations),
        )
 
    return AccessControlResult(is_valid=True)
