"""
SQL Safety Validator - AST-based validation using sqlglot.

This is the core module of QueryMind's safety pipeline. Instead of a
fragile, hard-coded Regex pattern-matching approach, here we parse
LLM-generated SQL into an Abstract Syntax Tree (AST) and programmatically
inspects the tree structure. Adapting the same approach used by production
-grade data platforms (e.g. Snowflake, Databricks) for query governance.

Three validation stages:
    1. Statement type whitelisting - only SELECT statements are allowed
    2. LIMIT enforcement - auto-append or cap LIMIT to prevent runaway queries
    3. Subquery validation - recursively verify nested queries

Usage:
    from src.safety.sql_validator import validate_sql

    result = validate_sql("SELECT * FROM orders LIMIT 10")
    if result.is_valid:
        safe_sql = result.sql   # possibly modified (e.g. LIMIT)
    else:
        print(result.error)     # human-readable explanation of failure
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


# ---------------------------------------------------------------------------
# Configuration - consider moving to settings.yaml if needed later
# ---------------------------------------------------------------------------
DEFAULT_LIMIT = 1000    # Applied when the query has no LIMIT clause
MAX_LIMIT = 10000       # Any generated LIMIT above this gets capped
MAX_SUBQUERY_DEPTH = 3  # Reject queries nested deeper than this


# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """
    Outcome of SQL validation.

    Attributes:
        is_valid: Whether the generated SQL passed all safety checks.
        sql: The (potentially modified) SQL string. If LIMIT is missing,
            append it to the returned SQL. Empty string if invalid.
        error: Human-readable explanation if validation failed. None if valid"""
    
    is_valid: bool
    sql: str
    error: str | None = None

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Statement types that are allowed through the validator.
# Everything else - INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
# TRUNCATE, ... is rejected.
_ALLOWED_STATEMENT_TYPES = (exp.Select,)

# Statement types that are explicitly dangerous - used to generate
# targeted error messages rather than a generic "Error".
_DESTRUCTIVE_TYPES = {
    exp.Insert: "INSERT",
    exp.Update: "UPDATE",
    exp.Delete: "DELETE",
    exp.Drop: "DROP",
    exp.Create: "CREATE",
    exp.Alter: "ALTER TABLE",
    exp.Command: "command"      # Handles TRUNCATE, GRANT, etc.
}


def _check_statement_type(statement: exp.Expression) -> str | None:
    """Verify a single AST node is an allowed statement type.

    Returns:
        None if the statement is allowed, or an error message string.
    """
    # Check for explicitly dangerous types first (for better error message generation)
    for dangerous_type, label in _DESTRUCTIVE_TYPES.items():
        if isinstance(statement, dangerous_type):
            return f"{label} statements are not allowed. Only SELECT queries are permitted."
        
    # If it's not in our allowed list, reject with a generic message.
    # Catches anything we didn't explicitly name above
    if not isinstance(statement, _ALLOWED_STATEMENT_TYPES):
        return(
        f"Statement type '{type(statement).__name__}' is not allowed. "
        f"Only SELECT queries are permitted."
        )

    return None


def _enforce_limit(tree: exp.Select) -> exp.Select:
    """Ensure the outermost SELECT has a LIMIT clause.

    - If no LIMIT exists -> append DEFAULT_LIMIT
    - If LIMIT exists but exceeds MAX_LIMIT -> cap it at MAX_LIMIT
    - If LIMIT is within bounds -> stay as-is

    Operates on the AST in-place and returns it.
    """
    limit_clause = tree.args.get("limit")

    if limit_clause is None:
        # No LIMIT at all - append the default
        tree = tree.limit(DEFAULT_LIMIT)
    else:
        # LIMIT exists - extracts the numeric value and check bounds
        limit_expr = limit_clause.expression

        # The limit value is stored as a Literal node; extract its int value
        if isinstance(limit_expr, exp.Literal) and limit_expr.is_int:
            current_limit = int(limit_expr.this)
            if current_limit > MAX_LIMIT:
                # Cap it: replace the literal value in the AST
                limit_expr.set("this", str(MAX_LIMIT))

    return tree


def _check_subqueries(tree: exp.Expression, current_depth: int = 0) -> str | None:
    """Recursively validate all subqueries in the AST.

    Walks the tree looking for nested SELECT statements (subqueries in
    FROM, WHERE, HAVING, etc.) and verifies:
        - Each subquery is a SELECT (not a hidden destructive statement)
        - Nesting depth doesn't exceed MAX_SUBQUERY_DEPTH
    
    Returns:
        None if all subqueries are valid, or an error message string.
    """
    if current_depth > MAX_SUBQUERY_DEPTH:
        return (
            f"Query exceeds maximum subquery depth of {MAX_SUBQUERY_DEPTH}. "
            f"Please simplify the query."
        )
    
    # Find all subquery nodes - these are SELECT statements nested inside
    # the current expression. Use 'find_all' which traverses the tree.
    for node in tree.find_all(exp.Subquery):
        inner = node.this   # The actual statement inside the subquery

        # Verify the inner statement is a SELECT
        type_error = _check_statement_type(inner)
        if type_error:
            return f"Invalid subquery: {type_error}"
        
        # Recurse into the inner statement to check deeper nesting
        depth_error = _check_subqueries(inner, current_depth + 1)
        if depth_error:
            return depth_error
        
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> ValidationResult:
    """Validate LLM-generated SQL through the full safety pipeline.

    Stages:
        1. Parse the SQL string into an AST using sqlglot
        2. Verify only one statement is present (no semicolon injection)
        3. Verify the statement is a SELECT
        4. Recursively validate all subqueries
        5. Enforce LIMIT clause (append or cap)

    Args:
        sql: Raw SQL string from the LLM.

    Returns:
        ValidationResult with is_valid, the (possibly modified) sql,
        and an error message if validation fails.
    """
    # --- Stage 0: Basic input cleaning ---
    sql = sql.strip()
    if not sql:
        return ValidationResult(
            is_valid=False, sql="", error="Empty SQL query received."
        )
    
    # --- Stage 1: Parse into AST ---
    # sqlglot.parse() returns a list of statements (handles semicolons)
    # Use dialect="sqlite" so the parser understands SQLite-specific syntax.
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return ValidationResult(
            is_valid=False,
            sql="",
            error=f"SQL parsing failed: {e}",
        )
    
    # --- Stage 2: Single statement check ---
    # Filter out None entries (sqlglot can return None for empty statements)
    # caused by trailing semicolons, e.g. "SELECT 1;")
    statements = [s for s in statements if s is not None]

    if len(statements) == 0:
        return ValidationResult(
            is_valid=False, sql="", error="No valid SQL statement found."
        )
    
    if len(statements) > 1:
        return ValidationResult(
            is_valid=False,
            sql="",
            error=(
                "Multiple SQL statements detected. Only a single SELECT "
                "query is allowed. Ensure there are no semicolons separating "
                "multiple statements."
            ),
        )
    
    tree = statements[0]

    # --- Stage 3: Statement type check ---
    type_error = _check_statement_type(tree)
    if type_error:
        return ValidationResult(is_valid=False, sql="", error=type_error)
    
    # --- Stage 4: Subquery validation ---
    subquery_error = _check_subqueries(tree)
    if subquery_error:
        return ValidationResult(is_valid=False, sql="", error=subquery_error)

    # --- Stage 5: LIMIT enforcement ---
    tree = _enforce_limit(tree)

    # Generate the final SQL string from the (potentially modified) AST.
    # Using dialect="sqlite" ensures the output is a valid SQLite syntax.
    final_sql = tree.sql(dialect="sqlite")

    return ValidationResult(is_valid=True, sql=final_sql)
