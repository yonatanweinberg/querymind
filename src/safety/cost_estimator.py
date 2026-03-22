"""
Query Cost Estimator - pre-execution cost check using EXPLAIN QUERY PLAN.

Before executing a validated query, this module asks SQLite's query planner
HOW it intends to execute it. The planner's response reveals whether the
query will use indexes (fast) or resort to full table scans (potentially
slow and resource-intensive).

This is an advisory check, not a hard block (like validator & access control).
The result includes warnings and a recommendation, but the caller (pipeline.py)
decides whether to proceed, add a protective LIMIT, or just inform the user.

In production system (e.g. Snowflake, BigQuery, etc.) this pattern maps
to query cost budgets and resource governors. SQLite's planner is simpler,
but the architectural pattern is identical.

Usage:
    from src.safety.cost_estimator import estimate_query_cost
    from src.database.connection import get_read_only_engine
 
    engine = get_read_only_engine()
    result = estimate_query_cost("SELECT * FROM orders", engine)
    if result.warnings:
        print(result.warnings)  # ["Full table scan on 'orders'"]
"""

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tables with row counts above this threshold trigger a warning on full scan.
# Small tables (e.g., category_translation with ~71 rows) are fine to scan.
LARGE_TABLE_THRESHOLD = 10_000
 
 
# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CostEstimateResult:
    """Outcome of query cost estimation.
 
    Attributes:
        is_expensive: True if the query plan contains concerning patterns.
        warnings: List of human-readable warnings about the query plan.
        plan_details: Raw EXPLAIN QUERY PLAN output for transparency.
    """
    is_expensive: bool
    warnings: list[str] = field(default_factory=list)
    plan_details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Table size cache
# ---------------------------------------------------------------------------
 
# Cache table row counts so we don't query them repeatedly within a session.
# Reset when the module is reloaded (which is fine for our use case).
_table_sizes: dict[str, int] = {}


def _get_table_size(table_name: str, engine: Engine) -> int:
    """Get the row count for a table, with caching.
 
    Args:
        table_name: Name of the SQLite table.
        engine: SQLAlchemy engine to query against.
 
    Returns:
        Approximate row count. Returns 0 if the table doesn't exist
        or the query fails.
    """
    if table_name in _table_sizes:
        return _table_sizes[table_name]
 
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM [{table_name}]")
            )
            count = result.scalar() or 0
    except Exception:
        count = 0
 
    _table_sizes[table_name] = count
    return count


# ---------------------------------------------------------------------------
# EXPLAIN QUERY PLAN parser
# ---------------------------------------------------------------------------

def _parse_explain_output(
    plan_rows: list[str], engine: Engine
) -> tuple[bool, list[str]]:
    """Analyze EXPLAIN QUERY PLAN output for expensive operations.

    SQLite's EXPLAIN QUERY PLAN returns rows describing execution steps.
    The key pattern we look for:
        - "SCAN TABLE <name>" -> full table scan (no index used)
        - "SEARCH TABLE <name> USING INDEX ..." -> index lookup (efficient)

    A full scan is only concerning on large tables. Scanning a 71-row lookup
    table (category_translation) is harmless; scanning the 1M-row geolocation
    table, without a WHERE clause is not.
    
    Args:
        plan_rows: List of detail strings from EXPLAIN QUERY PLAN.
        engine: SQLAlchemy engine (needed to check table sizes).
 
    Returns:
        Tuple of (is_expensive, warnings).
    """
    warnings = []
    is_expensive = False

    for row in plan_rows:
        # EXPLAIN QUERY PLAN rows look like:
        #   "SCAN TABLE orders"
        #   "SEARCH TABLE orders USING INDEX idx_orders_date ..."
        #   "SCAN TABLE orders USING COVERING INDEX ..."
        
        row_upper = row.upper()

        # Detect full table scans — "SCAN TABLE <name>" or "SCAN <name>"
        # depending on SQLite version. Exclude rows containing "USING"
        # since "SCAN TABLE ... USING COVERING INDEX" is efficient.
        if row_upper.startswith("SCAN") and "USING" not in row_upper:
            parts = row.split()

            # Extract table name — handle both formats:
            #   "SCAN TABLE table_name" → table name at index 2
            #   "SCAN table_name"       → table name at index 1
            try:
                if len(parts) >= 2 and parts[1].upper() == "TABLE":
                    table_name = parts[2].strip("\"'`")
                else:
                    table_name = parts[1].strip("\"'`")
            except IndexError:
                continue

            row_count = _get_table_size(table_name, engine)

            if row_count > LARGE_TABLE_THRESHOLD:
                is_expensive = True
                warnings.append(
                    f"Full table scan on '{table_name}' "
                    f"({row_count:,} rows). Consider adding a WHERE clause "
                    f"to filter results."
                )
 
    return is_expensive, warnings
 

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_query_cost(sql: str, engine: Engine) -> CostEstimateResult:
    """Estimate query cost using SQLite's EXPLAIN QUERY PLAN.

    Runs the query plan without executing the actual query, then
    analyzes the plan for expensive operations (primarily full table
    scans on large tables).

    Args:
        sql: Validated SQL string (should have passed sql_validator
             and access_control checks first).
        engine: SQLAlchemy engine for the target database.

    Returns:
        CostEstimateResult with warnings and expense assessment.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"EXPLAIN QUERY PLAN {sql}"))
            rows = result.fetchall()
    except Exception as e:
        # If EXPLAIN itself fails, return a warning but don't block.
        # The actual execution will catch the real error.
        return CostEstimateResult(
            is_expensive=False,
            warnings=[f"Could not analyze query plan: {e}"],
        )
    
    # Each row from EXPLAIN QUERY PLAN is a tuple:
    # (id, parent, notused, detail)
    # We care about the 'detail' field (index 3).
    plan_details = [row[3] for row in rows if len(row) > 3]

    is_expensive, warnings = _parse_explain_output(plan_details, engine)

    return CostEstimateResult(
        is_expensive=is_expensive,
        warnings=warnings,
        plan_details=plan_details
    )
