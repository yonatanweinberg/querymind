"""
Test suite for QueryMind safety pipeline.

Covers three safety modules:
    - sql_validator: AST-based SQL validation (statement types, LIMIT, subqueries)
    - access_control: Column-level restriction enforcement
    - cost_estimator: EXPLAIN QUERY PLAN cost analysis

Execute with: python -m pytest tests/test_safety.py -v
"""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.safety.sql_validator import validate_sql, ValidationResult
from src.safety.access_control import check_access_control
from src.safety.cost_estimator import estimate_query_cost, _table_sizes


# ===========================================================================
# SQL VALIDATOR TESTS
# ===========================================================================


class TestValidSelectQueries:
    # Valid SELECT queries should pass this validation.

    def test_simple_select(self):
        result = validate_sql("SELECT * FROM orders LIMIT 10")
        assert result.is_valid
        assert result.error is None

    def test_select_with_where(self):
        result = validate_sql(
            "SELECT order_id, order_status FROM orders "
            "WHERE order_status = 'delivered' LIMIT 50"
        )
        assert result.is_valid

    def test_select_with_join(self):
        result = validate_sql(
            "SELECT o.order_id, c.customer_city "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "LIMIT 100"
        )
        assert result.is_valid

    def test_select_with_aggregation(self):
        result = validate_sql(
            "SELECT customer_state, COUNT(*) as order_count "
            "FROM customers GROUP BY customer_state LIMIT 50"
        )
        assert result.is_valid

    def test_select_with_subquery(self):
        result = validate_sql(
            "SELECT * FROM orders WHERE customer_id IN "
            "(SELECT customer_id FROM customers WHERE customer_state = 'SP') "
            "LIMIT 100"
        )
        assert result.is_valid

    def test_trailing_semicolon_is_allowed(self):
        """A single SELECT with a trailing semicolon should pass.
        sqlglot produces a None entry for the empty statement after
        the semicolon, which we filter out."""
        result = validate_sql("SELECT 1;")
        assert result.is_valid


class TestDestructiveStatements:
    # Dangerous statement types must be rejected.

    def test_insert_rejected(self):
        result = validate_sql("INSERT INTO orders VALUES (1, 2, 3)")
        assert not result.is_valid
        assert "INSERT" in result.error

    def test_update_rejected(self):
        result = validate_sql(
            "UPDATE orders SET order_status = 'canceled' WHERE order_id = 1"
        )
        assert not result.is_valid
        assert "UPDATE" in result.error

    def test_delete_rejected(self):
        result = validate_sql("DELETE FROM orders WHERE order_id = 1")
        assert not result.is_valid
        assert "DELETE" in result.error
 
    def test_drop_table_rejected(self):
        result = validate_sql("DROP TABLE orders")
        assert not result.is_valid
        assert "DROP" in result.error
 
    def test_create_table_rejected(self):
        result = validate_sql("CREATE TABLE hacked (id INT)")
        assert not result.is_valid
        assert "CREATE" in result.error
 
    def test_alter_table_rejected(self):
        result = validate_sql("ALTER TABLE orders ADD COLUMN hacked TEXT")
        assert not result.is_valid
        assert "ALTER" in result.error


class TestInjectionAttempts:
    # SQL injection attempts must be caught.

    def test_semicolon_injection(self):
        """SELECT followed by DROP via semicolon - both parsed as
        separate statements, rejected because count > 1."""
        result = validate_sql("SELECT 1; DROP TABLE orders")
        assert not result.is_valid
        assert "Multiple" in result.error

    def test_drop_in_string_literal_is_safe(self):
        """The word DROP inside a string literal is NOT a DROP statement.
        AST parsing handles this correctly; whereas regex would fail here."""
        result = validate_sql(
            "SELECT * FROM products WHERE product_name = 'Drop Everything Sale' "
            "LIMIT 10"
        )
        assert result.is_valid

    def test_keyword_in_alias_is_safe(self):
        """Using SQL keywords as column aliases is valid SQL."""
        result = validate_sql(
            "SELECT COUNT(*) AS delete_count FROM orders LIMIT 10"
        )
        assert result.is_valid


class TestLimitEnforcement:
    # LIMIT clause must be auto-appended or capped.

    def test_limit_appended_when_missing(self):
        result = validate_sql("SELECT * FROM orders")
        assert result.is_valid
        # The returned SQL should contain a LIMIT clause
        assert "LIMIT" in result.sql.upper()

    def test_reasonable_limit_unchanged(self):
        result = validate_sql("SELECT * FROM orders LIMIT 500")
        assert result.is_valid
        assert "500" in result.sql

    def test_excessive_limit_capped(self):
        """LIMIT 999999 should be capped to predefined MAX_LIMIT (10000)."""
        result = validate_sql("SELECT * FROM orders LIMIT 999999")
        assert result.is_valid
        assert "10000" in result.sql
        assert "999999" not in result.sql


class TestSubqueryValidation:
    # Subqueries must be recursively validated.

    def test_valid_nested_subquery(self):
        result = validate_sql(
            "SELECT * FROM (SELECT order_id FROM orders) AS sub LIMIT 10"
        )
        assert result.is_valid

    def test_deeply_nested_beyond_limit(self):
        """Queries nested beyond MAX_SUBQUERY_DEPTH should be rejected."""
        # Build a 5-level deep nested query (exceeds default depth of 3)
        sql = "SELECT * FROM orders"
        for _ in range(5):
            sql = f"SELECT * FROM ({sql})"
        sql += " LIMIT 10"
        result = validate_sql(sql)
        assert not result.is_valid
        assert "depth" in result.error.lower()


class TestEdgeCases:
    # Edge cases and boundary conditions.

    def test_empty_string(self):
        result = validate_sql("")
        assert not result.is_valid
        assert "Empty" in result.error

    def test_whitespace_only(self):
        result = validate_sql("   \n\t  ")
        assert not result.is_valid
        assert "Empty" in result.error

    def test_invalid_sql_syntax(self):
        result = validate_sql("SELECTTTT everything FROM nowhere")
        assert not result.is_valid


# ===========================================================================
# ACCESS CONTROL TESTS
# ===========================================================================

# Fixture -> creates a temporary access_control.yaml - for testing
@pytest.fixture
def access_config(tmp_path) -> Path:
    # Create a temporary access control config file - purely for testing.
    config_content = """
restricted_columns:
  - table: customers
    column: customer_zip_code_prefix
    reason: "Customer location data is restricted"
  - table: geolocation
    column: geolocation_lat
    reason: "Precise coordinates are restricted"
  - table: geolocation
    column: geolocation_lng
    reason: "Precise coordinates are restricted"
  - table: order_reviews
    column: review_comment_message
    reason: "Review text may contain PII"
"""
    config_file = tmp_path / "access_control.yaml"
    config_file.write_text(config_content)
    return config_file


class TestAccessControlAllowed:
    # Queries referencing non-restricted columns should pass.

    def test_safe_query(self, access_config):
        result = check_access_control(
            "SELECT order_id, order_status FROM orders LIMIT 10",
            config_path=access_config,
        )
        assert result.is_valid
        assert len(result.violations) == 0
 
    def test_safe_aggregation(self, access_config):
        result = check_access_control(
            "SELECT customer_state, COUNT(*) FROM customers "
            "GROUP BY customer_state LIMIT 50",
            config_path=access_config,
        )
        assert result.is_valid


class TestAccessControlBlocked:
    # Queries referencing restricted columns must be rejected.

    def test_restricted_column_in_select(self, access_config):
        result = check_access_control(
            "SELECT customer_zip_code_prefix FROM customers LIMIT 10",
            config_path=access_config
        )
        assert not result.is_valid
        assert len(result.violations) > 0
        assert "restricted" in result.error.lower()

    def test_restricted_column_in_where(self, access_config):
        """Restricted columns in WHERE clauses should also be blocked."""
        result = check_access_control(
            "SELECT order_id FROM customers "
            "WHERE customer_zip_code_prefix = '01310' LIMIT 10",
            config_path=access_config,
        )
        assert not result.is_valid

    def test_restricted_column_without_table_qualifier(self, access_config):
        """Unqualified column references should be conservatively blocked."""
        result = check_access_control(
            "SELECT geolocation_lat FROM geolocation LIMIT 10",
            config_path=access_config,
        )
        assert not result.is_valid

    def test_multiple_restricted_columns(self, access_config):
        """Multiple violations should all be reported."""
        result = check_access_control(
            "SELECT geolocation_lat, geolocation_lng FROM geolocation LIMIT 10",
            config_path=access_config,
        )
        assert not result.is_valid
        assert len(result.violations) >= 2

    def test_restricted_review_text(self, access_config):
        result = check_access_control(
            "SELECT review_comment_message FROM order_reviews LIMIT 10",
            config_path=access_config,
        )
        assert not result.is_valid
        assert "PII" in result.error

    def test_select_star_blocks_restricted_tables(self, access_config):
        """SELECT * expands to every column in the table, including any
        restricted ones. Must be rejected regardless of whether the
        restricted column is written out explicitly."""
        result = check_access_control(
            "SELECT * FROM customers LIMIT 10",
            config_path=access_config
        )
        assert not result.is_valid
        assert len(result.violations) > 0

    def test_select_qualified_star_blocks_restricted(self, access_config):
        """SELECT t.* - same star expansion but scoped to one aliased
        table. The alias must resolve to the real table name before
        the restriction check."""
        result = check_access_control(
            "SELECT c.* FROM customers c LIMIT 10",
            config_path=access_config
        )
        assert not result.is_valid
        assert len(result.violations) > 0

    def test_aliased_restricted_column_blocked(self, access_config):
        """An alias-qualified reference to a restricted column must
        still be blocked. The alias is resolved to the real table
        name before the config lookup."""
        result = check_access_control(
            "SELECT c.customer_zip_code_prefix FROM customers c LIMIT 10",
            config_path=access_config
        )
        assert not result.is_valid
        assert len(result.violations) > 0

    def test_aliased_restricted_in_join(self, access_config):
        """Aliased restricted columns inside a multi-table JOIN are
        the most common real-world pattern. Must be rejected the same
        way as a standalone reference."""
        result = check_access_control(
            "SELECT o.order_id, c.customer_zip_code_prefix "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "LIMIT 10",
            config_path=access_config
        )
        assert not result.is_valid
        assert len(result.violations) > 0
        
class TestAccessControlConfig:
    # Config loading edge cases:

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            check_access_control(
                "SELECT 1",
                config_path=Path("/nonexistent/path.yaml"),
            )

    def test_empty_config(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("restricted_columns: []")
        result = check_access_control(
            "SELECT customer_zip_code_prefix FROM customers",
            config_path=config_file,
        )
        # No restrictions defined, so everything should pass
        assert result.is_valid


# ===========================================================================
# COST ESTIMATOR TESTS
# ===========================================================================

@pytest.fixture
def test_engine():
    """Create an in-memory SQLite database with test data.
    
    Inserts enough rows into 'large_table' to exceed LARGE_TABLE_THRESHOLD
    and keeps 'small_table' under the threshold. Test that only large-
    table scans trigger warnings
    """
    engine = create_engine("sqlite:///:memory:")

    with engine.connect() as conn:
        # Small table - scanning this SHOULD NOT trigger any warnings
        conn.execute(text("CREATE TABLE small_table (id INTEGER, name TEXT)"))
        for i in range(100):
            conn.execute(
                text("INSERT INTO small_table VALUES (:id, :name)"),
                {"id": i, "name": f"item_{i}"},
            )

        # Large table - scanning this SHOULD trigger a warning
        conn.execute(text("CREATE TABLE large_table (id INTEGER, value TEXT)"))
        for i in range(15_000):
            conn.execute(
                text("INSERT INTO large_table VALUES (:id, :val)"),
                {"id": i, "val": f"data_{i}"},
            )
 
        conn.commit()

    # Clear the table size cache so each test starts from scratch
    _table_sizes.clear()

    return engine


class TestCostEstimatorWarnings:
    # Full table scans on large tables should return warnings.

    def test_full_scan_large_table(self, test_engine):
        result = estimate_query_cost(
            "SELECT * FROM large_table", test_engine
        )
        assert result.is_expensive
        assert len(result.warnings) > 0
        assert "large_table" in result.warnings[0]

    def test_full_scan_small_table_no_warning(self, test_engine):
        result = estimate_query_cost(
            "SELECT * FROM small_table", test_engine
        )
        assert not result.is_expensive
        assert len(result.warnings) == 0

    def test_plan_details_populated(self, test_engine):
        result = estimate_query_cost(
            "SELECT * FROM large_table", test_engine
        )
        assert len(result.plan_details) > 0


class TestCostEstimatorEdgeCases:
    # Edge cases for cost estimation.

    def test_invalid_sql_returns_warning(self, test_engine):
        """If EXPLAIN fails, we get a warning but not is_expensive."""
        result = estimate_query_cost(
            "SELECT * FROM nonexistent_table", test_engine
        )
        assert len(result.warnings) > 0
        assert not result.is_expensive

    def test_filtered_query_uses_scan(self, test_engine):
        """A WHERE clause on an unindexed column still scans, but
        SQLite may report it differently. This test verifies the
        module handles it gracefully either way."""
        result = estimate_query_cost(
            "SELECT * FROM large_table WHERE value = 'data_1'",
            test_engine,
        )
        # Whether this triggers a warning depends on SQLite's planner
        # The important thing is that it doesn't crash
        assert isinstance(result.is_expensive, bool)
