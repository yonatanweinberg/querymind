"""
Test suite for chart type selection heuristics.

Covers every branch of select_chart_type:
    - KPI (single value)
    - LINE (datetime + numeric)
    - BAR / PIE (caategorical + numeric, branching on cradinality)
    - HISTOGRAM (single numeric, many rows)
    - SCATTER (two numeric, no categorical)
    - TABLE_ONLY (empty, None, fall-through)

Also covers the helpers _prettify and _is_datetime_column directly,
since they're the building blocks the public function relies on.

Run via: pytest tests/test_chart_selector.py -v
"""

import pandas as pd

from src.visualization.chart_selector import (
    ChartType,
    _is_datetime_column,
    _prettify,
    select_chart_type,
)

# ===========================================================================
# KPI - single numeric value renders as a big-number card
# ===========================================================================


class TestKPIChart:
    def test_single_numeric_value(self):
        df = pd.DataFrame({"total_revenue": [12345.67]})
        config = select_chart_type(df)
        assert config.chart_type == ChartType.KPI
        assert config.value == "total_revenue"

    def test_title_hint_is_prettified(self):
        df = pd.DataFrame({"avg_score": [4.2]})
        config = select_chart_type(df)
        assert config.chart_type == ChartType.KPI
        assert config.title_hint == "Avg Score"


# ===========================================================================
# LINE - datetime + numeric
# ===========================================================================


class TestLineChart:
    def test_iso_date_format(self):
        df = pd.DataFrame(
            {
                "order_purchase_timestamp": [
                    "2017-01-15",
                    "2017-02-15",
                    "2017-03-15",
                    "2017-04-15",
                ],
                "revenue": [1000.0, 1500.0, 1200.0, 1800.0],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.LINE
        assert config.x == "order_purchase_timestamp"
        assert config.y == "revenue"

    def test_year_month_format(self):
        # Common GROUP BY result shape: "2017-01", "2017-02", ...
        df = pd.DataFrame(
            {
                "month": ["2017-01", "2017-02", "2017-03"],
                "order_count": [100, 120, 150],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.LINE


# ===========================================================================
# BAR / PIE - categorical + numeric, with cardinality threshold
# ===========================================================================


class TestBarAndPieCharts:
    def test_pie_with_few_categories(self):
        df = pd.DataFrame(
            {
                "payment_type": ["credit_card", "boleto", "voucher", "debit_card"],
                "count": [100, 50, 20, 10],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.PIE
        assert config.label == "payment_type"
        assert config.value == "count"

    def test_bar_with_many_categories(self):
        # 27 states - well above the pie threshold
        states = [f"S{i:02d}" for i in range(27)]
        df = pd.DataFrame(
            {
                "customer_state": states,
                "order_count": list(range(27)),
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.BAR
        assert config.x == "customer_state"
        assert config.y == "order_count"

    def test_pie_bar_threshold_boundary(self):
        # _PIE_MAX_CATEGORIES = 6: exactly 6 -> pie, 7 -> bar.
        # Locks the threshold so a future change to it surfaces here.
        df_pie = pd.DataFrame(
            {
                "category": [f"c{i}" for i in range(6)],
                "count": list(range(6)),
            }
        )
        df_bar = pd.DataFrame(
            {
                "category": [f"c{i}" for i in range(7)],
                "count": list(range(7)),
            }
        )
        assert select_chart_type(df_pie).chart_type == ChartType.PIE
        assert select_chart_type(df_bar).chart_type == ChartType.BAR

    def test_three_columns_forces_bar(self):
        # PIE requires exactly 2 columns. With 3+ columns a categorical
        # + numeric pair still goes to BAR even if categories are few.
        df = pd.DataFrame(
            {
                "category": ["a", "b", "c"],
                "count": [10, 20, 30],
                "avg_price": [1.0, 2.0, 3.0],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.BAR


# ===========================================================================
# HISTOGRAM - single numeric column, many rows
# ===========================================================================


class TestHistogram:
    def test_many_rows_single_numeric(self):
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.HISTOGRAM
        assert config.x == "price"

    def test_five_rows_falls_through(self):
        # Histogram requires n_rows > 5. With exactly 5 rows nothing matches:
        # not KPI (n_rows != 1), not line/bar/pie (no datetime/categorical),
        # not scatter (only one numeric col) -> TABLE_ONLY.
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]})
        config = select_chart_type(df)
        assert config.chart_type == ChartType.TABLE_ONLY


# ===========================================================================
# SCATTER - two numeric columns, no categorical
# ===========================================================================


class TestScatter:
    def test_two_numeric_no_categorical(self):
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0],
                "freight_value": [2.0, 4.0, 6.0],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.SCATTER
        assert config.x == "price"
        assert config.y == "freight_value"


# ===========================================================================
# TABLE_ONLY - guards and fall-throughs
# ===========================================================================
class TestTableOnly:
    def test_empty_dataframe(self):
        df = pd.DataFrame()
        config = select_chart_type(df)
        assert config.chart_type == ChartType.TABLE_ONLY

    def test_none_input(self):
        config = select_chart_type(None)
        assert config.chart_type == ChartType.TABLE_ONLY

    def test_all_categorical(self):
        # No numeric column anywhere -> nothing to plot
        df = pd.DataFrame(
            {
                "name": ["a", "b", "c"],
                "category": ["x", "y", "z"],
                "tag": ["foo", "bar", "baz"],
            }
        )
        config = select_chart_type(df)
        assert config.chart_type == ChartType.TABLE_ONLY

    def test_single_row_single_col_categorical(self):
        # KPI requires the lone column to be numeric. A categorical-only
        # 1x1 frame should fall through to TABLE_ONLY
        df = pd.DataFrame({"status": ["delivered"]})
        config = select_chart_type(df)
        assert config.chart_type == ChartType.TABLE_ONLY


# ===========================================================================
# Helpers - _prettify and _is_datetime_column
# ===========================================================================


class TestPrettify:
    def test_snake_case_becomes_title_case(self):
        assert _prettify("total_revenue") == "Total Revenue"

    def test_single_word_capitalized(self):
        assert _prettify("price") == "Price"


class TestDatetimeDetection:
    def test_iso_date_strings_detected(self):
        s = pd.Series(
            ["2017-01-15", "2017-02-15", "2017-03-15"],
            name="order_date",
        )
        assert _is_datetime_column(s)

    def test_year_month_strings_detected(self):
        s = pd.Series(["2017-01", "2017-02", "2017-03"], name="month")
        assert _is_datetime_column(s)

    def test_plain_strings_not_detected(self):
        s = pd.Series(["apple", "banana", "cherry"], name="fruit")
        assert not _is_datetime_column(s)
