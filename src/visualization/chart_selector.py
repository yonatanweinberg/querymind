"""
Chart type selector - heuristic classification of query results.

Inspects a Pandas DataFrame's shape, column types, and value distributions
and determines the most appropriate chart type. This is the "decision" module.
Actual Plotly rendering happens in chart_builder.py, based on the made decision.

The heuristics map result shapes to chart types based on column types
(numeric, datetime, or categorical).

Usage:
    from src.visualization.chart_selector import select_chart_type, ChartType

    chart_type, config = select_chart_type(df)
    if chart_type == ChartType.BAR:
        print(f"Bar chart: x={config['x']}, y={config['y']}")
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart Types
# ---------------------------------------------------------------------------


class ChartType(Enum):
    # Supported chart types for auto-visualization.
    KPI = "kpi"  # Single big number
    LINE = "line"  # Time series
    BAR = "bar"  # Categorical comparison
    PIE = "pie"  # Proportional breakdown (few categories)
    HISTOGRAM = "histogram"  # Distribution of continuous values
    SCATTER = "scatter"  # Two numeric variables
    TABLE_ONLY = "table_only"  # No chart = show full table


# ---------------------------------------------------------------------------
# Chart configuration container
# ---------------------------------------------------------------------------


@dataclass
class ChartConfig:
    """Configuration for building a chart.

    Contains the chart type and the column assignments - which
    chart_builder.py needs for rendering the visualization

    Attributes:
        chart_type: Which chart to render.
        x: Column name for the x-axis (bar, line, scatter).
        y: Column name for the y-axis (bar, line, scatter).
        value: Column name for the display value (KPI, pie).
        label: Column name for labels (pie).
        title_hint: Suggested chart title based on actual column name.
    """

    chart_type: ChartType
    x: str = ""
    y: str = ""
    value: str = ""
    label: str = ""
    title_hint: str = ""


# ---------------------------------------------------------------------------
# Column classification helpers
# ---------------------------------------------------------------------------

# Patterns that suggest a column contains date/time values
_DATE_NAME_PATTERNS = re.compile(
    r"(date|time|timestamp|month|year|day|quarter|week|period)",
    re.IGNORECASE,
)

# ISO date pattern: YYYY-MM-DD with optional time component
_ISO_DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}(-\d{2})?",
)

# Year-month pattern: YYYY-MM (common in GROUP BY results)
_YEAR_MONTH_PATTERN = re.compile(
    r"^\d{4}-\d{2}$",
)


def _is_datetime_column(series: pd.Series) -> bool:
    """Determine if a column contains date/time values.

    Check both the column name and actual values, since Olist stores dates
    as TEXT strings in ISO format - instead of traditional datetime types.

    Args:
        series: A pandas Series (single DataFrame column).

    Returns:
        True if the column appears to contain date/time data.
    """
    # Check 1: Is it already a datetime dtype?
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Check 2: Does the column name suggest dates?
    name_match = _DATE_NAME_PATTERNS.search(str(series.name))

    # Check 3: Do the values look like dates?
    # Sample up to 10 non-null values to check
    sample = series.dropna().head(10).astype(str)
    if len(sample) == 0:
        return False

    value_match = all(
        _ISO_DATE_PATTERN.match(val) or _YEAR_MONTH_PATTERN.match(val) for val in sample
    )

    # Require either name match + some value evidence, or strong value match
    if value_match:
        return True
    if name_match and len(sample) > 0:
        # Name suggests date - check if at least half the values look like dates
        date_like_count = sum(
            1
            for val in sample
            if _ISO_DATE_PATTERN.match(val) or _YEAR_MONTH_PATTERN.match(val)
        )
        return date_like_count >= len(sample) / 2

    return False


def _classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify each DataFrame column as numeric, datetime, or categorical.

    Args:
        df: The query result DataFrame.

    Returns:
        Dict with keys 'numeric', 'datetime', 'categorical', each
        mapping to a list of column names.
    """
    classification = {
        "numeric": [],
        "datetime": [],
        "categorical": [],
    }

    for col in df.columns:
        if _is_datetime_column(df[col]):
            classification["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            classification["numeric"].append(col)
        else:
            classification["categorical"].append(col)

    logger.debug(f"Column classification: {classification}")
    return classification


# ---------------------------------------------------------------------------
# Prettify column names for chart titles
# ---------------------------------------------------------------------------


def _prettify(column_name: str) -> str:
    """Convert SQL column name to a less technical, human-readable label.

    'total_revenue' ->  "Total Revenue"
    'avg_score'     -> "Avg Score"
    """
    return column_name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Maximum categories before switching from pie to bar chart
_PIE_MAX_CATEGORIES = 6


def select_chart_type(df: pd.DataFrame) -> ChartConfig:
    """Select the best chart type for a query result DataFrame.

    Applies heuristic rules based on the outputted DataFrame's shape and
    column types to determine which visualization communicates the
    data most effectively.

    Args:
        df: The query result DataFrame.

    Returns:
        ChartConfig with the selected chart type and column assignments.
    """
    # --- Guard: empty or None ---
    if df is None or df.empty:
        return ChartConfig(chart_type=ChartType.TABLE_ONLY)

    n_rows, n_cols = df.shape
    cols = _classify_columns(df)

    # --- Rule 1: KPI card (single value) ---
    if n_rows == 1 and n_cols == 1 and len(cols["numeric"]) == 1:
        value_col = cols["numeric"][0]
        return ChartConfig(
            chart_type=ChartType.KPI,
            value=value_col,
            title_hint=_prettify(value_col),
        )

    # --- Rule 2: Line chart (datetime + numeric) ---
    if cols["datetime"] and cols["numeric"]:
        x_col = cols["datetime"][0]
        y_col = cols["numeric"][0]
        return ChartConfig(
            chart_type=ChartType.LINE,
            x=x_col,
            y=y_col,
            title_hint=f"{_prettify(y_col)} Over Time",
        )

    # --- Rule 3 & 4: Categorical + numeric -> pie or bar charts ---
    if cols["categorical"] and cols["numeric"]:
        cat_col = cols["categorical"][0]
        num_col = cols["numeric"][0]
        n_categories = df[cat_col].nunique()

        if n_categories <= _PIE_MAX_CATEGORIES and n_cols == 2:
            return ChartConfig(
                chart_type=ChartType.PIE,
                label=cat_col,
                value=num_col,
                title_hint=f"{_prettify(num_col)} by {_prettify(cat_col)}",
            )
        else:
            return ChartConfig(
                chart_type=ChartType.BAR,
                x=cat_col,
                y=num_col,
                title_hint=f"{_prettify(num_col)} by {_prettify(cat_col)}",
            )

    #  --- Rule 5: Histogram (single numeric column, many rows) ---
    if len(cols["numeric"]) == 1 and n_cols == 1 and n_rows > 5:
        num_col = cols["numeric"][0]
        return ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            x=num_col,
            title_hint=f"Distribution of {_prettify(num_col)}",
        )

    # --- Rule 6: Scatter plot (two numeric columns) ---
    if len(cols["numeric"]) >= 2 and not cols["categorical"]:
        x_col = cols["numeric"][0]
        y_col = cols["numeric"][1]
        return ChartConfig(
            chart_type=ChartType.SCATTER,
            x=x_col,
            y=y_col,
            title_hint=f"{_prettify(x_col)} vs {_prettify(y_col)}",
        )

    # --- Default: table only ---
    return ChartConfig(chart_type=ChartType.TABLE_ONLY)
