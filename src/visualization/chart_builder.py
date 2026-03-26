"""
Chart builder - Plotly figure generation from ChartConfig.

Takes the chart type decision from chart_selector.py and renders
the actual Plotly figure. Each chart type has a dedicated builder
function. Finally build_chart() dispatches to the relevant one.

The generated figures are designed for Streamlit rendering via
st.plotly_chart(), however those are still standard Plotly figures
which work anywhere.

Usage:
    from src.visualization.chart_selector import select_chart_type
    from src.visualization.chart_builder import build_chart

    config = select_chart_type(df)
    fig = build_chart(df, config)
    if fig is not None:
        fig.show()  # or st.plotly_chart(fig) in Streamlit
"""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualization.chart_selector import ChartConfig, ChartType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared, global styling
# ---------------------------------------------------------------------------

# Consistent color palette across all chart types.
# Plotly's default "plotly" palette is fine, however, using a specified one
# ensures visual consistency if theming is later added on.
_COLOR_SEQUENCE = px.colors.qualitative.Set2

_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(size=13),
    margin=dict(l=60, r=30, t=50, b=60),
    height=450,
)


def _apply_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent layout styling to a Plotly figure.
    
    Args:
        fig: The Plotly figure to style.
        title: Chart title.

    Returns:
        The same figure with updated layout.
    """
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _build_kpi(df: pd.DataFrame, config: ChartConfig) -> go.Figure:
    """Build a KPI card showing a single big number.
    
    Uses a Plotly Indicator trace, which renders as a large centered
    number - clean and immediately readable for single-value results.
    """
    value = df[config.value].iloc[0]

    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=value,
            title=dict(text=config.title_hint),
            number=dict(
                font=dict(size=56),
                valueformat=",.2f",
            ),
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
    )

    return fig


def _build_line(df: pd.DataFrame, config: ChartConfig) -> go.Figure:
    """Build a line chart for time-series data."""
    # Sort by the time column to ensure correct line ordering
    df_sorted = df.sort_values(config.x)

    fig = px.line(
        df_sorted,
        x=config.x,
        y=config.y,
        markers=True,
        color_discrete_sequence=_COLOR_SEQUENCE,
    )

    fig.update_xaxes(title_text=config.x.replace("_", " ").title())
    fig.update_yaxes(title_text=config.y.replace("_", " ").title())

    return _apply_layout(fig, config.title_hint)


def _build_bar(df: pd.DataFrame, config: ChartConfig) -> go.Figure:
    """Build a bar chart for categorical comparisons."""
    fig = px.bar(
        df,
        x=config.x,
        y=config.y,
        color_discrete_sequence=_COLOR_SEQUENCE,
    )

    fig.update_xaxes(
        title_text=config.x.replace("_", " ").title(),
        tickangle=-45 if df[config.x].astype(str).str.len().max() > 10 else 0,
    )
    fig.update_yaxes(title_text=config.y.replace("_", " ").title())

    return _apply_layout(fig, config.title_hint)


def _build_pie(df: pd.DataFrame, config: ChartConfig) -> go.Figure:
    """Build a pie/donut chart for proportional breakdowns."""
    fig = px.pie(
        df,
        names=config.label,
        values=config.value,
        hole=0.4,
        color_discrete_sequence=_COLOR_SEQUENCE,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    return _apply_layout(fig, config.title_hint)


def _build_histogram(df: pd.DataFrame, config: ChartConfig) -> go.Figure:
    """Build a histogram for value distributions."""
    fig = px.histogram(
        df,
        x=config.x,
        nbins=30,
        color_discrete_sequence=_COLOR_SEQUENCE,
    )

    fig.update_xaxes(title_text=config.x.replace("_", " ").title())
    fig.update_yaxes(title_text="Count")

    return _apply_layout(fig, config.title_hint)


def _build_scatter(df: pd.DataFrame, config: ChartConfig) -> go.Figure:
    """Build a scatter plot for two numeric variables."""
    fig = px.scatter(
        df,
        x=config.x,
        y=config.y,
        color_discrete_sequence=_COLOR_SEQUENCE,
    )

    fig.update_xaxes(title_text=config.x.replace("_", " ").title())
    fig.update_yaxes(title_text=config.y.replace("_", " ").title())

    return _apply_layout(fig, config.title_hint)


# ---------------------------------------------------------------------------
# Dispatch map
# ---------------------------------------------------------------------------

_BUILDERS = {
    ChartType.KPI: _build_kpi,
    ChartType.LINE: _build_line,
    ChartType.BAR: _build_bar,
    ChartType.PIE: _build_pie,
    ChartType.HISTOGRAM: _build_histogram,
    ChartType.SCATTER: _build_scatter,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chart(
    df: pd.DataFrame,
    config: ChartConfig,
) -> go.Figure | None:
    """Build a Plotly figure based on the chart configuration.
    
    Dispatches to the appropriate builder function based on the
    chart type in the config. Returns None for TABLE_ONLY or if
    the chart type is not supported.

    Args:
        df: The query result DataFrame.
        config: ChartConfig from select_chart_type().

    Returns:
        A Plotly Figure ready for rendering, or None if no chart
        should be displayed.
    """
    if config.chart_type == ChartType.TABLE_ONLY:
        logger.info("Chart type is TABLE_ONLY - no chart generated.")
        return None

    builder = _BUILDERS.get(config.chart_type)
    if builder is None:
        logger.warning(f"No builder for chart type: {config.chart_type}")
        return None
    
    try:
        fig = builder(df, config)
        logger.info(f"Built {config.chart_type.value} chart: '{config.title_hint}'")
        return fig
    except Exception as e:
        logger.error(f"Chart building failed: {e}")
        return None
