"""
QueryMind - Streamlit Application

User-facing interface for the QueryMind Conversational BI Agent.
Provides a chat-style interface where users type natural-language
questions and receive SQL queries, data tables, and auto-generated,
relevant charts in response.

Run via:
    streamlit run app/streamlit_app.py
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is assigned to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import ensure_ready
from src.database.connection import get_engine
from src.pipeline import PipelineResult, run_query
from src.visualization.chart_builder import build_chart
from src.visualization.chart_selector import select_chart_type

# ---------------------------------------------------------------------------
# Page Configuration - MUST be the first Streamlit command in script
# Any element before this (barring imports) will return an error.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QueryMind - Conversational BI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment bootstrap
# ---------------------------------------------------------------------------
# On a fresh hosted container the data/ directory is empty (both the DB and
# the Chroma store are gitignored). Mirror the Anthropic key from the Streamlit
# secrets into the environment for provider.py, then build the data
# artifacts. Both steps are no-ops locally, where .env supplies the key and
# the README quick start has already built the artifacts.

try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
except Exception:
    # st.secrets can raise (rather than return empty) when no secrets file
    # exists at all, which is the normal local case. The broad catch is
    # deliberate: this block is best-effort, and any failure should fall
    # through to the .env path that load_dotenv() handles in provider.py.
    pass

ensure_ready()

# ---------------------------------------------------------------------------
# Markdown-rendering helpers
# ---------------------------------------------------------------------------


def _escape_for_markdown(text: str) -> str:
    """Escape characters that Streamlit's markdown renderer treats specially.

    Currently handles:
        - '$': Streamlit passes $...$ to its LaTeX math renderer. Narration
            strings that mention dollar amounts (e.g. "revenue of $7.2M") get
            mangled without escaping - returning as LaTeX.

    Applied at every site where LLM-generated text is rendered via
    st.markdown(). Centralized so new escape rules only need one simple edit.
    """
    if not text:
        return text
    return text.replace("$", "\\$")


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_cached_engine():
    """Create and cache the read-only database engine.

    @st.cache_resource ensures this runs once per server session.
    Without it, every Streamlit rerun (which happens on every click,
    every keystroke) would create a new SQLAlchemy engine instance.
    """
    return get_engine(readonly=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------


def _init_session_state():
    """Initialize session state variables on first run.

    st.session_state is a dict that persists across Streamlit reruns.
    Normal Python variables reset on every rerun - session state does not.
    """
    if "history" not in st.session_state:
        # List of PipelineResult objects - one per question asked
        st.session_state.history = []

    if "selected_index" not in st.session_state:
        # When set to an int, main area shows only that result
        # (clicked from sidebar). When None, full conversation is shown
        st.session_state.selected_index = None

    if "advanced_mode" not in st.session_state:
        # Toggles the per-result metrics caption AND the sidebar's
        # session-stats / retrieval-details panels. Default off so the
        # first-impression view stays clean for non-technical visitors;
        # advanced mode is the "I'm here to evaluate the engineering" view.
        st.session_state.advanced_mode = False


_init_session_state()


# ---------------------------------------------------------------------------
# Example Questions - shown to first-time visitors
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What was the total revenue in 2017?",
    "Show me monthly order trends from 2017 to 2018",
    "Which product categories have the highest average review scores?",
    "What is the breakdown of payment types, amongst our 3 most popular states?",
    "Which states should we prioritize for growth?",  # Advisory question
    "What can you tell me about this dataset?",  # Conversational question
]

# Hard per-session query limit for the hosted demo. Dormant while the demo
# URL is shared on request rather than published; see the "Session query
# cap" block just above the chat input (bottom of this file) for the
# activation steps and the design notes.
# SESSION_QUERY_CAP = 10


# ---------------------------------------------------------------------------
# Core: Process a question through the pipeline
# ---------------------------------------------------------------------------


def _process_question(question: str):
    """Run a question through the full pipeline and store the result.

    This is called both from the chat input and from the example buttons.
    After processing, result is appended to session_state.history, so
    that it persists across reruns.
    """
    engine = get_cached_engine()

    with st.spinner("Retrieving context and generating SQL..."):
        result = run_query(question, engine=engine)

    st.session_state.history.append(result)
    # Reset to full conversation view (in case user was in focused view)
    st.session_state.selected_index = None

    logger.info(
        f"Processed: '{question}' - "
        f"{'success' if result.success else 'failed'} "
        f"in {result.execution_time_seconds:.2f}s"
    )


# ---------------------------------------------------------------------------
# Render: Display single PipelineResult
# ---------------------------------------------------------------------------


def _render_result(result: PipelineResult) -> None:
    """Render a PipelineResult inside the current Streamlit container.

    Handles all outcome types:
        - CONVERSATIONAL: plain-text response (no SQL, no chart)
        - CANNOT_ANSWER: informational message
        - Error: error message with plain-language narration
        - Success: narration, SQL expander, chart, data table, metadata
    """
    # --- CONVERSATIONAL response (no SQL involved) ---
    if result.conversational_response:
        st.markdown(_escape_for_markdown(result.conversational_response))
        st.caption(f"⏱️ {result.execution_time_seconds:.2f}s")
        return

    # --- CANNOT_ANSWER ---
    if result.cannot_answer_reason:
        st.info(
            f"🤔 I couldn't answer that with the available data: "
            f"{result.cannot_answer_reason}"
        )
        return

    # --- Empty-result ---
    if result.is_empty:
        st.info(
            "✅ Query ran successfully but returned no results. "
            "Try adjusting filters or expanding the date range."
        )

    # --- Error ---
    if not result.success:
        # Show the narrated error if available, raw error otherwise
        if result.narration:
            # Escape $ signs to prevent Streamlit interpreting them as LaTeX
            st.markdown(_escape_for_markdown(result.narration))
        else:
            st.error(f"Something went wrong: {result.error}")

        # Show the failed SQL for debugging transparency
        if result.sql:
            with st.expander("🔍 Generated SQL (failed)"):
                st.code(result.sql, language="sql")
        return

    # --- Success: Narration (the conversational summary) ---
    if result.narration:
        st.markdown(_escape_for_markdown(result.narration))

    # --- Success: SQL (collapsible) ---
    with st.expander("🔍 View Generated SQL", expanded=False):
        st.code(result.sql, language="sql")

    # --- Success: Retrieved context (advanced mode only) ---
    # Shows the RAG chunks that fed this query's prompt - one row per
    # chunk with its source-type label and L2 distance from the question
    # embedding. Lives behind the SQL expander because the natural reading
    # order is "what we generated -> what we used to generate it".
    if st.session_state.advanced_mode and result.retrieval is not None:
        chunks = result.retrieval.all_chunks
        with st.expander(
            f"🧩 Retrieved context ({len(chunks)} chunks)", expanded=False
        ):
            # L2 distance from ChromaDB - lower means semantically closer
            # to the question. Not a normalized similarity score; explained
            # inline so reviewers can interpret the numbers correctly.
            st.caption(
                "Lower distance = closer match. ChromaDB returns raw L2 "
                "distances, not similarity scores."
            )

            # Sort chunks by distance so the most-relevant appear first.
            # Within source_types ChromaDB already returns sorted, but the
            # all_chunks property concatenates types in insertion order -
            # re-sorting gives a single relevance-ranked view across types.
            sorted_chunks = sorted(chunks, key=lambda c: c.distance)

            for chunk in sorted_chunks:
                # Each chunk gets its own collapsible inner expander.
                # Header carries the headline data (distance + label);
                # body shows the actual chunk text on demand.
                with st.expander(
                    f"`{chunk.distance:.4f}` - {chunk.display_label}",
                    expanded=False,
                ):
                    st.code(chunk.text, language="text")

    # --- Success: Pipeline metrics caption (advanced mode only) ---
    # One-line caption showing total latency, token usage, and estimated
    # cost. Replaces the previous expandable metrics block - dense enough
    # to read at a glance, and the per-stage breakdown lives in the
    # session-stats panel where averaging over many queries is more
    # informative than per-query stage timings.
    if st.session_state.advanced_mode:
        usage = result.llm_usage
        call_word = "call" if usage.call_count == 1 else "calls"
        st.caption(
            f"⏱ {result.execution_time_seconds:.2f}s "
            f"• 🪙 {usage.input_tokens:,} in / {usage.output_tokens:,} out "
            f"({usage.call_count} {call_word}) "
            f"• 💵 ${usage.estimated_cost_usd:.4f}"
        )

    # --- Success: Chart ---
    if result.dataframe is not None and not result.dataframe.empty:
        config = select_chart_type(result.dataframe)
        fig = build_chart(result.dataframe, config)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        # --- Success: Data Table ---
        n_rows = len(result.dataframe)
        # Collapse large tables by default to keep the chat readable
        table_expanded = n_rows <= 20
        with st.expander(
            f"📊 Data Table ({n_rows} row{'s' if n_rows != 1 else ''})",
            expanded=table_expanded,
        ):
            st.dataframe(result.dataframe, use_container_width=True)

    # --- Metadata row: warnings + execution time ---
    # The caption shows here only when advanced mode is OFF - in advanced
    # mode the latency is already in the metrics caption above, no need to
    # render it twice. Cost warnings always show regardless of mode since
    # they're a substantive concern, not "advanced" diagnostics.
    meta_col1, meta_col2 = st.columns([3, 1])
    with meta_col1:
        if result.cost_warnings:
            for warning in result.cost_warnings:
                st.warning(f"⚠️ {warning}")
    with meta_col2:
        if not st.session_state.advanced_mode:
            st.caption(f"⏱️ {result.execution_time_seconds:.2f}s")


# ---------------------------------------------------------------------------
# Sidebar functionality
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 QueryMind 🔍")
    st.caption("Conversational BI Agent")

    st.divider()

    # --- Advanced mode toggle ---
    # Placed near the top so it's easy to find for technical visitors,
    # without being so prominent that casual users feel they need to enable it.
    # The key="advanced_mode" binding writes the toggle's bool straight into
    # st.session_state.advanced_mode on every interaction - downstream code
    # reads from session_state, not from the toggle's return value.
    st.toggle(
        "⚙️ Advanced mode",
        key="advanced_mode",
        help=(
            "Show per-stage latency, token usage, estimated cost, and "
            "retrieval details under each result and in the sidebar."
        ),
    )

    st.divider()

    # About section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            QueryMind translates natural-language questions into SQL
            queries against the **Olist Brazilian E-Commerce** dataset.

            **How it works:**
            1. Your question is matched to relevant schema context (RAG)
            2. An LLM generates a fitting SQL query
            3. The query passes through a safety validation pipeline
            4. Results are displayed with auto-generated charts

            **Dataset:** ~100K orders from a Brazilian e-commerce
            marketplace (2016-2018).
            """
        )

    # --- Session stats (advanced mode only) ---
    # Aggregate metrics across every query in this session, plus a
    # per-stage average so it's clear where typical queries spend their
    # time. Useful for live demos: "see, narration is half the latency -
    # that's the cost of the conversational layer". Hidden by default
    # because non-technical visitors don't need to see it.
    if st.session_state.advanced_mode and st.session_state.history:
        history = st.session_state.history
        n_queries = len(history)

        # Top-line aggregates. Sum per-result fields rather than recomputing
        # from raw token counts - that way is_empty / cannot_answer paths
        # (which still cost real money) get correctly counted.
        total_time = sum(r.execution_time_seconds for r in history)
        total_cost = sum(r.llm_usage.estimated_cost_usd for r in history)
        total_tokens = sum(
            r.llm_usage.input_tokens + r.llm_usage.output_tokens for r in history
        )
        avg_latency = total_time / n_queries

        st.subheader("📈 Session stats")

        # Two-column metric grid. st.metric is Streamlit's idiom for the
        # "big number + small label" pattern - more visually distinctive
        # than plain text, and the label automatically uses muted styling.
        # Two columns keeps each metric readable in a narrow sidebar.
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Queries", n_queries)
            st.metric("Total cost", f"${total_cost:.4f}")
        with col_b:
            st.metric("Avg latency", f"{avg_latency:.2f}s")
            st.metric("Tokens", f"{total_tokens:,}")

        # Per-stage average breakdown. Shows where the typical query spends
        # its time - much more informative than per-query timings, which
        # vary based on query complexity. Iterates the StageTimings fields
        # explicitly so any future field addition is a one-line update.
        avg_classify = sum(r.stage_timings.classify_s for r in history) / n_queries
        avg_retrieval = sum(r.stage_timings.retrieval_s for r in history) / n_queries
        avg_sql_gen = sum(r.stage_timings.sql_generation_s for r in history) / n_queries
        avg_validation = sum(r.stage_timings.validation_s for r in history) / n_queries
        avg_execution = sum(r.stage_timings.execution_s for r in history) / n_queries
        avg_narration = sum(r.stage_timings.narration_s for r in history) / n_queries

        # Markdown code block for monospace alignment. Same pattern as the
        # old per-result metrics expander, just averaged across the session.
        st.caption("**Avg time per stage**")
        st.markdown(
            f"```\n"
            f"Classification:  {avg_classify * 1000:>6.0f} ms\n"
            f"Retrieval:       {avg_retrieval * 1000:>6.0f} ms\n"
            f"SQL generation:  {avg_sql_gen * 1000:>6.0f} ms\n"
            f"Validation:      {avg_validation * 1000:>6.0f} ms\n"
            f"Execution:       {avg_execution * 1000:>6.0f} ms\n"
            f"Narration:       {avg_narration * 1000:>6.0f} ms\n"
            f"```"
        )

    st.divider()

    # Query History
    st.subheader("📋 Query History")

    if not st.session_state.history:
        st.caption("No queries yet. Ask a question to get started!")
    else:
        # "Back to conversation" button when viewing a single result
        if st.session_state.selected_index is not None:
            if st.button("← Back to conversation", use_container_width=True):
                st.session_state.selected_index = None
                st.rerun()

            st.divider()

        # Render each history item as a clickable button
        for i, result in enumerate(st.session_state.history):
            # Truncate long questions for sidebar display
            max_display_len = 45
            display_q = (
                result.question[:max_display_len] + "..."
                if len(result.question) > max_display_len
                else result.question
            )

            # Status icon based on outcome
            if result.cannot_answer_reason:
                icon = "❓"
            elif result.success:
                icon = "✅"
            else:
                icon = "❌"

            # Highlight the currently selected item
            button_type = (
                "primary" if st.session_state.selected_index == i else "secondary"
            )

            if st.button(
                f"{icon} {display_q}",
                key=f"history_{i}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state.selected_index = i
                st.rerun()


# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------

st.title("🔍 QueryMind")
st.markdown("Ask questions about the Olist e-commerce dataset in plain English.")
st.markdown("Data covers Jan. 2017 - Aug. 2018.")
st.markdown("~100k orders. Some personal fields are restricted by design.")
st.divider()

# --- Mode 1: Focused View (single result following sidebar click) ---
if st.session_state.selected_index is not None:
    idx = st.session_state.selected_index

    # Guard against out-of-range index (edge case after clearing history)
    if idx < len(st.session_state.history):
        result = st.session_state.history[idx]
        st.markdown(f"**Q: {result.question}**")
        _render_result(result)
    else:
        st.session_state.selected_index = None
        st.rerun()

# --- Mode 2: Full chat conversation ---
elif st.session_state.history:
    for result in st.session_state.history:
        # User message bubble
        with st.chat_message("user"):
            st.markdown(result.question)

        # Assistant response bubble
        with st.chat_message("assistant"):
            _render_result(result)

# --- Mode 3: Empty state with example questions - "first-time visitors" ---
else:
    st.markdown("### 👋 Welcome! ")
    st.markdown("Try one of these example questions:")
    st.markdown("")

    # Render example questions as full-width clickable buttons
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(
            f"💬  {question}",
            key=f"example_{i}",
            use_container_width=True,
        ):
            _process_question(question)
            st.rerun()


# ---------------------------------------------------------------------------
# Chat Input - always visible at the bottom of the page
# ---------------------------------------------------------------------------

# --- Session query cap (dormant) ---
# Hard per-session limit for a publicly shared demo: once a visitor has run
# SESSION_QUERY_CAP questions, the banner explains the limit and the input
# is disabled. Both live entry points are covered: the example buttons only
# render on an empty history (Mode 3 above), so they can never fire at the
# cap, and the chat input is gated by its disabled= kwarg. This is a cost
# throttle, not a security boundary - the safety pipeline is the boundary,
# and anyone can lift the cap by cloning the repo, which is exactly what
# the banner suggests.
# To activate, uncomment three things: SESSION_QUERY_CAP (defined next to
# EXAMPLE_QUESTIONS above), the banner block below, and the disabled= kwarg
# inside st.chat_input. Uncommenting only part of them fails loudly with a
# NameError rather than half-working.
#
# if len(st.session_state.history) >= SESSION_QUERY_CAP:
#     st.info(
#         f"🔒 This demo session has reached its {SESSION_QUERY_CAP}-query "
#         "limit. To keep exploring, clone the repo and run QueryMind "
#         "locally with your own API key - the README quick start is "
#         "four commands."
#     )

user_input = st.chat_input(
    "Ask a question about the Olist dataset...",
    # disabled=len(st.session_state.history) >= SESSION_QUERY_CAP,
)

if user_input:
    # Reset to conversation view if we were previously in focused mode
    st.session_state.selected_index = None
    _process_question(user_input)
    st.rerun()
