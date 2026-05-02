"""
QueryMind - Streamlit Application

User-facing interface for the QueryMind Conversational BI Agent.
Provides a chat-style interface where users type natural-language
questions and receive SQL queries, data tables, and auto-generated,
relevant charts in response.

Run via:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# Ensure project root is assigned to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_query, PipelineResult
from src.database.connection import get_engine
from src.visualization.chart_selector import select_chart_type
from src.visualization.chart_builder import build_chart
from src.llm.response_generator import QuestionType


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
# Markdown-rendering helpers
# ---------------------------------------------------------------------------

def _escape_for_markdown(text: str) -> str:
    """Escape characters that Streamlit's markdown renderer treats specially.
    
    Currently handles:
        - '$': Streamlit passes $...$ to its LaTex math renderer. Narration
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


_init_session_state()


# ---------------------------------------------------------------------------
# Example Questions - shown to first-time visitors
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What was the total revenue in 2017?",
    "Show me monthly order trends from 2017 to 2018",
    "Which product categories have the highest average review scores?",
    "What is the breakdown of payment types, amongst our 3 most popular states?",
    "Which states should we prioritize for growth?", # Advisory question
    "What can you tell me about this dataset?",      # Conversational question
]


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
        f"Processed: '{question}' — "
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
            # Escape $ signs to prevent Streamlit interpreting them as LaTex
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

    # --- Success: Pipeline metrics (collapsible) ---
    # Surfaces the per-stage timing and token usage that the pipeline
    # instrumented in run_qeruy. Hidden by default - most users don't
    # want to see this on everu qery, but it's invaluable for
    # diagnosing "why was that one slow?", and for the evaluation stage.
    with st.expander("📊 Pipeline metrics", expanded=False):
        st_timings = result.stage_timings
        usage = result.llm_usage

        # Stage timings rendered as a left-aligned label / right-aligned
        # value pair. Using a markdown code block forces true monospace
        # rendering across browsers (st.text can pick up a near-mono
        # font that breaks alignment on some characters).
        timings_block = (
            f"```\n"
            f"Classification:    {st_timings.classify_ms:>7.0f} ms\n"
            f"Retrieval:         {st_timings.retrieval_ms:>7.0f} ms\n"
            f"SQL generation:    {st_timings.sql_generation_ms:>7.0f} ms\n"
            f"Validation:        {st_timings.validation_ms:>7.0f} ms\n"
            f"Execution:         {st_timings.execution_ms:>7.0f} ms\n"
            f"Narration:         {st_timings.narration_ms:>7.0f} ms\n"
            f"───────────────────────────\n"
            f"Total:             {result.execution_time_seconds * 1000:>7.0f} ms\n"
            f"```"
        )
        st.markdown(timings_block)

        # Token counts. Label this honestly as "SQL generation only"
        # since classification, narration, and conversational tokens are
        # not yet captued (they live inside response_generator.py and
        # would require pushing LLMResponse through those functions).
        st.markdown(
            f"**Tokens (SQL generation):** "
            f"{usage.input_tokens:,} input / {usage.output_tokens:,} output"
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
    meta_col1, meta_col2 = st.columns([3, 1])
    with meta_col1:
        if result.cost_warnings:
            for warning in result.cost_warnings:
                st.warning(f"⚠️ {warning}")
    with meta_col2:
        st.caption(f"⏱️ {result.execution_time_seconds:.2f}s")

# ---------------------------------------------------------------------------
# Sidebar functionality
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 QueryMind 🔍")
    st.caption("Conversational BI Agent")

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
            marketplace (2016–2018).
            """
        )

    if st.session_state.history:
        total_time = sum(
            r.execution_time_seconds for r in st.session_state.history
        )
        total_input_tokens = sum(
            r.llm_usage.input_tokens for r in st.session_state.history
        )
        total_output_tokens = sum(
            r.llm_usage.output_tokens for r in st.session_state.history
        )

        st.subheader("📈 Session stats")
        # st.metric is the cleanest way to render a single big number with
        # a small label. 3 metrics in 3 columns gives a compact dashboard-style
        # block that scales to mobile sidebar widths too (if it comes to it)
        st.metric("Queries", len(st.session_state.history))
        st.metric("Total time", f"{total_time:.1f}s")
        st.metric(
            "Tokens",
            f"{total_input_tokens + total_output_tokens:,}",
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
                "primary"
                if st.session_state.selected_index == i
                else "secondary"
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

user_input = st.chat_input ("Ask a question about the Olist dataset...")

if user_input:
    # Reset to conversation view if we were previously in focused mode
    st.session_state.selected_index = None
    _process_question(user_input)
    st.rerun()
