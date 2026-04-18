"""
Tests for the response generator module (Phase 4b).

Covers:
    - Heuristic question classification (Tier 1 fast-exit logic)
    - DataFrame formatting for narration prompts

The heuristic classifier is the routing layer that determines how every
question flows through the pipeline. Misclassification means either
wasted LLM calls (routing conversational questions to SQL) or missed
data queries (routing data questions to conversational response).
These tests verify the heuristic boundaries are correct.

LLM-dependent functions (Tier 2 classification, narration, error
narration) are validated through manual integration testing rather
than mocked unit tests, since the value of those tests comes from
real LLM responses, not from verifying that a mock was called.

Run with:
    pytest tests/test_response_generator.py -v
"""

import pandas as pd
import pytest

from src.llm.response_generator import (
    QuestionType,
    _classify_heuristic,
    _format_result_for_narration,
)


# ===================================================================
# Tier 1 Heuristic Classifier — DATA questions
# ===================================================================

class TestClassifyData:
    """Questions that should be classified as DATA by heuristics."""

    def test_aggregation_how_many(self):
        result = _classify_heuristic("How many orders were placed in 2017?")
        assert result == QuestionType.DATA

    def test_aggregation_total(self):
        result = _classify_heuristic("What was the total revenue last year?")
        assert result == QuestionType.DATA

    def test_aggregation_average(self):
        result = _classify_heuristic("What is the average review score?")
        assert result == QuestionType.DATA

    def test_ranking_top_n(self):
        result = _classify_heuristic("Show me the top 10 sellers by revenue")
        assert result == QuestionType.DATA

    def test_ranking_bottom_n(self):
        result = _classify_heuristic("What are the bottom 5 product categories?")
        assert result == QuestionType.DATA

    def test_ranking_highest(self):
        result = _classify_heuristic("Which state has the highest order count?")
        assert result == QuestionType.DATA

    def test_temporal_monthly(self):
        result = _classify_heuristic("Show me monthly revenue from 2017 to 2018")
        assert result == QuestionType.DATA

    def test_temporal_trend(self):
        result = _classify_heuristic("What is the trend in order volume?")
        assert result == QuestionType.DATA

    def test_temporal_growth_rate(self):
        result = _classify_heuristic("What is the growth rate by quarter?")
        assert result == QuestionType.DATA

    def test_explicit_show_me(self):
        result = _classify_heuristic("Show me the breakdown of payment types")
        assert result == QuestionType.DATA

    def test_explicit_list(self):
        result = _classify_heuristic("List all product categories")
        assert result == QuestionType.DATA

    def test_domain_revenue(self):
        result = _classify_heuristic("Revenue by state")
        assert result == QuestionType.DATA

    def test_domain_delivery_time(self):
        result = _classify_heuristic("What is the average delivery time?")
        assert result == QuestionType.DATA

    def test_domain_review_score(self):
        result = _classify_heuristic("Average review score by category")
        assert result == QuestionType.DATA

    def test_domain_orders(self):
        result = _classify_heuristic("How many orders per month?")
        assert result == QuestionType.DATA

    def test_distribution(self):
        result = _classify_heuristic("Show me the distribution of order values")
        assert result == QuestionType.DATA


# ===================================================================
# Tier 1 Heuristic Classifier — CONVERSATIONAL questions
# ===================================================================

class TestClassifyConversational:
    """Questions that should be classified as CONVERSATIONAL by heuristics."""

    def test_greeting_hello(self):
        result = _classify_heuristic("Hello")
        assert result == QuestionType.CONVERSATIONAL

    def test_greeting_hi(self):
        result = _classify_heuristic("hi")
        assert result == QuestionType.CONVERSATIONAL

    def test_greeting_hey(self):
        result = _classify_heuristic("hey there")
        assert result == QuestionType.CONVERSATIONAL

    def test_greeting_good_morning(self):
        result = _classify_heuristic("Good morning")
        assert result == QuestionType.CONVERSATIONAL

    def test_meta_what_can_you_do(self):
        result = _classify_heuristic("What can you do?")
        assert result == QuestionType.CONVERSATIONAL

    def test_meta_who_are_you(self):
        result = _classify_heuristic("Who are you?")
        assert result == QuestionType.CONVERSATIONAL

    def test_meta_help(self):
        result = _classify_heuristic("help")
        assert result == QuestionType.CONVERSATIONAL

    def test_about_olist(self):
        result = _classify_heuristic("What is Olist?")
        assert result == QuestionType.CONVERSATIONAL

    def test_about_olist_tell_me(self):
        result = _classify_heuristic("Tell me about Olist")
        assert result == QuestionType.CONVERSATIONAL

    def test_about_dataset(self):
        result = _classify_heuristic("Describe the database")
        assert result == QuestionType.CONVERSATIONAL

    def test_about_tables(self):
        result = _classify_heuristic("What tables are available?")
        assert result == QuestionType.CONVERSATIONAL

    def test_very_short_input(self):
        result = _classify_heuristic("hi")
        assert result == QuestionType.CONVERSATIONAL

    def test_single_character(self):
        result = _classify_heuristic("?")
        assert result == QuestionType.CONVERSATIONAL

    def test_empty_string(self):
        result = _classify_heuristic("")
        assert result == QuestionType.CONVERSATIONAL


# ===================================================================
# Tier 1 Heuristic Classifier — ADVISORY questions
# ===================================================================

class TestClassifyAdvisory:
    """Questions that should be classified as ADVISORY by heuristics.

    Advisory requires BOTH advisory language AND data signals.
    """

    def test_should_we_with_data_signal(self):
        result = _classify_heuristic(
            "Should we focus on states with the highest revenue?"
        )
        assert result == QuestionType.ADVISORY

    def test_recommend_with_data_signal(self):
        result = _classify_heuristic(
            "Can you recommend which product categories to prioritize?"
        )
        assert result == QuestionType.ADVISORY

    def test_prioritize_with_data_signal(self):
        result = _classify_heuristic(
            "Which states should we prioritize for growth based on orders?"
        )
        assert result == QuestionType.ADVISORY

    def test_downsize_with_data_signal(self):
        result = _classify_heuristic(
            "If we had to downsize, which sellers should we drop?"
        )
        assert result == QuestionType.ADVISORY

    def test_optimize_with_data_signal(self):
        result = _classify_heuristic(
            "How can we optimize delivery time in the worst states?"
        )
        assert result == QuestionType.ADVISORY

    def test_invest_with_data_signal(self):
        result = _classify_heuristic(
            "Where should we invest based on customer reviews?"
        )
        assert result == QuestionType.ADVISORY


# ===================================================================
# Tier 1 Heuristic Classifier — AMBIGUOUS (should return None)
# ===================================================================

class TestClassifyAmbiguous:
    """Questions that should return None (fall through to LLM).

    These are questions where the heuristic isn't confident enough
    to classify, so they should go to the LLM for Tier 2 classification.
    """

    def test_advisory_without_data_signal(self):
        """Advisory language but no clear data signal — ambiguous."""
        result = _classify_heuristic("Should we change our strategy?")
        assert result is None

    def test_vague_question(self):
        """No clear signal in either direction."""
        result = _classify_heuristic("Is the marketplace doing well?")
        assert result is None

    def test_open_ended(self):
        """Exploratory question without concrete data terms."""
        result = _classify_heuristic("What's interesting about this data?")
        assert result is None

    def test_general_business(self):
        """Business question without specific data vocabulary."""
        result = _classify_heuristic("Are we profitable?")
        assert result is None


# ===================================================================
# Tier 1 Heuristic Classifier — Edge cases
# ===================================================================

class TestClassifyEdgeCases:
    """Edge cases and boundary conditions for the classifier."""

    def test_data_signal_mid_sentence(self):
        """Data signal words can appear anywhere, not just at the start."""
        result = _classify_heuristic(
            "In 2017, what was the total number of orders?"
        )
        assert result == QuestionType.DATA

    def test_case_insensitive(self):
        """Classification should be case-insensitive."""
        result = _classify_heuristic("SHOW ME THE TOTAL REVENUE")
        assert result == QuestionType.DATA

    def test_conversational_anchored_to_start(self):
        """Conversational patterns must match at string start."""
        # "hello" at the start = conversational
        result = _classify_heuristic("Hello, how are you?")
        assert result == QuestionType.CONVERSATIONAL

    def test_data_keyword_in_conversational_context(self):
        """'What tables' starts the string — should be conversational,
        even though 'orders' is a data signal."""
        result = _classify_heuristic("What tables contain order data?")
        assert result == QuestionType.CONVERSATIONAL

    def test_whitespace_handling(self):
        """Extra whitespace should not affect classification."""
        result = _classify_heuristic("   How many orders were placed?   ")
        assert result == QuestionType.DATA


# ===================================================================
# DataFrame Formatting for Narration
# ===================================================================

class TestFormatResultForNarration:
    """Tests for _format_result_for_narration helper."""

    def test_small_dataframe_included_fully(self):
        """DataFrames within max_rows should be included in full."""
        df = pd.DataFrame({
            "state": ["SP", "RJ", "MG"],
            "revenue": [1000, 2000, 3000],
        })
        result = _format_result_for_narration(df, max_rows=30)
        assert "SP" in result
        assert "RJ" in result
        assert "MG" in result
        assert "more rows" not in result

    def test_large_dataframe_truncated(self):
        """DataFrames exceeding max_rows should be truncated."""
        df = pd.DataFrame({
            "id": range(100),
            "value": range(100),
        })
        result = _format_result_for_narration(df, max_rows=10)
        assert "90 more rows not shown" in result
        assert "100 total" in result

    def test_exact_max_rows_not_truncated(self):
        """DataFrame with exactly max_rows should not be truncated."""
        df = pd.DataFrame({
            "id": range(30),
            "value": range(30),
        })
        result = _format_result_for_narration(df, max_rows=30)
        assert "more rows" not in result

    def test_single_row(self):
        """Single-row DataFrame (KPI result) should work."""
        df = pd.DataFrame({"total_revenue": [15858341.83]})
        result = _format_result_for_narration(df)
        assert "15858341.83" in result

    def test_empty_dataframe(self):
        """Empty DataFrame should not crash."""
        df = pd.DataFrame(columns=["a", "b"])
        result = _format_result_for_narration(df)
        assert isinstance(result, str)
