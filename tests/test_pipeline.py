"""
Test suite for the pipeline orchestrator.

Two layers:
    1. _clean_llm_output - pure function, simple unit tests.
    2. run_query - integration tests using monkeypatched call_llm and
       a tiny in-memory SQLite database.

The LLM is mocked because real calls cost money, are slow, and are
non-deterministic. The database is real (SQLite in-memory) because
SQL execution is part of what we're testing - mocking it would mean
testing nothing meaningful.

Run: pytest tests/test_pipeline.py -v
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.pipeline import _clean_llm_output, run_query, PipelineResult
from src.llm.provider import LLMError, LLMResponse
from src.llm.response_generator import QuestionType


# ===========================================================================
# Pure function tests - _clean_llm_output
# ===========================================================================

class TestCleanLLMOutput:
    """The cleaner strips markdown fences and 'SQL:' prefixes that the
    LLM sometimes adds despite being told not to."""

    def test_plain_sql_unchanged(self):
        sql = "SELECT * FROM orders LIMIT 10"
        assert _clean_llm_output(sql) == sql

    def test_strips_sql_fence(self):
        raw = "```sql\nSELECT * FROM orders LIMIT 10\n```"
        assert _clean_llm_output(raw) == "SELECT * FROM orders LIMIT 10"

    def test_strips_bare_fence(self):
        raw = "```\nSELECT * FROM orders LIMIT 10\n```"
        assert _clean_llm_output(raw) == "SELECT * FROM orders LIMIT 10"

    def test_strips_sql_prefix(self):
        raw = "SQL: SELECT * FROM orders LIMIT 10"
        assert _clean_llm_output(raw) == "SELECT * FROM orders LIMIT 10"

    def test_strips_fence_and_prefix(self):
        # Belt-and-suspenders case: LLM wrapped in fence AND prefixed.
        raw = "```sql\nSQL: SELECT * FROM orders LIMIT 10\n```"
        assert _clean_llm_output(raw) == "SELECT * FROM orders LIMIT 10"

    def test_strips_surrounding_whitespace(self):
        raw = "   \n  SELECT * FROM orders LIMIT 10  \n  "
        assert _clean_llm_output(raw) == "SELECT * FROM orders LIMIT 10"

    def test_preserves_internal_newlines(self):
        # Multi-line SQL inside a fence should keep its line breaks
        # so the validator and EXPLAIN output stay readable.
        raw = "```sql\nSELECT *\nFROM orders\nLIMIT 10\n```"
        assert _clean_llm_output(raw) == "SELECT *\nFROM orders\nLIMIT 10"

    def test_preserves_cannot_answer(self):
        # CANNOT_ANSWER responses must pass through untouched - the
        # pipeline's CANNOT_ANSWER branch parses the prefix.
        raw = "CANNOT_ANSWER: Schema does not include sentiment data"
        assert _clean_llm_output(raw) == raw


# ===========================================================================
# run_query integration tests
# ===========================================================================

# A tiny in-memory database with one table. Matches the shape of the real
# Olist orders table closely enough that the pipeline doesn't notice.
_TEST_SCHEMA = """
CREATE TABLE olist_orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    order_status TEXT,
    order_purchase_timestamp TEXT
);
"""

_TEST_DATA = [
    ("o1", "c1", "delivered", "2017-01-15"),
    ("o2", "c2", "delivered", "2017-02-15"),
    ("o3", "c3", "canceled",  "2017-03-15"),
]


@pytest.fixture
def test_engine():
    """Create a fresh in-memory SQLite database for each test.

    The pipeline expects an engine it can read from; we give it one
    backed by a 3-row table. Each test gets its own engine so changes
    in one test can't leak into another.
    """
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(_TEST_SCHEMA))
        for row in _TEST_DATA:
            conn.execute(
                text(
                    "INSERT INTO olist_orders VALUES "
                    "(:id, :cust, :status, :ts)"
                ),
                {"id": row[0], "cust": row[1], "status": row[2], "ts": row[3]},
            )
    return engine


def _stub_retriever(monkeypatch):
    """Replace retrieve_context with a no-op that returns a real
    (but empty) RetrievalResult.

    Returns a production RetrievalResult rather than a SimpleNamespace
    so any new field added to the data class continues to work without
    test updates. The empty chunk lists let formatted_prompt return
    a usable string and keep result.retrieval.all_chunks introspectable.
    """
    from src.rag.retriever import RetrievalResult
    fake_result = RetrievalResult(question="(stubbed)")
    monkeypatch.setattr(
        "src.pipeline.retrieve_context",
        lambda question: fake_result,
    )


def _stub_classifier(monkeypatch, question_type=QuestionType.DATA):
    """Force classify_question to return a fixed type.

    Tests pin the classification deterministically so we don't accidentally
    test the classifier's accuracy here - that's a separate concern from
    pipeline orchestration.
    """
    monkeypatch.setattr(
        "src.pipeline.classify_question",
        lambda q, usage=None: question_type,
    )


def _stub_narrators(monkeypatch):
    """Replace the narration helpers with deterministic stubs.

    Each one normally makes its own LLM call; we replace them with
    functions that just return identifiable strings, so assertions can
    confirm 'narration was set' without depending on real LLM output.
    """
    monkeypatch.setattr(
        "src.pipeline.narrate_result",
        lambda question, sql, df, question_type, usage=None: "STUBBED_NARRATION",
    )
    monkeypatch.setattr(
        "src.pipeline.narrate_error",
        lambda question, error=None, is_empty=False, usage=None:
            "STUBBED_EMPTY" if is_empty else "STUBBED_ERROR",
    )


class TestRunQueryHappyPath:
    """The full pipeline runs end-to-end against a real (in-memory) DB
    when call_llm returns valid SQL."""

    def test_successful_query(self, monkeypatch, test_engine):
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)
        monkeypatch.setattr(
            "src.pipeline.call_llm",
            lambda system, messages: LLMResponse(
                text="SELECT * FROM olist_orders LIMIT 10",
                input_tokens=0,
                output_tokens=0,
            ),
        )

        result = run_query("how many orders?", engine=test_engine)

        assert result.success is True
        assert result.is_empty is False
        assert result.dataframe is not None
        assert len(result.dataframe) == 3  # All rows from the test fixture
        assert "olist_orders" in result.sql.lower()
        assert result.narration == "STUBBED_NARRATION"
        assert result.error is None

    def test_limit_appended_when_missing(self, monkeypatch, test_engine):
        # The validator should append a LIMIT - confirms the safety
        # pipeline ran and modified the SQL before execution.
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)
        monkeypatch.setattr(
            "src.pipeline.call_llm",
            lambda system, messages: LLMResponse(
                text="SELECT * FROM olist_orders",
                input_tokens=0,
                output_tokens=0,
            ),
        )

        result = run_query("show all orders", engine=test_engine)

        assert result.success is True
        assert "LIMIT" in result.sql.upper()


class TestRunQueryEmptyResults:
    """SQL runs to completion but returns zero rows -> is_empty=True,
    success=True (the two flags compose orthogonally)."""

    def test_empty_result_sets_is_empty(self, monkeypatch, test_engine):
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)
        # Filter to a status that doesn't exist in the test data
        monkeypatch.setattr(
            "src.pipeline.call_llm",
            lambda system, messages: LLMResponse(
                text="SELECT * FROM olist_orders WHERE order_status = 'nonexistent'",
                input_tokens=0,
                output_tokens=0,
            ),
        )

        result = run_query("orders with bogus status", engine=test_engine)

        assert result.success is True
        assert result.is_empty is True
        assert len(result.dataframe) == 0
        assert result.narration == "STUBBED_EMPTY"


class TestRunQueryFailures:
    """The pipeline should fail gracefully and return a result with
    success=False and a populated narration for any error."""

    def test_invalid_sql_rejected(self, monkeypatch, test_engine):
        # An UPDATE statement should be blocked by the SQL validator.
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)
        monkeypatch.setattr(
            "src.pipeline.call_llm",
            lambda system, messages: LLMResponse(
                text="UPDATE olist_orders SET order_status = 'shipped'",
                input_tokens=0,
                output_tokens=0,
            ),
        )

        result = run_query("update everything", engine=test_engine)

        assert result.success is False
        assert result.error is not None
        assert "validation failed" in result.error.lower()
        assert result.narration == "STUBBED_ERROR"

    def test_llm_error_propagates(self, monkeypatch, test_engine):
        # Provider failure surfaces as success=False with an LLM-flavored
        # error message - catches the LLMError handler in run_query.
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)

        def raise_llm_error(system, messages):
            raise LLMError("simulated network failure")

        monkeypatch.setattr("src.pipeline.call_llm", raise_llm_error)

        result = run_query("anything", engine=test_engine)

        assert result.success is False
        assert "LLM call failed" in result.error
        assert result.narration == "STUBBED_ERROR"

    def test_cannot_answer_returns_success_with_reason(
        self, monkeypatch, test_engine
    ):
        # CANNOT_ANSWER is a valid outcome, not a failure: success=True,
        # but cannot_answer_reason is set and the narration carries it.
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)
        monkeypatch.setattr(
            "src.pipeline.call_llm",
            lambda system, messages: LLMResponse(
                text="CANNOT_ANSWER: Schema doesn't include sentiment data",
                input_tokens=0,
                output_tokens=0,
            ),
        )

        result = run_query("how do customers feel?", engine=test_engine)

        assert result.success is True
        assert result.cannot_answer_reason is not None
        assert "sentiment" in result.cannot_answer_reason.lower()
        assert result.dataframe is None  # No SQL was executed


class TestRunQueryConversational:
    """CONVERSATIONAL questions short-circuit before any SQL work -
    no retrieval, no LLM SQL call, no DB hit."""

    def test_conversational_short_circuit(self, monkeypatch, test_engine):
        _stub_classifier(monkeypatch, question_type=QuestionType.CONVERSATIONAL)
        monkeypatch.setattr(
            "src.pipeline.generate_conversational_response",
            lambda question, usage=None: "Hi, I help with the Olist dataset.",
)

        # Deliberately don't stub retriever or call_llm - if the pipeline
        # tries to call them, the test will fail with NameError or
        # network error, exposing a regression in the short-circuit.
        result = run_query("what can you do?", engine=test_engine)

        assert result.success is True
        assert result.conversational_response == (
            "Hi, I help with the Olist dataset."
        )
        assert result.sql == ""
        assert result.dataframe is None

class TestRunQueryRetrieval:
    """The retrieval result should be attached to PipelineResult so
    the UI can surface chunk-level transparency. Stubbing the retriever
    means we're testing 'does pipeline assign the field' - not 'does
    ChromaDB work', which is a separate concern."""

    def test_retrieval_set_on_success(self, monkeypatch, test_engine):
        # The retrieval attribute is populated whenever the pipeline
        # actually called retrieve_context, regardless of result-set size.
        _stub_retriever(monkeypatch)
        _stub_classifier(monkeypatch)
        _stub_narrators(monkeypatch)
        monkeypatch.setattr(
            "src.pipeline.call_llm",
            lambda system, messages: LLMResponse(
                text="SELECT * FROM olist_orders LIMIT 10",
                input_tokens=0,
                output_tokens=0,
            ),
        )

        result = run_query("any data question", engine=test_engine)

        assert result.success is True
        assert result.retrieval is not None
        assert result.retrieval.question == "(stubbed)"

    def test_retrieval_none_for_conversational(
        self, monkeypatch, test_engine
    ):
        # CONVERSATIONAL short-circuits before retrieve_context() runs,
        # so retrieval should remain None. Verifies the short-circuit
        # really skipped the RAG layer.
        _stub_classifier(monkeypatch, question_type=QuestionType.CONVERSATIONAL)
        monkeypatch.setattr(
            "src.pipeline.generate_conversational_response",
            lambda question, usage=None: "Hi there!",
        )

        result = run_query("hello", engine=test_engine)

        assert result.success is True
        assert result.retrieval is None