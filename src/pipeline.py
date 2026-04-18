"""
Pipeline orchestrator - ties all QueryMind components together.

This is the single entry point for the full question-to-answer flow.
The Streamlit app (or any other alternative caller) calls run_query() with a
natural-language question and gets back a PipelineResult - containing the
generated SQL, query results, alongside any warnings or errors that may arise.

Flow:
    0. Classify question (DATA / ADVISORY / CONVERSATIONAL)
    1. Retrieve context (RAG)
    2. Construct LLM prompt
    3. Call LLM to generate SQL
    4. Clean LLM output (stripping markdown leftovers, etc.)
    5. Check for CANNOT_ANSWER response
    6. Validate SQL (AST parsing, statement type, LIMIT)
    7. Check column-level access control
    8. Estimate query cost
    9. Execute query
    10. Narrate results (or errors)
    11. Package results

Usage:
    from src.pipeline import run_query

    result = run_query("What was the total revenue in 2017?")
    if result.success:
        print(result.narration)
        print(result.dataframe)
    else:
        print(result.narration)  # Plain-language error explanation
"""

import logging
import re
import time

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine
from src.rag.retriever import retrieve_context
from src.llm.prompts import build_messages
from src.llm.provider import generate_sql, LLMError
from src.safety.sql_validator import validate_sql
from src.safety.access_control import check_access_control
from src.safety.cost_estimator import estimate_query_cost
from src.llm.response_generator import (
    classify_question,
    generate_conversational_response,
    narrate_result,
    narrate_error,
    QuestionType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    """Complete result from the query pipeline.
    
    Carries everything the UI layer needs to render a response:
    the question, generated SQL, results, and any warnings/errors.

    Attributes:
        question: The original natural-language question.
        success: Whether the full pipeline completed successfully.
        sql: The generated (AND validated) SQL query. Empty if failed
        dataframe: Query results as a DataFrame. None if failed.
        error: Human-readable, specific error message if pipeline failed.
        cannot_answer_reason: In case the LLM determines the question
            cannot be answered, with the available schema.
        cost_warnings: Advisory warnings from the cost estimator.
        execution_time_seconds: TOtal pipeline execution time.
        raw_llm_output: Unmodified LLM response (for debugging purposes).
        question_type: Classification of question intent.
        narration: Natural-language summary of results or errors.
        conversational_response: Direct response for non-data
            questions (set ony when question_type is CONVERSATIONAL).
    """
    question: str
    success: bool
    sql: str = ""
    dataframe: pd.DataFrame | None = None
    error: str | None = None
    cannot_answer_reason: str | None = None
    cost_warnings: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    raw_llm_output: str = ""
    question_type: QuestionType = QuestionType.DATA
    narration: str = ""
    conversational_response: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_llm_output(raw: str) -> str:
    """Clean the raw LLM output to extract pure SQL query.

    LLMs tend to wrap SQL in markdown code 'fences', even when told
    specifically not to. This function's purpose is to clean - i.e.
    strip those wrappers so the validator receives clean, ready-to-run SQL.
    
    Args:
        raw: The raw text response from the LLM.

    Returns:
        Cleaned SQL String.
    """
    cleaned = raw.strip()

    # Remove markdown fences - e.g. ```sql... ```
    # Uses re.DOTALL - '.' matches newlines inside the fences
    fence_pattern = r"```(?:sql)?\s*\n?(.*?)\n?\s*```"
    match = re.search(fence_pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    # Remove leading "SQL:" prefix - in case it was outputted by LLM
    if cleaned.upper().startswith("SQL:"):
        cleaned = cleaned[4:].strip()

    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_query(
        question: str,
        engine=None,
) -> PipelineResult:
    """Execute the full question-to-answer pipeline.

    Takes a natural-language question, classifies it, and either generates
    a conversational response (no SQL) or runs the full RAG -> SQL ->
    validation -> execution -> narration pipeline.
    
    Args:
        question: Natural-language question from the user.
        engine: Optional SQLAlchemy engine. If None - create a read-only
            engine from the default database path. Accepts an argument
            purely for testing.

    Returns:
        PipelineResult with all information needed to generate the response.
    """
    start_time = time.time()
    # Initialize with failure state - overwritten on success
    result = PipelineResult(question=question, success=False)

    # --- Step 0: Classify the question ---
    question_type = classify_question(question)
    result.question_type = question_type

    # --- CONVERSATIONAL short-circut ---
    # No SQL needed - generate a direct response and return.
    if question_type == QuestionType.CONVERSATIONAL:
        result.conversational_response = (
            generate_conversational_response(question)
        )
        result.success = True
        result.execution_time_seconds = time.time() - start_time
        logger.info(
            f"Conversational response generated in "
            f"{result.execution_time_seconds:.2f}s"
        )
        return result

    # --- Step 1: Input validation ---
    try:
        retrieval = retrieve_context(question)
        rag_context = retrieval.formatted_prompt
    except Exception as e:
        result.error = f"Context retrieval failed: {e}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result
    
    # --- Step 2: Build prompt ---
    system_prompt, messages = build_messages(question, rag_context)

    # --- Step 3: Call LLM ---
    try:
        raw_output = generate_sql(system_prompt, messages)
        result.raw_llm_output = raw_output
    except LLMError as e:
        result.error = f"LLM call failed: {e}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result

    # --- Step 4: Clean LLM output ---
    cleaned_sql = _clean_llm_output(raw_output)

    # --- Step 5: Check for CANNOT_ANSWER ---
    if cleaned_sql.upper().startswith("CANNOT_ANSWER"):
        # Extract the reason, immediately after the colon
        reason = cleaned_sql.split(":", 1)[1].strip() if ":" in cleaned_sql else "Unknown reason"
        result.cannot_answer_reason = reason
        result.narration = reason
        result.success = True   # This is considered a valid outcome, not a failure
        result.execution_time_seconds = time.time() - start_time
        return result
    
    # --- Step 6: Validate SQL ---
    validation = validate_sql(cleaned_sql)
    if not validation.is_valid:
        result.error = f"SQL validation failed: {validation.error}"
        result.sql = cleaned_sql    # Store invalid SQL - for debugging
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result

    # From here on, use the validated (perhaps modified) SQL
    safe_sql = validation.sql
    result.sql = safe_sql

    # --- Step 7: Check access control ---
    access_result = check_access_control(safe_sql)
    if not access_result.is_valid:
        result.error = f"Access control violation: {access_result.error}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result
    
    # --- Step 8: Estimate cost ---
    if engine is None:
        engine = get_engine(readonly=True)

    cost_result = estimate_query_cost(safe_sql, engine)
    result.cost_warnings = cost_result.warnings

    if cost_result.warnings:
        logger.warning(f"Cost warnings: {cost_result.warnings}")

    # --- Step 9: Execute query ---
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(safe_sql), conn)
        result.dataframe = df
        result.success = True
        logger.info(
            f"Query executed successfully: {len(df)} rows, "
            f"{len(df.columns)} columns"
        )
    except Exception as e:
        result.error = f"Query execution failed: {e}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result 
    
    # --- Step 10: Narrate results ---
    if len(df) == 0:
        # Query succeeded but returned no data
        result.narration = narrate_error(
            question, is_empty=True
        )
    else:
        result.narration = narrate_result(
            question=question,
            sql=safe_sql,
            df=df,
            question_type=question_type,
        )

    # --- Wrapping up pipeline ---
    result.execution_time_seconds = time.time() - start_time
    logger.info(
        f"Pipeline completed in {result.execution_time_seconds:.2f}s"
    )

    return result
    

# ---------------------------------------------------------------------------
# CLI Entry Point — for testing the pipeline interactively
# ---------------------------------------------------------------------------
# Run with: python -m src.pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_questions = [
        "What was the total revenue in 2017?",
        "Which product categories have the highest average review scores?",
        "How many unique customers placed orders in each month of 2018?",
        "Hello, what can you do for me?",
        "Which states should we prioritize for growth?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print(f"{'='*70}")

        r = run_query(q)

        print(f"Type: {r.question_type.value}")

        if r.conversational_response:
            print(f"Response: {r.conversational_response}")
        elif r.success and r.dataframe is not None:
            print(f"SQL: {r.sql}")
            print(f"Rows: {len(r.dataframe)}")
            print(r.dataframe.head(5))
            if r.narration:
                print(f"Narration: {r.narration}")
        elif r.cannot_answer_reason:
            print(f"Cannot answer: {r.cannot_answer_reason}")
        else:
            print(f"Error: {r.error}")
            if r.narration:
                print(f"Narration: {r.narration}")

        print(f"Time: {r.execution_time_seconds:.2f}s")
