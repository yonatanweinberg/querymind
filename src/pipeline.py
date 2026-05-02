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
from time import perf_counter

import pandas as pd
from sqlalchemy import text

from dataclasses import dataclass, field

from src.database.connection import get_engine
from src.rag.retriever import retrieve_context
from src.llm.prompts import build_messages
from src.llm.provider import call_llm, LLMError
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
# Usage tracking
# ---------------------------------------------------------------------------
@dataclass
class StageTimings:
    """Per-stage latency for a pipeline run, in milliseconds.

    All timings measured with time.perf_counter() - high-resolution
    monotonic clock, right tool for measuring elapsed time of short operations.

    Stages that didn't run (e.g. narration on a CANNOT_ANSWER path) are left
    at 0.0. A reader can spot non-zero values to see what executed.

    Attributes:
        classify_ms: Tier 2 LLM classifcation call. 0 if the question
            was classified by the fast heuristic pattern-matching path or
            didn't need claassification.
        retrieval_ms: ChromaDB query for relevant context (cached after
            first call - subsequent calls are dominated by the actual
            similarity search).
        sql_generation_ms: Main LLM call to produce SQL. Typically the
            single largest contributor on the DATA path.
        validation_ms: Combined time for SQL parsing, access control,
            and cost estimation - all FAST, AST-based operations.
        execution_ms: Time for the SQLite query itself. Varies with
            query complexity and result size.
        narration_ms: Final LLM call to produce the user-facing summary.
            Skipped on error and CANNOT_ANSWER paths.
    """
    classify_ms: float = 0.0
    retrieval_ms: float = 0.0
    sql_generation_ms: float = 0.0
    validation_ms: float = 0.0
    execution_ms: float = 0.0
    narration_ms: float = 0.0

@dataclass
class LLMUsage:
    """Token counts accumulated across every LLM call in a pipeline run.

    Different paths make different numbers of LLM calls (classify, SQL
    generation, narrate, conversational response). These totals add up
    every call's tokens, so a single number gives a sense of cost.

    Attributes:
        input_tokens: Total prompt tokens across all LLM calls.
        output_tokens: Total response tokens across all LLM calls.
        call_count: Number of LLM calls made. Useful for sanity-checking
            the path the pipeline actually took (DATA = 3, CANNOT_ANSWER
            = 2, etc.).
    """
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0

    def add(self, llm_response) -> None:
        # Accumulate 1 LLM call's tokens into the running total
        self.input_tokens += llm_response.input_tokens
        self.output_tokens += llm_response.output_tokens
        self.call_count += 1

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete result from the query pipeline.
    
    Carries everything the UI layer needs to render a response:
    the question, generated SQL, results, and any warnings/errors.

    Attributes:
        question: The original natural-language question.
        success: Whether the full pipeline ran to completion without error.
            True even when the SQL returned 0 rows - see is_empty
            for that distinction.
        is_empty: True when the SQL executed successfully, but returned
            0 rows. Always False unless success is True.
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
        stage_timings: Per-stage latency breakdown (ms) for diagnostics
            and evaluation phase analysis.
        llm_usage: Total token counts and call count across all LLM
            calls in a specific pipeline run.
    """
    question: str
    success: bool
    is_empty: bool = False
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
    stage_timings: StageTimings = field(default_factory=StageTimings)
    llm_usage: LLMUsage = field(default_factory=LLMUsage)


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
        PipelineResult with all information needed to generate the response,
        including per-stage timings (stage_timings) and total token usage
        (llm_usage).
    """
    start_time = time.time()
    # Initialize with failure state - overwritten on success
    result = PipelineResult(question=question, success=False)

    # --- Step 0: Classify the question ---
    t0 = perf_counter()
    question_type = classify_question(question)
    result.stage_timings.classify_ms = (perf_counter() - t0) * 1000
    result.question_type = question_type

    # --- CONVERSATIONAL short-circut ---
    # No SQL needed - generate a direct response and return.
    if question_type == QuestionType.CONVERSATIONAL:
        # Note: narration_ms reuses this slot for the conversational LLM
        # call - only LLM call on this path, so field's name is not the
        # most indicative - meaning is still clear in context.
        t0 = perf_counter()
        result.conversational_response = (
            generate_conversational_response(question)
        )
        result.stage_timings.narration_ms = (perf_counter() - t0) * 1000
        result.success = True
        result.execution_time_seconds = time.time() - start_time
        logger.info(
            f"Conversational response generated in "
            f"{result.execution_time_seconds:.2f}s"
        )
        return result

    # --- Step 1: Input validation ---
    t0 = perf_counter()
    try:
        retrieval = retrieve_context(question)
        rag_context = retrieval.formatted_prompt
    except Exception as e:
        result.error = f"Context retrieval failed: {e}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result
    result.stage_timings.retrieval_ms = (perf_counter() - t0) * 1000
    
    # --- Step 2: Build prompt ---
    system_prompt, messages = build_messages(question, rag_context)

    # --- Step 3: Call LLM ---
    t0 = perf_counter()
    try:
        llm_response = call_llm(system_prompt, messages)
        raw_output = llm_response.text
        result.raw_llm_output = raw_output
        result.llm_usage.add(llm_response)
    except LLMError as e:
        result.error = f"LLM call failed: {e}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result
    result.stage_timings.sql_generation_ms = (perf_counter() - t0) * 1000

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
    # Both step 6 & 7 (access control) bundled into validation_ms - fast AST operations
    t0 = perf_counter()
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
        result.stage_timings.validation_ms = (perf_counter() - t0) * 1000
        result.error = f"Access control violation: {access_result.error}"
        result.narration = narrate_error(question, result.error)
        result.execution_time_seconds = time.time() - start_time
        return result
    
    # --- Step 8: Estimate cost ---
    if engine is None:
        engine = get_engine(readonly=True)

    cost_result = estimate_query_cost(safe_sql, engine)
    result.cost_warnings = cost_result.warnings
    result.stage_timings.validation_ms = (perf_counter() - t0) * 1000

    if cost_result.warnings:
        logger.warning(f"Cost warnings: {cost_result.warnings}")

    # --- Step 9: Execute query ---
    t0 = perf_counter()
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
    result.stage_timings.execution_ms = (perf_counter() - t0) * 1000
    
    # --- Step 10: Narrate results ---
    t0 = perf_counter()
    if len(df) == 0:
        # Query succeeded but returned no data
        result.is_empty = True
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
    result.stage_timings.narration_ms = (perf_counter() - t0) * 1000

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
        st = r.stage_timings
        print(
            f"  Stages (ms): classify={st.classify_ms:.0f} "
            f"retrieval={st.retrieval_ms:.0f} "
            f"sql_gen={st.sql_generation_ms:.0f} "
            f"validation={st.validation_ms:.0f} "
            f"execution={st.execution_ms:.0f} "
            f"narration={st.narration_ms:.0f}"
        )
        print(
            f"  Tokens (SQL gen only): "
            f"{r.llm_usage.input_tokens}in/{r.llm_usage.output_tokens}out"
        )
