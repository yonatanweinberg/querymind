"""
Response Generator - Conversational Intelligence for QueryMind

This module adds the "conversational" layer that transforms QueryMind from an SQL-
generation, paired with auto-visualization tool, into a conversational BI agent. Handles:

    1. Question classification: Is this a data-related question (needs SQL),
        an advisory question (needs SQL + deeper analysis), or a conversational
        question (answer directly, no SQL required)?

    2. Result narration: After SQL execution, generate a natural-language summary
        of the findings. Keep it short for DATA questions, provide deeper analysis
        for ADVISORY questions.
    
    3. Error/empty narration: When queries fail or return no rows, provide
        an explanation of what has happened, in plain language.

    4. Conversational response: For non-data questions, generate a helpful
    response about the Olist dataset, system's capabilities, or even
    general greetings.

The question classifier uses a hybrid approach:
    - Tier 1 (fast exit): Pattern matching (heuristic) catches obviously data-
      related or obviously conversational questions, avoiding an LLM call.
    - Tier 2 (LLM fallback): Ambiguous questions are classified by Claude,
      costing ~$0.003 and ~0.5-1s BUT handling edge cases accurately.

      
Usage:
    from src.llm.response_generator import (
        classify_question,
        narrate_result,
        narrate_error,
        generate_conversational_response,
        QuestionType,
    )
 
    q_type = classify_question("How did revenue change in 2017?")
    # QuestionType.DATA - fast exit, no LLM call
 
    q_type = classify_question("Should we expand into new states?")
    # QuestionType.ADVISORY - LLM classified
"""

import logging
import re
from enum import Enum

import pandas as pd

from src.llm.provider import generate_sql, LLMError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question type enum
# ---------------------------------------------------------------------------

class QuestionType(Enum):
    """Classification of user questions by intent.

    DATA: Requires SQL execution. User is asking for numbers, charts, or
        tabular results. Narration is brief - 1-2 sentence summary.
        Example: "What was the total revenue in 2017?"

    ADVISORY: Requires SQL execution AND deeper analysis. User wants
        data-driven recommendations or interpretation. Narration
        goes beyond summarizing - offering reasoning.
        Example: "Which states should we focus on for growth?"

    CONVERSATIONAL: No SQL needed. User is asking about the system,
        the dataset, or simply making general conversation
        Example: "What tables are in our database?"
    """
    DATA = "data"
    ADVISORY = "advisory"
    CONVERSATIONAL = "conversational"
 
 
# ---------------------------------------------------------------------------
# Tier 1: Fast-exit pattern matching
# ---------------------------------------------------------------------------

# Strong signals that a question needs SQL execution.
# These patterns only make sense in a "query the database" context.
# Organized by category, for maintainability.

_DATA_PATTERNS = re.compile(
    r"\b("
    # Aggregation language
    r"how many|how much|total|average|avg|sum|count|minimum|maximum"
    r"|median|percentage|percent"
    # Comparison and ranking
    r"|top \d+|bottom \d+|highest|lowest|most|least|best|worst|rank"
    # Temporal analysis
    r"|monthly|month over month|trend|over time|by year|by quarter"
    r"|year over year|growth rate|by month"
    # Explicit data requests
    r"|show me|list all|give me|breakdown|distribution"
    # Domain-specific terms (Olist schema)
    r"|revenue|orders|sellers|customers|reviews|payment"
    r"|freight|delivery time|product category|product categories|order status"
    r"|avg review|review score|shipping"
    r")\b",
    re.IGNORECASE,
)
 
# Strong signals that a question is conversational (no SQL needed).
_CONVERSATIONAL_PATTERNS = re.compile(
    r"^("
    # Greetings
    r"hi\b|hello\b|hey\b|good morning|good afternoon|good evening"
    r"|what'?s up"
    # Meta questions about the system
    r"|what can you do|how do you work|help$|what are you"
    r"|who are you|what is this"
    # General knowledge about Olist (no data needed)
    r"|what is olist|tell me about olist|explain olist"
    r"|what tables|what data|describe the (database|dataset|schema)"
    r"|what columns"
    r")",
    re.IGNORECASE,
)
 
# Advisory signals — questions that want interpretation, not just data.
# These are checked AFTER data patterns, so a question needs BOTH
# advisory language AND data-related content to be classified as ADVISORY
# at the heuristic level.
_ADVISORY_PATTERNS = re.compile(
    r"\b("
    r"should we|should i|recommend|suggestion|advise|advice"
    r"|what would you|do you think|strategy|focus on|prioritize"
    r"|improve|optimize|opportunity|risk|concern|insight"
    r"|why do you think|what explains|how can we"
    r"|if we had to|downsize|expand|invest"
    r")\b",
    re.IGNORECASE,
)


def _classify_heuristic(question: str) -> QuestionType | None:
    """Tier 1: Attempt to classify a question using pattern matching.

    Returns a QuestionType if the classification is obvious, or None if
    the question is ambiguous and should fall through to the LLM's judgement.

    Check order matters:
        1. Conversational patterns are checked first (anchored to start
            of String - to maximize precision).
        2. If both advisory AND data patterns match, classify as ADVISORY.
        3. If only data patterns match, classify as DATA.
        4. If nothing matches confidently, return None - Ambiguous.

    Args:
        question: The user's natural-language question.
    
    Returns:
        QuestionType if confident, None if ambiguous.
    """
    stripped = question.strip()

    # Very short inputs are likely greetings or noise
    if len(stripped) < 3:
        return QuestionType.CONVERSATIONAL
    
    # Check conversational first - anchored to String start, making
    # false positives very rare (routing a conversational question to SQL pipeline)
    if _CONVERSATIONAL_PATTERNS.match(stripped):
        return QuestionType.CONVERSATIONAL
    
    has_data_signal = bool(_DATA_PATTERNS.search(stripped))
    has_advisory_signal = bool(_ADVISORY_PATTERNS.search(stripped))

    # Advisory = wants data AND interpretation
    if has_data_signal and has_advisory_signal:
        return QuestionType.ADVISORY
    
    # Pure data question
    if has_data_signal:
        return QuestionType.DATA
    
    # Advisory language without clear data signal — ambiguous,
    # let the LLM decide (could be conversational advice-seeking)
    # No match at all — also ambiguous
    return None


# ---------------------------------------------------------------------------
# Tier 2: LLM-based classification (fallback for ambiguous questions)
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM_PROMPT = """\
You classify user questions for a BI chatbot connected to a Brazilian \
e-commerce database (Olist marketplace). The database contains tables \
for orders, customers, products, sellers, reviews, and payments.

Respond with EXACTLY one word - DATA, ADVISORY, or CONVERSATIONAL.

DATA: The user wants specific numbers, lists, charts, or tabular results \
that require querying the database.
Examples: "Show sales by state", "What's the average delivery time?", \
"How did revenue change over time?"
 
ADVISORY: The user wants data-driven recommendations, strategy advice, \
or interpretation that requires querying the database AND reasoning \
about the results.
Examples: "Which states should we prioritize for growth?", \
"Is our marketplace healthy?", "What should we do about low review scores?"
 
CONVERSATIONAL: The user is asking about the system itself, the dataset \
in general terms, or making conversation. No database query needed.
Examples: "What kind of questions can I ask?", "Tell me about this dataset", \
"Hi there"
 
Respond with exactly one word: DATA, ADVISORY, or CONVERSATIONAL.
"""


def _classify_llm(question: str) -> QuestionType:
    """Tier 2: Classify a question using the LLM.

    Called only when heuristic classification returns None (ambiguous).
    Costs ~$0.003 and ~0.5-1s compute time.

    Args:
        question: The user's natural-language question.
 
    Returns:
        QuestionType based on the LLM's classification.
    """
    messages = [{"role": "user", "content": question}]
 
    try:
        response = generate_sql(
            system_prompt=_CLASSIFICATION_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=10,  # We only need one word back
        )
        classification = response.strip().upper()
 
        if classification == "ADVISORY":
            return QuestionType.ADVISORY
        elif classification == "CONVERSATIONAL":
            return QuestionType.CONVERSATIONAL
        else:
            # Default to DATA for any unexpected response —
            # safer to run a query than to skip one
            return QuestionType.DATA
 
    except LLMError as e:
        # If the classification call fails, default to DATA.
        # Better to attempt SQL generation (which has its own
        # error handling) than to give a non-answer.
        logger.warning(f"LLM classification failed, defaulting to DATA: {e}")
        return QuestionType.DATA
    

# ---------------------------------------------------------------------------
# Public API: Question Classification
# ---------------------------------------------------------------------------

def classify_question(question: str) -> QuestionType:
    """Classify a user question as DATA, ADVISORY, or CONVERSATIONAL.
    
    Uses a hybrid approach:
        - Tier 1: Fast pattern matching for obvious cases (no LLM call required).
        - Tier 2: LLM classification for ambiguous or edge questions.

    Args:
        question: The user's natural-language question.
 
    Returns:
        QuestionType indicating how the pipeline should handle this question.
    """
    # Tier 1: Try heuristic classification
    heuristic_result = _classify_heuristic(question)

    if heuristic_result is not None:
        logger.info(
            f"Question classified as {heuristic_result.value} "
            f"(heuristic fast exit)"
        )
        return heuristic_result
    
    # Tier 2: Fall back to LLM
    logger.info("Question is ambiguous — using LLM classification")
    llm_result = _classify_llm(question)
    logger.info(f"Question classified as {llm_result.value} (LLM)")
    return llm_result


# ---------------------------------------------------------------------------
# Conversational Response Generator
# ---------------------------------------------------------------------------

_CONVERSATIONAL_SYSTEM_PROMPT = """\
You are QueryMind, a conversational BI assistant connected to the Olist \
Brazilian e-commerce database. The database contains (reliable) data from January \
2017 through August 2018, covering ~100k orders across 9 tables: \
orders, order items, products, sellers, customers, payments, reviews, \
geolocation, and product category translations.

The user has asked a conversational question (not a data-related query). Respond \
helpfully and concisely (2-4 sentences). If they're asking what you can \
do, give 2-3 example questions they could ask. If they're greeting you, \
be friendly and brief. If they're asking about the dataset, summarize \
what's available.

Do NOT generate SQL. Do NOT make up specific numbers or statistics.
"""


def generate_conversational_response(question: str) -> str:
    """Generate a direct response for conversational (non-data) questions.

    Called when the classifier determines no SQL generation/execution is needed.
    Provides helpful information about the system, the dataset, or even
    responds to greetings.
    
    Args:
        question: The user's conversational question.
 
    Returns:
        A natural-language response string.
    """
    messages = [{"role": "user", "content": question}]

    try:
        response = generate_sql( # Old function "generate_sql" has not been renamed for now
            system_prompt=_CONVERSATIONAL_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=256,
        )
        return response.strip()
    
    except LLMError as e:
        logger.error(f"Conversational response generation failed: {e}")
        return (
            "I'm QueryMind, a conversational BI assistant for the Olist "
            "e-commerce dataset. You can ask me data questions like "
            "'What was total revenue in 2017?' or 'Show me monthly order "
            "trends.' \nHow can I help?"
        )
    

# ---------------------------------------------------------------------------
# Result Narration
# ---------------------------------------------------------------------------

_DATA_NARRATION_SYSTEM_PROMPT = """\
You are a BI assistant summarizing query results. Given the user's \
original question, the SQL that was executed, and the outputted result data. \
write a brief 1-2 sentence summary of key findings.

RULES:
1. Lead with the most important number or insight.
2. Use natural language, not technical jargon.
3. Format large numbers with commas (e.g., 1,234,567).
4. Use R$ for currency values (Brazilian Reais).
5. If the data shows a clear trend or outlier, mention it.
6. Do NOT repeat the SQL or explain how the query works.
7. Keep it to 1-2 sentences maximum.
"""

_ADVISORY_NARRATION_SYSTEM_PROMPT = """\
You are a senior BI analyst providing data-driven recommendations. \
Given the user's original question, the SQL that was executed, and the \
outputted result data, provide a concise analytical response.

RULES:
1. Start with a brief summary of what the data shows (1-2 sentences).
2. Then provide 2-3 sentences of analysis, interpretation, or \
   actionable recommendations based on the data patterns.
3. Use natural language. Format numbers with commas, use R$ for currency.
4. Ground every claim in the actual data - do not speculate beyond \
   what the numbers show.
5. If the data shows clear winners/losers, trends, or outliers, \
   highlight them and explain their significance.
6. Do NOT repeat the SQL or explain how the query works.
7. Keep the total response to 3-5 sentences.
"""


def _format_result_for_narration(
        df: pd.DataFrame,
        max_rows: int = 30,
) -> str:
    """Format a DataFrame as a compact string for the narration prompt.

    Truncates large results to keep the prompt short and focused.
    LLM doesn't need to get 1,000+ rows to write a summary - the first
    30 rows, alongside shape metadata should be sufficient.
    
        Args:
        df: The query result DataFrame.
        max_rows: Maximum number of rows to include in the prompt.
 
    Returns:
        A string representation of the data for the LLM.
    """
    total_rows = len(df)

    if total_rows <= max_rows:
        data_str = df.to_string(index=False)
    else:
        data_str = (
            df.head(max_rows).to_string(index=False)
            + f"\n\n... ({total_rows - max_rows} more rows not shown, "
            f"{total_rows} total)"
        )
 
    return data_str


def narrate_result(
        question: str,
        sql: str,
        df: pd.DataFrame,
        question_type: QuestionType = QuestionType.DATA,
) -> str:
    """Generate a natural-language summary of query results.

    Adapts the narration depth based on the question type:
        - DATA: Brief -> 1-2 sentences summary of key findings.
        - ADVISORY: Deeper -> 3-5 sentences analysis, with recommendations.
    
    Args:
        question: The original user question.
        sql: The SQL that was executed.
        df: The query result DataFrame.
        question_type: DATA or ADVISORY, controls narration depth.
 
    Returns:
        A natural-language narration string.
    """
    # Choose the appropriate system prompt
    if question_type == QuestionType.ADVISORY:
        system_prompt = _ADVISORY_NARRATION_SYSTEM_PROMPT
    else:
        system_prompt = _DATA_NARRATION_SYSTEM_PROMPT

    # Format the data for the prompt
    data_str = _format_result_for_narration(df)

    user_message = (
        f"USER QUESTION: {question}\n\n"
        f"SQL EXECUTED:\n{sql}\n\n"
        f"RESULT DATA ({len(df)} rows, {len(df.columns)} columns):\n"
        f"{data_str}"
    )

    messages = [{"role": "user", "content": user_message}]
 
    try:
        response = generate_sql(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=512,
        )
        return response.strip()
 
    except LLMError as e:
        logger.error(f"Result narration failed: {e}")
        # Graceful fallback — at least tell the user something
        row_word = "row" if len(df) == 1 else "rows"
        return f"Query returned {len(df)} {row_word}."
    

# ---------------------------------------------------------------------------
# Error / Empty Result Narration
# ---------------------------------------------------------------------------

_ERROR_NARRATION_SYSTEM_PROMPT="""\
You are a BI assistant explaining why a query failed or returned no \
results. Given the user's question and the error or empty result, \
write a brief but helpful 1-2 sentence explanation.

Rules:
1. Explain in plain language - no stack traces or technical errors.
2. If the result is empty, suggest possible reasons (e.g., date range, \
   filters, data coverage). The Olist dataset holds relevant data from \
   January 2017 through August 2018.
3. If it was a validation failure, explain what was blocked and why \
   (e.g., restricted data, invalid query type).
4. If possible, suggest how to rephrase the question, so that it succeedes.
5. Keep it to 1-2 sentences.
"""


def narrate_error(
        question: str,
        error: str | None = None,
        is_empty: bool = False,
) -> str:
    """Generate a plain-language explanation of a failure or empty result.
 
    Args:
        question: The original user question.
        error: The error message from the pipeline, if any.
        is_empty: True if the query executed but returned zero rows.
 
    Returns:
        A user-friendly explanation string.
    """
    if is_empty:
        context = (
            f"USER QUESTION: {question}\n\n"
            f"The SQL query executed successfully but returned zero rows."
        )
    elif error:
        context = (
            f"USER QUESTION: {question}\n\n"
            f"PIPELINE ERROR: {error}"
        )
    else:
        context = (
            f"USER QUESTION: {question}\n\n"
            f"The query could not be completed for an unknown reason."
        )
 
    messages = [{"role": "user", "content": context}]
 
    try:
        response = generate_sql(
            system_prompt=_ERROR_NARRATION_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=256,
        )
        return response.strip()
 
    except LLMError as e:
        logger.error(f"Error narration failed: {e}")
        # Fallback — use the raw error or a generic message
        if is_empty:
            return (
                "The query returned no results. The Olist dataset covers "
                "January 2017 through August 2018 — try adjusting your "
                "date range or filters."
            )
        elif error:
            return f"The query could not be completed: {error}"
        else:
            return "Something went wrong. Try rephrasing your question."
