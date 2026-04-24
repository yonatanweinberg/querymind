"""
Prompt templates for QueryMind's text-to-SQL complete pipeline.

This module defines the system prompt that instructs the LLM on how to
generate SQL, and provides functions to assemble the complete message
payload for the Anthropic API using dynamically retrieved RAG context.

The system prompt is deliberately concise and directive — true value
comes dynamically from the retrieved context (schema, glossary, examples),
not from lengthy instructions.
"""

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
# Sent as system message on every LLM call. Defines the LLM's role,
# output format, and key constraints. Should NOT contain schema
# details - those come from the RAG-retrieved context.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an SQL expert working with a Brazilian e-commerce SQL database (Olist marketplace).

Your job is to generate a single, correct SQLite-compatible SELECT query that answers the user's question.

RULES:
1. Return ONLY the SQL query — no explanations, no markdown, no code fences or wrappers.
2. Use ONLY tables and columns covered in the DATABASE SCHEMA section below.
3. Follow the patterns shown in the EXAMPLE QUERIES section when available.
4. Apply business definitions from the BUSINESS DEFINITIONS section (e.g., revenue formula, customer counting).
5. All dates are stored as TEXT in ISO format. Use STRFTIME() for date extraction and JULIANDAY() for date arithmetic.
6. Filter to order_status = 'delivered' for revenue and delivery analyses unless the user explicitly asks otherwise.
7. Use LEFT JOIN when joining to product_category_name_translation (2 categories lack translations).
8. Use customer_unique_id (not customer_id) when counting distinct customers.
9. Always include a LIMIT clause for queries that could return many rows. Default to LIMIT 100 if not specified.
10. Use ROUND() for decimal results and meaningful column aliases (AS) for readability.
11. When using UNION, wrap each SELECT branch in a subquery if it needs its own ORDER BY or LIMIT: SELECT * FROM (SELECT ... ORDER BY ... LIMIT n) UNION ALL SELECT * FROM (SELECT ... ORDER BY ... LIMIT n).

FALLBACK:
If the question cannot be answered with the available schema, respond with: CANNOT_ANSWER: <brief reason>
"""


# ---------------------------------------------------------------------------
# Message Assembly
# ---------------------------------------------------------------------------

def build_messages(
        user_question: str,
        rag_context: str,
) -> tuple[str, list[dict]]:
    """
    Assemble the system prompt and messages for the Anthropic API.

    Combines the static system prompt, the dynamically retrieved RAG
    context, and the user's question into the format expected by the
    Anthropic messages API.

    Args:
        user_question: Natural-language question from the user.
        rag_context: The formatted context string from
            RetrievalResult.formatted_prompt.

    Returns:
        A tuple of (system_prompt, messages) where:
        - system_prompt is the string to pass as the 'system' parameter
        - messages is the list to pass as the 'messages' parameter
    """
    # User message includes both the RAG context and the question.
    # This structure (context first, question last) follows the
    # "put instructions close to the content they reference" principle
    # and ensures the question is the last thing the LLM reads before
    # generating a response.
    user_message = f"""{rag_context}

---

USER QUESTION: {user_question}

SQL:"""

    messages = [
        {"role": "user", "content": user_message},
    ]

    return SYSTEM_PROMPT, messages


def build_messages_with_history(
    user_question: str,
    rag_context: str,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Assemble message with optional conversation history.

    This is a stretch-goal variant that supports follow-up questions
    within a session. For now, QueryMind treats each question
    independently (no memory across questions), but this function
    is ready for when conversation context is added - next steps.

    Args:
        user_question: Natural-language question from the user.
        rag_context: Formatted context string from retrieval.
        conversation_history: Optional list of previous message dicts
            with 'role' and 'content' keys.

    Returns:
        A tuple of (system_prompt, messages).
    """
    messages = []

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current question with RAG context
    user_message = f"""{rag_context}

---

USER QUESTION: {user_question}

SQL:"""

    messages.append({"role": "user", "content": user_message})

    return SYSTEM_PROMPT, messages