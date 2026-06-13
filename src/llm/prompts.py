"""
Prompt templates for QueryMind's text-to-SQL complete pipeline.

This module defines the system prompt that instructs the LLM on how to
generate SQL, and provides functions to assemble the complete message
payload for the Anthropic API using dynamically retrieved RAG context.

The system prompt is deliberately concise and directive - true value
comes dynamically from the retrieved context (schema, glossary, examples),
not from lengthy instructions.
"""

from src.config import get_settings

# ---------------------------------------------------------------------------
# Prompt version
# ---------------------------------------------------------------------------
# Bumped whenever the system prompt changes meaningfully (rules added,
# wording rewritten, behavior shift). Lets the evaluation phase tag
# results with the prompt version that produced them, so a regression
# in the eval table can be traced to the specific prompt edit.
# Stick to major.minor.patch format:
#   - major: structural change (sections reordered/removed).
#   - minor: rule added or substantially rewritten.
#   - patch: typo fix, wording polish, no behavioral intent change.
PROMPT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
# Sent as system message on every LLM call. Defines the LLM's role,
# output format, and key constraints. Should NOT contain schema
# details - those come from the RAG-retrieved context.
#
# The template contains a {default_limit} placeholder that is filled from
# config/settings.yaml (safety.default_limit) by _render_system_prompt().
# This keeps the prompt's guidance to the LLM in lockstep with the
# validator's actual default - change the YAML value and both surfaces
# update without any code change.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
[QueryMind prompt v{PROMPT_VERSION}]
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
9. NEVER reference columns described as RESTRICTED in the schema - those queries will be rejected by the safety layer. Use the alternative columns suggested in the description instead.
10. Always include a LIMIT clause for queries that could return many rows. Default to {default_limit} if not specified.
11. Use ROUND() for decimal results and meaningful column aliases (AS) for readability.
12. When using UNION, wrap each SELECT branch in a subquery if it needs its own ORDER BY or LIMIT: SELECT * FROM (SELECT ... ORDER BY ... LIMIT n) UNION ALL SELECT * FROM (SELECT ... ORDER BY ... LIMIT n).

FALLBACK:
If the question cannot be answered with the available schema, respond with: CANNOT_ANSWER: <brief reason>
"""


def _render_system_prompt() -> str:
    """Render the system prompt template with current config values.

    Pulls safety.default_limit from settings.yaml at call time so the
    prompt's guidance to the LLM stays in sync with the validator's
    actual default. Both surfaces read from one config - no drift to manage.
    """
    settings = get_settings()
    return SYSTEM_PROMPT_TEMPLATE.format(
        PROMPT_VERSION=PROMPT_VERSION,
        default_limit=settings.safety.default_limit,
    )


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

    return _render_system_prompt(), messages
