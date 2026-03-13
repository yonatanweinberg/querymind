"""
End-to-end RAG pipeline test.

This script ties together the complete Phase 2 pipeline:
    1. User asks a question
    2. Retriever finds relevant context from ChromaDB
    3. Prompt builder assembles the LLM message
    4. Anthropic API generates SQL
    5. SQL executes against SQLite
    6. Results are displayed

Run with: python -m src.rag.test_rag_pipeline

Requires ANTHROPIC_API_KEY in your .env file.
"""

import os
import sqlite3
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.rag.retriever import retrieve_context
from src.llm.prompts import build_messages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "olist.db"
MODEL = "claude-sonnet-4-20250514"


def run_question(question: str) -> None:
    """Run a single question through the full RAG pipeline."""

    print(f"\n{'=' * 70}")
    print(f"QUESTION: {question}")
    print("=" * 70)

    # --- Step 1: Retrieve context ---
    result = retrieve_context(question)
    print(f"\nRetrieved: {len(result.schema_chunks)} schema, "
          f"{len(result.glossary_chunks)} glossary, "
          f"{len(result.example_chunks)} examples, "
          f"{len(result.join_path_chunks)} join paths")

    # --- Step 2: Build prompt ---
    system_prompt, messages = build_messages(
        user_question=question,
        rag_context=result.formatted_prompt,
    )

    # --- Step 3: Call LLM ---
    print("\nCalling Claude...")
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )

    generated_sql = response.content[0].text.strip()

    # Check for CANNOT_ANSWER response
    if generated_sql.startswith("CANNOT_ANSWER"):
        print(f"\nLLM Response: {generated_sql}")
        return

    print(f"\nGENERATED SQL:\n{generated_sql}")

    # --- Step 4: Execute SQL ---
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        df = pd.read_sql_query(generated_sql, conn)
        conn.close()

        print(f"\nRESULTS ({len(df)} rows):")
        # Show up to 20 rows, formatted nicely
        with pd.option_context("display.max_columns", None,
                               "display.width", 120):
            print(df.head(20).to_string(index=False))

    except Exception as e:
        print(f"\nSQL EXECUTION ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in .env file")
        print("Add it to your .env: ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)

    # Test questions spanning different difficulty tiers
    test_questions = [
        # Easy — should match example almost directly
        "How many orders were placed in 2017?",

        # Medium — revenue with date filter
        "What was the total revenue in Q1 2018?",

        # Hard — multi-table join with business logic
        "What are the top 5 product categories by average review score?",

        # Edge — requires business context from glossary
        "How many repeat customers do we have?",
    ]

    for question in test_questions:
        run_question(question)

    print(f"\n{'=' * 70}")
    print("RAG pipeline test complete!")
    print("=" * 70)
