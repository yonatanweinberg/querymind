"""
Retrieve relevant context from the ChromaDB vector store for a user question.

This module connects to the persistent ChromaDB collection built by the
embedder and retrieves the most relevant schema descriptions, glossary
definitions, example queries, and join paths for a given natural-language
question.

Uses stratified retrieval: queries each source type separately to ensure
the LLM always receives a balanced mix of context types, rather than
getting an unbalanced set dominated by one type.

Usage:
    from src.rag.retriever import retrieve_context

    context = retrieve_context("What was the total revenue in Q1 2018?")
    print(context.formatted_prompt)  # Ready to inject into LLM prompt
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from src.config import get_settings
from src.rag._config import COLLECTION_NAME, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Chunk counts (n_schema, n_glossary, n_examples, n_join_paths) are loaded
# from config/settings.yaml via src.config.get_settings(). These four
# counts are the knobs the RAG ablation study sweeps.
#
# Note: EMBEDDING_MODEL and COLLECTION_NAME deliberately stay in
# src/rag/_config.py - they must match between embedder and retriever and
# belong with the embedder's own config.

# Must match the embedder's configuration exactly
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_store"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store."""

    text: str
    source_type: str
    metadata: dict
    distance: float  # Lower = more similar (ChromaDB uses L2 distance)

    @property
    def display_label(self) -> str:
        """Compact human-readable label for this chunk.

        Combines the source type with a type-specific identifier from
        metadata, e.g. 'schema:olist_orders' or 'glossary:revenue'.
        Used by the UI to label retrieved chunks at a glance.

        Example chunks have no unique identifier in their metadata
        (only difficulty and source_file, neither uniquely identifying),
        so they label as just 'example'. Same fallback for any unknown
        source_type that may be added in the future.

        If a recognized source_type has its expected identifier missing
        from metadata (malformed embedding), falls back to '<type>:?'
        rather than crashing the UI.
        """
        # Map each source_type to the metadata key that names it.
        # Must stay in sync with chunk_*() functions in embedder.py -
        # if a new chunk type with a unique identifier is added there,
        # add the mapping here too.
        identifier_keys = {
            "schema": "table_name",
            "glossary": "term",
            "join_path": "join_path_name",
            # "example" intentionally absent - examples carry no
            # uniquely identifying field in their metadata.
        }
        key = identifier_keys.get(self.source_type)
        if key is None:
            return self.source_type
        identifier = self.metadata.get(key, "?")
        return f"{self.source_type}:{identifier}"


@dataclass
class RetrievalResult:
    """
    Complete retrieval result for a single user question.

    Contains the raw chunks organized by type, plus a pre-formatted
    string ready for injection into the LLM prompt.
    """

    question: str
    schema_chunks: list[RetrievedChunk] = field(default_factory=list)
    glossary_chunks: list[RetrievedChunk] = field(default_factory=list)
    example_chunks: list[RetrievedChunk] = field(default_factory=list)
    join_path_chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def all_chunks(self) -> list[RetrievedChunk]:
        """All retrieved chunks in a flat list."""
        return (
            self.schema_chunks
            + self.glossary_chunks
            + self.example_chunks
            + self.join_path_chunks
        )

    @property
    def formatted_prompt(self) -> str:
        """
        Format all retrieved context into a structured string for the LLM prompt.

        Organizes context into clearly labeled sections so the LLM can
        distinguish between schema info, business definitions, SQL patterns,
        and example queries.
        """
        sections = []

        # --- Schema context ---
        if self.schema_chunks:
            schema_texts = []
            for chunk in self.schema_chunks:
                schema_texts.append(chunk.text)
            sections.append("DATABASE SCHEMA:\n" + "\n\n".join(schema_texts))

        # --- Join paths ---
        if self.join_path_chunks:
            jp_texts = []
            for chunk in self.join_path_chunks:
                jp_texts.append(chunk.text)
            sections.append("JOIN PATTERNS:\n" + "\n\n".join(jp_texts))

        # --- Business glossary ---
        if self.glossary_chunks:
            glossary_texts = []
            for chunk in self.glossary_chunks:
                glossary_texts.append(chunk.text)
            sections.append("BUSINESS DEFINITIONS:\n" + "\n\n".join(glossary_texts))

        # --- Example queries ---
        if self.example_chunks:
            example_texts = []
            for chunk in self.example_chunks:
                example_texts.append(chunk.text)
            sections.append("EXAMPLE QUERIES:\n" + "\n\n".join(example_texts))

        return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Cached collection
# ---------------------------------------------------------------------------
# Lazy singleton: collection (and its underlying SentenceTransformer
# embedding model) is loaded on first retrieve_context() call and reused
# for the rest of the process. The embedding model load is the dominant
# cost - roughly 1 second per call - caching turns retrieval from
# "slow once, slow always" to "slow once, fast forever".
_collection: chromadb.Collection | None = None


def _reset_collection() -> None:
    # Clear the cached collection. Primarily used for internal tests
    global _collection
    _collection = None


# ---------------------------------------------------------------------------
# Collection Access
# ---------------------------------------------------------------------------


def get_collection() -> chromadb.Collection:
    """
    Connect to the existing ChromaDB collection (cached after first call).

    Requires that the embedder has been run at least once to create
    the collection. Raises FileNotFoundError if the ChromaDB directory
    doesn't exist, or ValueError if the collection hasn't been created.
    """
    global _collection
    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DIR}. "
            "Run the embedder first: python -m src.rag.embedder"
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Use the same embedding function as the embedder so that query
    # embeddings are in the same vector space as stored embeddings.
    # This is critical - mismatched embedding models produce garbage results.
    embedding_function = (
        chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    )

    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    logger.info(
        f"Connected to collection '{COLLECTION_NAME}' with {_collection.count()} chunks"
    )

    return _collection


# ---------------------------------------------------------------------------
# Retrieval Logic
# ---------------------------------------------------------------------------


def _query_by_type(
    collection: chromadb.Collection,
    question: str,
    source_type: str,
    n_results: int,
) -> list[RetrievedChunk]:
    """
    Query the collection for chunks of a specific source type.

    Uses ChromaDB's metadata filtering to restrict results to a single
    source type, then returns the top-n most similar chunks.
    """
    if n_results <= 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where={"source_type": source_type},
    )

    chunks = []
    # ChromaDB returns parallel lists wrapped in an outer list
    # (one inner list per query text - only send one)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source_type=source_type,
                metadata=meta,
                distance=dist,
            )
        )

    return chunks


def retrieve_context(
    question: str,
    n_schema: int | None = None,
    n_glossary: int | None = None,
    n_examples: int | None = None,
    n_join_paths: int | None = None,
    collection: chromadb.Collection | None = None,
) -> RetrievalResult:
    """
    Retrieve relevant context for a user question using stratified retrieval.

    Queries each source type separately to ensure the LLM receives a
    balanced mix of schema descriptions, business definitions, SQL examples,
    and join patterns.

    Args:
        question: The natural-language question from the user.
        n_schema: Number of schema chunks to retrieve. If None, falls back
            to settings.rag.n_schema.
        n_glossary: Number of glossary chunks. If None, falls back to
            settings.rag.n_glossary.
        n_examples: Number of example-query chunks. If None, falls back
            to settings.rag.n_examples.
        n_join_paths: Number of join-path chunks. If None, falls back to
            settings.rag.n_join_paths.
        collection: Optional pre-loaded ChromaDB collection. If None,
            connects to the persistent collection automatically.

    Returns:
        A RetrievalResult containing all retrieved chunks and a
        pre-formatted prompt string.
    """
    # Resolve defaults from config. Callers can still override any count
    # explicitly; None here means "use the configured value".
    rag_settings = get_settings().rag
    if n_schema is None:
        n_schema = rag_settings.n_schema
    if n_glossary is None:
        n_glossary = rag_settings.n_glossary
    if n_examples is None:
        n_examples = rag_settings.n_examples
    if n_join_paths is None:
        n_join_paths = rag_settings.n_join_paths

    if collection is None:
        collection = get_collection()

    logger.info(f"Retrieving context for: '{question}'")

    # Stratified retrieval - query each source type independently
    schema_chunks = _query_by_type(collection, question, "schema", n_schema)
    glossary_chunks = _query_by_type(collection, question, "glossary", n_glossary)
    example_chunks = _query_by_type(collection, question, "example", n_examples)
    join_path_chunks = _query_by_type(collection, question, "join_path", n_join_paths)

    result = RetrievalResult(
        question=question,
        schema_chunks=schema_chunks,
        glossary_chunks=glossary_chunks,
        example_chunks=example_chunks,
        join_path_chunks=join_path_chunks,
    )

    # Log what was retrieved
    total = len(result.all_chunks)
    logger.info(
        f"Retrieved {total} chunks: "
        f"{len(schema_chunks)} schema, "
        f"{len(glossary_chunks)} glossary, "
        f"{len(example_chunks)} examples, "
        f"{len(join_path_chunks)} join paths"
    )

    return result


# ---------------------------------------------------------------------------
# CLI Entry Point - for testing retrieval interactively
# ---------------------------------------------------------------------------
# Run with: python -m src.rag.retriever
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test questions that exercise different parts of the knowledge base
    test_questions = [
        "What was the total revenue in 2017?",
        "Which product categories have the best customer reviews?",
        "How many repeat customers do we have?",
    ]

    print("=" * 70)
    print("QueryMind Retriever - Interactive Test")
    print("=" * 70)

    for question in test_questions:
        print(f"\n{'=' * 70}")
        print(f"QUESTION: {question}")
        print("=" * 70)

        result = retrieve_context(question)

        # Show what was retrieved, organized by type
        for chunk in result.all_chunks:
            label = chunk.source_type.upper()
            distance = f"{chunk.distance:.4f}"
            # Show first 120 chars of each chunk for a quick preview
            preview = chunk.text[:120].replace("\n", " ")
            print(f"  [{label}] (dist={distance}) {preview}...")

        # Show a separator before the formatted prompt
        print(f"\n--- FORMATTED PROMPT ({len(result.formatted_prompt)} chars) ---")
        # Print just the first 500 chars of the formatted prompt
        print(result.formatted_prompt[:500] + "...")
