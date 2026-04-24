"""
Embed schema metadata, business glossary, and example queries into Chroma DB.

This module reads the 3 YAML knowledge files from config/, converting
each logical unit (i.e. table, glossary entry, example query) into a self-contained
text chunk, and stores them as embeddings in a persistent ChromaDB collection.

Run this module directly to rebuild the vector store:
    python -m src.rag.embedder

The embedder is idempotent - running it again will delete the existing
collection and recreate it from scratch. This ensures the vector store
always reflects the current state of the YAML files.
"""

import logging
from pathlib import Path

import chromadb
import yaml
from src.rag._config import COLLECTION_NAME, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Paths to the 3 YAML knowledge files
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
SCHEMA_PATH = CONFIG_DIR / "schema_metadata.yaml"
GLOSSARY_PATH = CONFIG_DIR / "business_glossary.yaml"
EXAMPLES_PATH = CONFIG_DIR / "example_queries.yaml"

# ChromaDB persistence directory - stores vector database on disk
# so embeddings 'survive' between Python sessions
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_store"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------

def load_yaml(path:Path) -> dict:
    """Load and parse a YAML file, returning its contents as a dictionary"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

# ---------------------------------------------------------------------------
# Chunking Functions
# ---------------------------------------------------------------------------
# Each function below converts one type of YAML content into a list of
# (document_text, metadata) tuples. The document_text is what gets embedded
# The metadata is stored alongside it, for filtering.
# ---------------------------------------------------------------------------

def chunk_schema_tables(schema_data: dict) -> list[tuple[str, dict]]:
    """
    Convert each table definition into a single text chunk.

    Each chunk contains the table name, description, all column descriptions,
    relationship info, and notes - everything the LLM may need to understand
    that table in one self-contained block.
    """
    chunks = []

    for table in schema_data.get("tables", []):
        table_name = table.get("table_name", "unknown")

        # Start building the texxt block for this table
        parts = [
            f"Table: {table_name}",
            f"Description: {table.get('description', '')}",
            f"Row count: {table.get('row_count', 'unknown')}",
            f"Primary key: {table.get('primary_key', 'none')}",
            "",
            "Columns:",
        ]

        # Add each column's description
        for col in table.get("columns", []):
            col_name = col.get("name", "unknown")
            col_dtype = col.get("dtype", "unknown")
            col_desc = col.get("description", "")
            parts.append(f"  - {col_name} ({col_dtype}): {col_desc}")
            
        # Add relationship descriptions
        relationships = table.get("relationships", [])
        if relationships:
            parts.append("")
            parts.append("Relationships:")
            for rel in relationships:
                target = rel.get("target_table", "")
                join_key = rel.get("join_key", "")
                rel_type = rel.get("relationship_type", "")
                rel_desc = rel.get("description", "")
                parts.append(
                    f"  - {rel_type} to {target} on {join_key}: {rel_desc}"
                )

        # Add notes if present
        notes = table.get("notes", "")
        if notes:
            parts.append("")
            parts.append(f"Notes: {notes}")

        # Combine into a single text document
        document_text = "\n".join(parts)

        metadata = {
            "source_type": "schema",
            "table_name": table_name,
            "source_file": "schema_metadata.yaml",
        }

        chunks.append((document_text, metadata))

    return chunks


def chunk_join_paths(schema_data: dict) -> list[tuple[str, dict]]:
    """
    Converts each join path into a separate text chunk.

    Join paths describe common multi-table query patterns with SQL examples.
    They are embedded separately from table descriptions so the retriever
    can pull both both a relevant table AND a relevant join pattern.
    """
    chunks = []

    for jp in schema_data.get("join_paths", []):
        name = jp.get("name", "unknown")
        description = jp.get("description", "")
        sql_pattern = jp.get("sql_pattern", "")

        parts = [
            f"Join pattern: {name}",
            f"Description: {description}",
        ]

        if sql_pattern:
            parts.append(f"SQL pattern:\n{sql_pattern}")

        document_text = "\n".join(parts)

        metadata = {
            "source_type": "join_path",
            "join_path_name": name,
            "source_file": "schema_metadata.yaml",
        }

        chunks.append((document_text, metadata))

    return chunks


def chunk_glossary(glossary_data: dict) -> list[tuple[str, dict]]:
    """
    Convert each glossary entry into a text chunk.

    Includes the term, all aliases (which broaden retrieval matching),
    the definition, SQL expression if available, and notes.
    """
    chunks = []

    for entry in glossary_data.get("glossary", []):
        term = entry.get("term", "unknown")
        aliases = entry.get("aliases", [])
        definition = entry.get("definition", "")
        sql_expr = entry.get("sql_expression", "")
        source_table = entry.get("source_table", "")
        notes = entry.get("notes", "")

        parts = [
            f"Business term: {term}",
            f"Also known as: {', '.join(aliases)}" if aliases else "",
            f"Definition: {definition}",
        ]

        if sql_expr:
            parts.append(f"SQL expression: {sql_expr}")
        if source_table:
            parts.append(f"Source table: {source_table}")
        if notes:
            parts.append(f"Notes: {notes}")

        # Filter out empty strings before joining
        document_text = "\n".join(part for part in parts if part)

        metadata = {
            "source_type": "glossary",
            "term": term,
            "source_file": "business_glossary.yaml",
        }

        chunks.append((document_text, metadata))
    
    return chunks


def chunk_examples(examples_data: dict) -> list[tuple[str, dict]]:
    """
    Convert each example-question SQL pair into a text chunk.

    Embedded text includes the question (for similarity matching),
    the SQL (as a pattern for the LLM to recognize), and the reasoning
    (to help the LLM understand why SQL is strucutred in that way).
    """
    chunks = []

    for example in examples_data.get("examples", []):
        question = example.get("question", "")
        difficulty = example.get("difficulty", "unknown")
        sql = example.get("sql", "")
        reasoning = example.get("reasoning", "")

        parts = [
            f"Question: {question}",
            f"Difficulty: {difficulty}",
            f"SQL:\n{sql}",
        ]

        if reasoning:
            parts.append(f"Reasoning: {reasoning}")

        document_text = "\n".join(parts)

        metadata = {
            "source_type": "example",
            "difficulty": difficulty,
            "source_file": "example_queries.yaml",
        }

        chunks.append((document_text, metadata))
    
    return chunks


# ---------------------------------------------------------------------------
# Embedding and Storage
# ---------------------------------------------------------------------------

def build_vector_store() -> chromadb.Collection:
    """
    Read all YAML files, chunk them, and store embeddings in ChromaDB.

    This function is idempotent - deletes any existing collection and rebuilds
    from scratch. Ensures consistency with every time YAML files are loaded.

    Returns:
        The ChromaDB collection containing all embedded chunks.
    """
    # --- Load YAML files ---
    logger.info("Loading YAML knowledge files...")

    schema_data = load_yaml(SCHEMA_PATH)
    glossary_data = load_yaml(GLOSSARY_PATH)
    examples_data = load_yaml(EXAMPLES_PATH)

    # --- Chunk all content, based on logic defined above ---
    logger.info("Chunking content...")

    all_chunks: list[tuple[str, dict]] = []
    all_chunks.extend(chunk_schema_tables(schema_data))
    all_chunks.extend(chunk_join_paths(schema_data))
    all_chunks.extend(chunk_glossary(glossary_data))
    all_chunks.extend(chunk_examples(examples_data))

    logger.info(f"Created {len(all_chunks)} chunks total")

    # Log breakdown by source type for verification
    type_counts: dict[str, int] = {}
    for _, metadata in all_chunks:
        source_type = metadata["source_type"]
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
    for source_type, count in sorted(type_counts.items()):
        logger.info(f"  {source_type}: {count} chunks")

    # --- Initialize ChromaDB with persistence ---
    logger.info(f"Initializing ChromaDB at {CHROMA_DIR}...")

    # Create the directory if it doesn't exist
    CHROMA_DIR.mkdir(parents=True, exist_ok = True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists - IDEMPOTENT
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except (ValueError, chromadb.errors.NotFoundError):
        # Collection doesn't exist yet - acceptable
        pass

    # Create a new collection with the sentence-transfomers embedding function.
    # ChromaDB has built-in support for sentence-transformers - when specifying
    # model name, it automatically handles embedding documents and queries
    # using the same model - this is critical for retrieval quality.
    embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "QueryMind RAG knowledge base"},
    )

    # --- Adding all chunks to the collection ---
    logger.info("Embedding and storing chunks...")

    # ChromaDB expects parallel lists of ids, documents, and metadatas.
    # IDs must be unique strings - use format "type_index"
    # (e.g. "schema_0", "glossary_3", "example_15", etc.)
    ids = []
    documents = []
    metadatas = []

    for i, (doc_text, metadata) in enumerate(all_chunks):
        source_type = metadata["source_type"]
        chunk_id = f"{source_type}_{i}"

        ids.append(chunk_id)
        documents.append(doc_text)
        metadatas.append(metadata)

    # Adding all documents in single batch call.
    # ChromaDB handles embedding internally using the embedding_function
    # previously configured. For ~60 chunks with MiniLM, expect
    # this to take about 2-5 seconds on CPU.
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        f"Successfully stored {collection.count()} chunks in "
        f"collection '{COLLECTION_NAME}'"
    )

    return collection


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
# Execute with: python -m src.rag.embedder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting QueryMind knowledge base embedding...")
    collection = build_vector_store()
    logger.info("Done! Vector store is ready for retrieval.")

    # Quick sanity check - run test query to verify retrieval works
    print("\n" + "=" * 60)
    print("SANITY CHECK — Test query: 'monthly revenue trend'")
    print("=" * 60)

    results = collection.query(
        query_texts=["monthly revenue trend"],
        n_results=5,
    )

    for i, (doc, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        print(f"\n--- Result {i + 1} [{metadata['source_type']}] ---")
        # Print first 200 chars of each result for a quick preview
        print(doc[:200] + "..." if len(doc) > 200 else doc)