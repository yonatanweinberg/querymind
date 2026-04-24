"""Shared constants for the RAG subsystem.

Kept in a single module so embedder.py and retriever.py cannot drift
out of sync - a mismatch between them would silently break retrieval.
"""

# Name of ChromaDB collection
COLLECTION_NAME = "querymind_knowledge"

# Embedding model - runs locally on CPU, no API key needed
# all-MiniLM-L6-v2 produces 384-dimensional vectors
# Considered the standard lightweight choice for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"