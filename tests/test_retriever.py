"""
Test suite for retriever output formatting.

Scope is intentionally narrow: covers RetrievalResult.formatted_prompt
only. The actual retrieval (ChromaDB queries, embedding model usage)
is not tested here - those depend on a populated vector store and
would require heavyweight fixtures for limited additional value.
The formatted_prompt property is pure string-assembly logic and
directly determines what the LLM sees on every query, which makes
it the highest-leverage place to lock behavior down.

Run: pytest tests/test_retriever.py -v
"""

from src.rag.retriever import RetrievalResult, RetrievedChunk


def _make_chunk(text: str, source_type: str) -> RetrievedChunk:
    """Build a RetrievedChunk with throwaway metadata.

    The formatted_prompt property only reads .text from each chunk;
    metadata and distance are filled in to satisfy the dataclass but
    don't affect formatting.
    """
    return RetrievedChunk(
        text=text,
        source_type=source_type,
        metadata={},
        distance=0.0,
    )


# ===========================================================================
# Section assembly - each chunk type produces its own labeled section
# ===========================================================================

class TestFormattedPromptSections:

    def test_schema_only(self):
        result = RetrievalResult(
            question="...",
            schema_chunks=[_make_chunk("schema text A", "schema")],
        )
        prompt = result.formatted_prompt
        assert "DATABASE SCHEMA:" in prompt
        assert "schema text A" in prompt
        # The other section headers should NOT appear when their chunks are empty
        assert "JOIN PATTERNS:" not in prompt
        assert "BUSINESS DEFINITIONS:" not in prompt
        assert "EXAMPLE QUERIES:" not in prompt

    def test_glossary_only(self):
        result = RetrievalResult(
            question="...",
            glossary_chunks=[_make_chunk("revenue definition", "glossary")],
        )
        prompt = result.formatted_prompt
        assert "BUSINESS DEFINITIONS:" in prompt
        assert "revenue definition" in prompt
        assert "DATABASE SCHEMA:" not in prompt

    def test_examples_only(self):
        result = RetrievalResult(
            question="...",
            example_chunks=[_make_chunk("Q: ... SQL: SELECT ...", "example")],
        )
        prompt = result.formatted_prompt
        assert "EXAMPLE QUERIES:" in prompt
        assert "Q: ... SQL: SELECT ..." in prompt

    def test_join_paths_only(self):
        result = RetrievalResult(
            question="...",
            join_path_chunks=[_make_chunk("orders -> items -> products", "join_path")],
        )
        prompt = result.formatted_prompt
        assert "JOIN PATTERNS:" in prompt
        assert "orders -> items -> products" in prompt


# ===========================================================================
# Empty results - no chunks at all produces an empty string
# ===========================================================================

class TestEmptyResult:

    def test_no_chunks_produces_empty_string(self):
        # Misconfigured retrieval (n_*=0 across the board) should produce
        # an empty prompt string, not a string of dangling headers.
        result = RetrievalResult(question="...")
        assert result.formatted_prompt == ""


# ===========================================================================
# Section ordering - hardcoded, not driven by chunk insertion order
# ===========================================================================

class TestSectionOrdering:

    def test_canonical_order_with_all_sections(self):
        # Build a result populated in a deliberately scrambled order to
        # confirm the property doesn't leak that ordering into the output.
        result = RetrievalResult(
            question="...",
            example_chunks=[_make_chunk("EXAMPLE_MARKER", "example")],
            schema_chunks=[_make_chunk("SCHEMA_MARKER", "schema")],
            join_path_chunks=[_make_chunk("JOIN_MARKER", "join_path")],
            glossary_chunks=[_make_chunk("GLOSSARY_MARKER", "glossary")],
        )
        prompt = result.formatted_prompt

        # Find the position of each marker in the output and confirm
        # they appear in the canonical order: schema, join, glossary, example.
        schema_idx = prompt.index("SCHEMA_MARKER")
        join_idx = prompt.index("JOIN_MARKER")
        glossary_idx = prompt.index("GLOSSARY_MARKER")
        example_idx = prompt.index("EXAMPLE_MARKER")

        assert schema_idx < join_idx < glossary_idx < example_idx


# ===========================================================================
# Multi-chunk sections - chunks within a section are joined together
# ===========================================================================

class TestMultiChunkSections:

    def test_multiple_chunks_in_same_section(self):
        result = RetrievalResult(
            question="...",
            schema_chunks=[
                _make_chunk("first schema chunk", "schema"),
                _make_chunk("second schema chunk", "schema"),
                _make_chunk("third schema chunk", "schema"),
            ],
        )
        prompt = result.formatted_prompt
        assert "first schema chunk" in prompt
        assert "second schema chunk" in prompt
        assert "third schema chunk" in prompt

        # The single "DATABASE SCHEMA:" header should not be repeated -
        # all three chunks live under one header, separated by blank lines.
        assert prompt.count("DATABASE SCHEMA:") == 1


# ===========================================================================
# Section separator - sections separated by --- divider
# ===========================================================================

class TestSectionSeparator:

    def test_sections_separated_by_divider(self):
        # When two sections are present, the LLM-visible output uses
        # "---" between them so section boundaries are unambiguous.
        result = RetrievalResult(
            question="...",
            schema_chunks=[_make_chunk("schema text", "schema")],
            glossary_chunks=[_make_chunk("glossary text", "glossary")],
        )
        prompt = result.formatted_prompt
        assert "---" in prompt


# ===========================================================================
# all_chunks property - flat-list view across all sections
# ===========================================================================

class TestAllChunksProperty:

    def test_all_chunks_combines_every_section(self):
        # all_chunks is used by the retriever's logging line; lock its
        # contract so a refactor that drops one section type would fail here.
        c1 = _make_chunk("a", "schema")
        c2 = _make_chunk("b", "glossary")
        c3 = _make_chunk("c", "example")
        c4 = _make_chunk("d", "join_path")

        result = RetrievalResult(
            question="...",
            schema_chunks=[c1],
            glossary_chunks=[c2],
            example_chunks=[c3],
            join_path_chunks=[c4],
        )

        all_chunks = result.all_chunks
        assert len(all_chunks) == 4
        assert c1 in all_chunks
        assert c2 in all_chunks
        assert c3 in all_chunks
        assert c4 in all_chunks

    def test_all_chunks_empty_when_no_chunks(self):
        result = RetrievalResult(question="...")
        assert result.all_chunks == []
