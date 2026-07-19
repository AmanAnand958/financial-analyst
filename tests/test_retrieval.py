# Vector search database aur keyword search retriever ke verification tests.
"""
Tests for the Hybrid Retriever.

Uses an in-memory ChromaDB client (no persistent state) to test:
- Adding chunks to the vector store
- Hybrid search returns correct number of results
- Score ranges are valid
- Document deletion
- Empty collection edge cases
"""

# Testing framework packages pytest.
import pytest
# Database client libraries import.
import chromadb

# Target retrievers module packages loading.
from src.retrieval.hybrid_search import HybridRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Dummy segment data objects lists coordinates setup options properties.
SAMPLE_CHUNKS = [
    # Chunk one configs.
    {
        # Unique segment ID.
        "id": "chunk_001",
        # String text content details annual records.
        "text": "Reliance Industries reported consolidated revenue of ₹2,40,000 Crore in FY2023.",
        # Nested properties parameters mappings configs.
        "metadata": {
            # Parent doc index string.
            "document_id": "reliance123",
            # Human name targets files.
            "document_name": "reliance_2023.pdf",
            # Page number.
            "page_number": 45,
            # Category labels.
            "section_name": "Income Statement",
            # Index offset coordinates trace.
            "chunk_index": 0,
            # Words count values.
            "token_count": 14,
            # Timestamps configs paths parameters.
            "upload_timestamp": "2025-01-01T00:00:00Z",
        },  # Metadata end.
    },  # Chunk 1 end.
    # Chunk two configs.
    {
        "id": "chunk_002",
        "text": "TCS reported revenue of ₹2,35,000 Crore for the financial year 2023.",
        "metadata": {
            "document_id": "tcs456",
            "document_name": "tcs_2023.pdf",
            "page_number": 32,
            "section_name": "Income Statement",
            "chunk_index": 0,
            "token_count": 13,
            "upload_timestamp": "2025-01-01T00:00:00Z",
        },
    },
    # Chunk three configs.
    {
        "id": "chunk_003",
        "text": "Infosys cash flow from operations was positive at ₹30,000 Crore.",
        "metadata": {
            "document_id": "infosys789",
            "document_name": "infosys_2023.pdf",
            "page_number": 20,
            "section_name": "Cash Flow",
            "chunk_index": 0,
            "token_count": 11,
            "upload_timestamp": "2025-01-01T00:00:00Z",
        },
    },
    # Chunk four configs.
    {
        "id": "chunk_004",
        "text": "The company faces risks from geopolitical tensions affecting energy prices.",
        "metadata": {
            "document_id": "reliance123",
            "document_name": "reliance_2023.pdf",
            "page_number": 67,
            "section_name": "Risk Factors",
            "chunk_index": 1,
            "token_count": 11,
            "upload_timestamp": "2025-01-01T00:00:00Z",
        },
    },
    # Chunk five configs.
    {
        "id": "chunk_005",
        "text": "Net profit margin improved to 15.2% compared to 13.8% in the prior year.",
        "metadata": {
            "document_id": "tcs456",
            "document_name": "tcs_2023.pdf",
            "page_number": 40,
            "section_name": "Income Statement",
            "chunk_index": 1,
            "token_count": 13,
            "upload_timestamp": "2025-01-01T00:00:00Z",
        },
    },
]  # Chunks array end.


# Fixtures configuration setup retriever client.
@pytest.fixture
# Setup temp retrievers instances wrappers helper function.
def retriever(tmp_path):
    """HybridRetriever using a temp ChromaDB path, no reranker (faster tests)."""
    # Create test retriever mappings coordinates temporary directory paths options.
    return HybridRetriever(
        # Temp database path locations configs.
        chroma_path=str(tmp_path / "chroma"),
        # Collection label setups options.
        collection_name="test_collection",
        # Skip reranker logic parameter configurations.
        reranker_model=None,
    )  # Retriever setup end.


# Loaded collections fixture setups.
@pytest.fixture
# Pre populate segment data checks helper function.
def populated_retriever(retriever):
    """Retriever pre-loaded with sample chunks."""
    # Write segment lists values inside database retriever client interfaces.
    retriever.add_chunks(SAMPLE_CHUNKS)
    # Output loaded objects reference returns.
    return retriever


# ── add_chunks ────────────────────────────────────────────────────────────────

# Database write execution status tests check.
def test_add_chunks_returns_true(retriever):
    # Call write method checks parameters details.
    result = retriever.add_chunks(SAMPLE_CHUNKS)
    # Verify return checks boolean is True.
    assert result is True


# Empty data arrays write verification check tests.
def test_add_chunks_empty_list(retriever):
    # Ingest empty arrays.
    result = retriever.add_chunks([])
    # Check returns True.
    assert result is True


# Database count elements index sizes updates checks tests.
def test_chunk_count_after_add(populated_retriever):
    # Validate count matches input segments list size.
    assert populated_retriever.chunk_count() == len(SAMPLE_CHUNKS)


# Duplicates inputs write safety tests check.
def test_add_chunks_idempotent(populated_retriever):
    """Upserting the same chunks again should not increase count."""
    # Read size before.
    before = populated_retriever.chunk_count()
    # Write identical segments list records.
    populated_retriever.add_chunks(SAMPLE_CHUNKS)
    # Read size after.
    after = populated_retriever.chunk_count()
    # Assert sizes matches equals.
    assert before == after


# ── hybrid_search ─────────────────────────────────────────────────────────────

# Search matches retrieval query tests check.
def test_search_returns_results(populated_retriever):
    # Execute query hybrid search retrieves parameters configs.
    results = populated_retriever.hybrid_search("What was the revenue?", top_k=3)
    # Verify outputs results collections count is positive.
    assert len(results) > 0


# Limit candidate outputs sizes checks tests.
def test_search_top_k_respected(populated_retriever):
    # Call search limiting output size to 2 chunks.
    results = populated_retriever.hybrid_search("revenue profit", top_k=2)
    # Assert return size bounds less than or equal 2.
    assert len(results) <= 2


# Retrieved candidate outputs formats keys validation checks.
def test_search_results_have_required_keys(populated_retriever):
    # Call search.
    results = populated_retriever.hybrid_search("revenue", top_k=3)
    # Loop verify keys.
    for r in results:
        # ID check.
        assert "doc_id" in r
        # Text string.
        assert "text" in r
        # Weight score.
        assert "score" in r
        # Properties configs maps.
        assert "metadata" in r


# Similarity matches scores ranges checks validations tests.
def test_search_scores_in_valid_range(populated_retriever):
    # Call search query parameters.
    results = populated_retriever.hybrid_search("revenue", top_k=5)
    # Iterate segment elements validations checks.
    for r in results:
        # Cross encoder negative boundary checks.
        assert r["score"] >= -1.0
        # Max boundary limits checks.
        assert r["score"] <= 2.0


# Sort validation descending order scores tests.
def test_search_sorted_by_score_desc(populated_retriever):
    # Call search.
    results = populated_retriever.hybrid_search("revenue", top_k=5)
    # Extract scores lists.
    scores = [r["score"] for r in results]
    # Verify sorted matches.
    assert scores == sorted(scores, reverse=True)


# Empty databases searches outcomes check tests.
def test_search_empty_collection(retriever):
    # Search empty retriever instances.
    results = retriever.hybrid_search("revenue", top_k=3)
    # Assert return is empty list coordinates.
    assert results == []


# Matches relevance semantic accuracy check tests.
def test_search_returns_relevant_chunk(populated_retriever):
    """Revenue query should surface at least one chunk mentioning revenue."""
    # Execute target semantic query.
    results = populated_retriever.hybrid_search("Reliance revenue FY2023", top_k=3)
    # Concatenate texts strings.
    texts = " ".join(r["text"].lower() for r in results)
    # Verify keyword presence.
    assert "revenue" in texts or "reliance" in texts


# ── list_documents ────────────────────────────────────────────────────────────

# Index documents counts calculations verify tests.
def test_list_documents_count(populated_retriever):
    # Retrieve documents.
    docs = populated_retriever.list_documents()
    # Confirm unique document indexes size bounds.
    assert len(docs) == 3


# List documents elements formats checks validations tests.
def test_list_documents_structure(populated_retriever):
    # Fetch list.
    docs = populated_retriever.list_documents()
    # Loop checks keys.
    for doc in docs:
        # ID check.
        assert "document_id" in doc
        # Filename tags.
        assert "document_name" in doc
        # Chunk counts metric variables.
        assert "chunk_count" in doc


# ── delete_document ───────────────────────────────────────────────────────────

# Document delete execution checks validations tests.
def test_delete_document(populated_retriever):
    # Execute delete method on target doc index values.
    deleted = populated_retriever.delete_document("reliance123")
    # Verify deleted count positive.
    assert deleted > 0
    # Read size after deletion completes.
    remaining = populated_retriever.chunk_count()
    # Verify size decrease maps properties.
    assert remaining == len(SAMPLE_CHUNKS) - deleted


# Delete nonexistent documents test check.
def test_delete_nonexistent_document(populated_retriever):
    # Execute delete nonexistent ID targets.
    deleted = populated_retriever.delete_document("does_not_exist")
    # Verify return count is 0 deleted.
    assert deleted == 0
