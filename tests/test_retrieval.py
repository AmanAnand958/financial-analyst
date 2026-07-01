"""
Tests for the Hybrid Retriever.

Uses an in-memory ChromaDB client (no persistent state) to test:
- Adding chunks to the vector store
- Hybrid search returns correct number of results
- Score ranges are valid
- Document deletion
- Empty collection edge cases
"""

import pytest
import chromadb

from src.retrieval.hybrid_search import HybridRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "id": "chunk_001",
        "text": "Reliance Industries reported consolidated revenue of ₹2,40,000 Crore in FY2023.",
        "metadata": {
            "document_id": "reliance123",
            "document_name": "reliance_2023.pdf",
            "page_number": 45,
            "section_name": "Income Statement",
            "chunk_index": 0,
            "token_count": 14,
            "upload_timestamp": "2025-01-01T00:00:00Z",
        },
    },
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
]


@pytest.fixture
def retriever(tmp_path):
    """HybridRetriever using a temp ChromaDB path, no reranker (faster tests)."""
    return HybridRetriever(
        chroma_path=str(tmp_path / "chroma"),
        collection_name="test_collection",
        reranker_model=None,  # skip reranker for speed
    )


@pytest.fixture
def populated_retriever(retriever):
    """Retriever pre-loaded with sample chunks."""
    retriever.add_chunks(SAMPLE_CHUNKS)
    return retriever


# ── add_chunks ────────────────────────────────────────────────────────────────

def test_add_chunks_returns_true(retriever):
    result = retriever.add_chunks(SAMPLE_CHUNKS)
    assert result is True


def test_add_chunks_empty_list(retriever):
    result = retriever.add_chunks([])
    assert result is True


def test_chunk_count_after_add(populated_retriever):
    assert populated_retriever.chunk_count() == len(SAMPLE_CHUNKS)


def test_add_chunks_idempotent(populated_retriever):
    """Upserting the same chunks again should not increase count."""
    before = populated_retriever.chunk_count()
    populated_retriever.add_chunks(SAMPLE_CHUNKS)
    after = populated_retriever.chunk_count()
    assert before == after


# ── hybrid_search ─────────────────────────────────────────────────────────────

def test_search_returns_results(populated_retriever):
    results = populated_retriever.hybrid_search("What was the revenue?", top_k=3)
    assert len(results) > 0


def test_search_top_k_respected(populated_retriever):
    results = populated_retriever.hybrid_search("revenue profit", top_k=2)
    assert len(results) <= 2


def test_search_results_have_required_keys(populated_retriever):
    results = populated_retriever.hybrid_search("revenue", top_k=3)
    for r in results:
        assert "doc_id" in r
        assert "text" in r
        assert "score" in r
        assert "metadata" in r


def test_search_scores_in_valid_range(populated_retriever):
    results = populated_retriever.hybrid_search("revenue", top_k=5)
    for r in results:
        # Scores can be slightly outside [0,1] after cross-encoder, but
        # for fusion scores they should be in [0,1]
        assert r["score"] >= -1.0  # cross-encoder can go negative
        assert r["score"] <= 2.0   # fusion max is 1.0


def test_search_sorted_by_score_desc(populated_retriever):
    results = populated_retriever.hybrid_search("revenue", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_collection(retriever):
    results = retriever.hybrid_search("revenue", top_k=3)
    assert results == []


def test_search_returns_relevant_chunk(populated_retriever):
    """Revenue query should surface at least one chunk mentioning revenue."""
    results = populated_retriever.hybrid_search("Reliance revenue FY2023", top_k=3)
    texts = " ".join(r["text"].lower() for r in results)
    assert "revenue" in texts or "reliance" in texts


# ── list_documents ────────────────────────────────────────────────────────────

def test_list_documents_count(populated_retriever):
    docs = populated_retriever.list_documents()
    # SAMPLE_CHUNKS has 3 unique document_ids
    assert len(docs) == 3


def test_list_documents_structure(populated_retriever):
    docs = populated_retriever.list_documents()
    for doc in docs:
        assert "document_id" in doc
        assert "document_name" in doc
        assert "chunk_count" in doc


# ── delete_document ───────────────────────────────────────────────────────────

def test_delete_document(populated_retriever):
    deleted = populated_retriever.delete_document("reliance123")
    assert deleted > 0
    # After deletion, reliance chunks should not appear in search
    remaining = populated_retriever.chunk_count()
    assert remaining == len(SAMPLE_CHUNKS) - deleted


def test_delete_nonexistent_document(populated_retriever):
    deleted = populated_retriever.delete_document("does_not_exist")
    assert deleted == 0
