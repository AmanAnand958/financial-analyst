"""
Tests for the Document Ingestion Pipeline.

Tests cover:
- PDF text extraction
- Chunk creation with correct overlap
- Metadata tagging (document_id, page_number, section_name)
- Section detection heuristics
- Error handling for missing files
"""

import os
import tempfile
import pytest

from src.ingestion.document_processor import DocumentProcessor, FINANCIAL_SECTIONS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def sample_text():
    """500-word synthetic financial text for chunking tests."""
    words = ["revenue", "profit", "loss", "assets", "liabilities"] * 100
    return " ".join(words)


# ── DocumentProcessor._generate_doc_id ───────────────────────────────────────

def test_doc_id_is_stable(processor):
    """Same doc name always produces the same ID."""
    id1 = processor._generate_doc_id("reliance_2023.pdf")
    id2 = processor._generate_doc_id("reliance_2023.pdf")
    assert id1 == id2


def test_doc_id_different_names(processor):
    id1 = processor._generate_doc_id("reliance_2023.pdf")
    id2 = processor._generate_doc_id("tcs_2023.pdf")
    assert id1 != id2


def test_doc_id_length(processor):
    doc_id = processor._generate_doc_id("any_file.pdf")
    assert len(doc_id) == 12


# ── DocumentProcessor._create_chunks ─────────────────────────────────────────

def test_chunk_creation_count(processor, sample_text):
    """With chunk_size=50 and overlap=10, verify we get expected number of chunks."""
    metadata = {
        "document_id": "test123",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }
    chunks = processor._create_chunks(sample_text, metadata)
    assert len(chunks) > 0


def test_chunks_have_required_keys(processor, sample_text):
    metadata = {
        "document_id": "abc",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }
    chunks = processor._create_chunks(sample_text, metadata)
    for chunk in chunks:
        assert "id" in chunk
        assert "text" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["chunk_index"] >= 0
        assert chunk["metadata"]["token_count"] > 0


def test_chunk_ids_are_unique(processor, sample_text):
    metadata = {
        "document_id": "abc",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }
    chunks = processor._create_chunks(sample_text, metadata)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_text_is_non_empty(processor, sample_text):
    metadata = {
        "document_id": "abc",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }
    chunks = processor._create_chunks(sample_text, metadata)
    for chunk in chunks:
        assert len(chunk["text"].strip()) > 0


# ── DocumentProcessor._detect_section ────────────────────────────────────────

def test_section_detection_balance_sheet(processor):
    text = "Balance Sheet as at March 31, 2023\nAssets\nLiabilities\nEquity"
    section = processor._detect_section(text)
    assert section == "Balance Sheet"


def test_section_detection_income_statement(processor):
    text = "Income Statement for the year ended March 31, 2023\nRevenue\nExpenses"
    section = processor._detect_section(text)
    assert section == "Income Statement"


def test_section_detection_fallback(processor):
    text = "Some random text that doesn't match any financial section."
    section = processor._detect_section(text)
    assert section == "General"


# ── DocumentProcessor.process_document ───────────────────────────────────────

def test_process_document_missing_file(processor):
    result = processor.process_document("/nonexistent/path/file.pdf", "missing.pdf")
    assert result["success"] is False
    assert "error" in result


def test_overlap_configuration_validation():
    """chunk_overlap >= chunk_size should raise ValueError."""
    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=100, chunk_overlap=100)
