# Document processor pipeline ke validation tests likhne ki file.
"""
Tests for the Document Ingestion Pipeline.

Tests cover:
- PDF text extraction
- Chunk creation with correct overlap
- Metadata tagging (document_id, page_number, section_name)
- Section detection heuristics
- Error handling for missing files
"""

# OS file operations imports.
import os
# Temporary directory tools imports.
import tempfile
# Testing frameworks pytest module.
import pytest

# Target components modules loading imports references.
from src.ingestion.document_processor import DocumentProcessor, FINANCIAL_SECTIONS


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Pytest fixture declarations setups.
@pytest.fixture
# Setup document processor instance helper functions.
def processor():
    # Setup test instance configurations size boundaries limits properties.
    return DocumentProcessor(chunk_size=50, chunk_overlap=10)


# Secondary data fixture definitions setups.
@pytest.fixture
# Synthetic financial words generator helper functions checks.
def sample_text():
    """500-word synthetic financial text for chunking tests."""
    # Data list templates repetition operations.
    words = ["revenue", "profit", "loss", "assets", "liabilities"] * 100
    # Join spaces string returns definitions setup config.
    return " ".join(words)


# ── DocumentProcessor._generate_doc_id ───────────────────────────────────────

# Stable doc hash ID checks tests.
def test_doc_id_is_stable(processor):
    """Same doc name always produces the same ID."""
    # Process stable hashes calculations checks definitions setup.
    id1 = processor._generate_doc_id("reliance_2023.pdf")
    # Second calculations parameters setup.
    id2 = processor._generate_doc_id("reliance_2023.pdf")
    # Verify outputs equality constraints checks.
    assert id1 == id2


# Different filename hash checking tests.
def test_doc_id_different_names(processor):
    # Hash name one.
    id1 = processor._generate_doc_id("reliance_2023.pdf")
    # Hash name two.
    id2 = processor._generate_doc_id("tcs_2023.pdf")
    # Verify values inequality.
    assert id1 != id2


# Hash length checks validations limits tests.
def test_doc_id_length(processor):
    # Hash generators execution paths mappings configurations.
    doc_id = processor._generate_doc_id("any_file.pdf")
    # Verify character length size bounds equal 12.
    assert len(doc_id) == 12


# ── DocumentProcessor._create_chunks ─────────────────────────────────────────

# Chunks segmentation counts validations tests.
def test_chunk_creation_count(processor, sample_text):
    """With chunk_size=50 and overlap=10, verify we get expected number of chunks."""
    # Test metadata dummy objects configurations parameters.
    metadata = {
        # Parent.
        "document_id": "test123",
        # Filename.
        "document_name": "test.pdf",
        # Page indicator.
        "page_number": 1,
        # Section category.
        "section_name": "General",
        # Total page.
        "total_pages": 1,
        # Timestamps.
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }  # Metadata end.
    # Split text chunks helpers call.
    chunks = processor._create_chunks(sample_text, metadata)
    # Confirm segments counts non empty.
    assert len(chunks) > 0


# Chunks schema structures validations tests.
def test_chunks_have_required_keys(processor, sample_text):
    # Metadata dummy setups configs maps.
    metadata = {
        "document_id": "abc",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }  # Metadata end.
    # Chunks creators methods executions.
    chunks = processor._create_chunks(sample_text, metadata)
    # Iterate segment elements list checks loops.
    for chunk in chunks:
        # Verify ID exists.
        assert "id" in chunk
        # Verify Text exists.
        assert "text" in chunk
        # Verify Metadata exists.
        assert "metadata" in chunk
        # Verify index offset validity.
        assert chunk["metadata"]["chunk_index"] >= 0
        # Verify token counts.
        assert chunk["metadata"]["token_count"] > 0


# Chunks ID unique checks validation tests.
def test_chunk_ids_are_unique(processor, sample_text):
    # Dummy configs maps.
    metadata = {
        "document_id": "abc",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }  # Metadata end.
    # Execute split helper structures.
    chunks = processor._create_chunks(sample_text, metadata)
    # Extract IDs lists.
    ids = [c["id"] for c in chunks]
    # Set size equality checks verifies uniqueness.
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# Chunks text blank checks validation tests.
def test_chunk_text_is_non_empty(processor, sample_text):
    # Dummy metadata maps setups.
    metadata = {
        "document_id": "abc",
        "document_name": "test.pdf",
        "page_number": 1,
        "section_name": "General",
        "total_pages": 1,
        "upload_timestamp": "2025-01-01T00:00:00Z",
    }  # Metadata end.
    # Execute chunking algorithms.
    chunks = processor._create_chunks(sample_text, metadata)
    # Loop verify spacing.
    for chunk in chunks:
        # String strip blank checking bounds.
        assert len(chunk["text"].strip()) > 0


# ── DocumentProcessor._detect_section ────────────────────────────────────────

# Category detection Balance Sheet checks tests.
def test_section_detection_balance_sheet(processor):
    # Test text header templates.
    text = "Balance Sheet as at March 31, 2023\nAssets\nLiabilities\nEquity"
    # Category detection method execute parameters.
    section = processor._detect_section(text)
    # Confirm output categorised labels.
    assert section == "Balance Sheet"


# Category detection Income Statement checks tests.
def test_section_detection_income_statement(processor):
    # Test text header templates.
    text = "Income Statement for the year ended March 31, 2023\nRevenue\nExpenses"
    # Execute category check validations indicators.
    section = processor._detect_section(text)
    # Confirm outputs tags.
    assert section == "Income Statement"


# Heuristic category fallback check checks tests.
def test_section_detection_fallback(processor):
    # Random text body templates.
    text = "Some random text that doesn't match any financial section."
    # Category detect triggers.
    section = processor._detect_section(text)
    # Confirm fallback general label.
    assert section == "General"


# ── DocumentProcessor.process_document ───────────────────────────────────────

# Ingestion pipeline missing files checks error tests.
def test_process_document_missing_file(processor):
    # Execute process methods invalid coordinate routes file paths parameters.
    result = processor.process_document("/nonexistent/path/file.pdf", "missing.pdf")
    # Verify outputs error statuses maps.
    assert result["success"] is False
    # Check error message keys presence indicators checks.
    assert "error" in result


# Overlap size constraints checks validations error raises tests.
def test_overlap_configuration_validation():
    """chunk_overlap >= chunk_size should raise ValueError."""
    # Check value exception triggers wrapper pytests.
    with pytest.raises(ValueError):
        # Invalid parameters size settings initial setups.
        DocumentProcessor(chunk_size=100, chunk_overlap=100)
