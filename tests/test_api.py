# Yeh file API endpoints ko test karne ke liye automated tests contain karti hai.
"""
Integration tests for the FastAPI backend.

Uses TestClient to test all endpoints without starting a real server.
Mocks the HybridRetriever and FinancialAnswerGenerator to avoid
needing real embeddings or an LLM API key during CI.
"""

# Binary data stream memory handles imports.
import io
# JSON output parsers formats configs.
import json
# Testing mocks modules patch parameters unit tests.
from unittest.mock import MagicMock, patch

# Unit testing framework packages.
import pytest
# Testing client tools for FastAPI.
from fastapi.testclient import TestClient

# Environment paths setup configs before loads.
import os
# Temporary directory generators helper tools.
import tempfile
# Set mock API key inside environments configurations.
os.environ["GROQ_API_KEY"] = "test_key_for_ci"
# Set temporary database paths locations.
os.environ["CHROMA_DB_PATH"] = tempfile.mkdtemp()

# Load main application module.
from src.api.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Pytest fixture declarations setups.
@pytest.fixture
# Test client setup helper function.
def client():
    """Create a TestClient with mocked components."""
    # API authentication check dependencies imports.
    from src.api.main import verify_api_key
    # Bypass api authentication check requirements.
    app.dependency_overrides[verify_api_key] = lambda: None
    # Client instance initialization test environments.
    with TestClient(app) as c:
        # Yield client objects interfaces.
        yield c
    # Cleanup dependency overrides maps.
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────

# Health route get endpoint test.
def test_health_check(client):
    # Call health check path get request.
    response = client.get("/health")
    # Verify response code equals 200.
    assert response.status_code == 200
    # Parse output responses payload maps.
    data = response.json()
    # Confirm active status.
    assert data["status"] == "healthy"
    # Confirm version presence.
    assert "version" in data
    # Confirm components states.
    assert "components" in data


# ── /documents ────────────────────────────────────────────────────────────────

# List documents route empty checks test.
def test_list_documents_empty(client):
    # Retrieve documents.
    response = client.get("/documents")
    # Verify status.
    assert response.status_code == 200
    # Parse payload map.
    data = response.json()
    # Confirm details structure targets.
    assert "documents" in data
    # Confirm doc total count exist.
    assert "total_documents" in data
    # Chunks.
    assert "total_chunks" in data


# ── /eval-metrics ─────────────────────────────────────────────────────────────

# Evaluation metrics initial check tests.
def test_eval_metrics_empty(client):
    # Call evaluation metrics endpoints.
    response = client.get("/eval-metrics")
    # Check status 200.
    assert response.status_code == 200
    # Parse output configurations maps.
    data = response.json()
    # Validate context precision properties.
    assert "context_precision" in data
    # Validate faithfulness.
    assert "faithfulness" in data
    # Validate relevancy.
    assert "answer_relevancy" in data
    # Validate total records count indicators.
    assert "total_queries" in data


# ── /upload ───────────────────────────────────────────────────────────────────

# Upload file type validation error check tests.
def test_upload_non_pdf_rejected(client):
    """Only PDF files should be accepted."""
    # Build text byte stream parameters options.
    fake_txt = io.BytesIO(b"this is not a pdf")
    # Call upload POST routes parameters setup configs.
    response = client.post(
        # Upload paths.
        "/upload",
        # File contents body metadata formats configurations.
        files={"file": ("document.txt", fake_txt, "text/plain")},
    )  # Post end.
    # Check status code is bad request 400.
    assert response.status_code == 400


# Upload valid minimal PDF file check tests.
def test_upload_valid_pdf(client, tmp_path):
    """A minimal valid PDF should be processed (even if it extracts no text)."""
    # Minimal PDF bytes format string configurations coordinates setup.
    minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj
2 0 obj<</Type /Pages /Kids [3 0 R] /Count 1>>endobj
3 0 obj<</Type /Page /MediaBox [0 0 3 3]>>endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer<</Size 4 /Root 1 0 R>>
startxref
190
%%EOF"""
    # Open byte streams for minimal pdf data payload.
    pdf_file = io.BytesIO(minimal_pdf)
    # Post upload requests.
    response = client.post(
        # Upload paths.
        "/upload",
        # Payload PDF.
        files={"file": ("test.pdf", pdf_file, "application/pdf")},
    )  # Post end.
    # Assert return code is standard successful 200 or unprocessable 422.
    assert response.status_code in (200, 422)


# ── /query ────────────────────────────────────────────────────────────────────

# Query processing check empty databases scenarios.
def test_query_no_documents(client):
    """Query without any uploaded documents should return 400."""
    # Formulate test request query payloads coordinates setup.
    payload = {
        # Questions.
        "question": "What was the revenue in FY2023?",
        # Chunks count.
        "top_k": 3,
        # Threshold checks bounds.
        "confidence_threshold": 0.6,
    }  # Payload end.
    # POST query requests execution paths mappings.
    response = client.post("/query", json=payload)
    # Check return codes values limits conditions standard.
    assert response.status_code in (400, 503)


# Question input size validation checks tests.
def test_query_validation_short_question(client):
    """Questions shorter than 5 chars should fail validation."""
    # Too short question text maps definitions.
    payload = {
        # Short question.
        "question": "Hi",
        # Chunks.
        "top_k": 3,
        # Thresholds.
        "confidence_threshold": 0.6,
    }  # Payload end.
    # Post query request check validations.
    response = client.post("/query", json=payload)
    # Check response status is 422 validation error.
    assert response.status_code == 422


# Top K value parameter boundary limits verification tests.
def test_query_validation_invalid_top_k(client):
    # Invalid zero chunk selection request body setup.
    payload = {
        # Question.
        "question": "What was the revenue?",
        # Chunks limit zero (invalid).
        "top_k": 0,
        # Thresholds.
        "confidence_threshold": 0.6,
    }  # Payload end.
    # Post query requests.
    response = client.post("/query", json=payload)
    # Check validation error response code 422.
    assert response.status_code == 422


# ── /documents/{id} DELETE ────────────────────────────────────────────────────

# Delete documents check invalid IDs scenarios.
def test_delete_nonexistent_document(client):
    # Call delete routes with nonexistent ID hash strings path.
    response = client.delete("/documents/does_not_exist_abc123")
    # Verify response code matches 404 not found.
    assert response.status_code == 404
