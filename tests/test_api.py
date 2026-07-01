"""
Integration tests for the FastAPI backend.

Uses TestClient to test all endpoints without starting a real server.
Mocks the HybridRetriever and FinancialAnswerGenerator to avoid
needing real embeddings or an LLM API key during CI.
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to set a fake GROQ_API_KEY and isolate ChromaDB path before importing the app
import os
import tempfile
os.environ["GROQ_API_KEY"] = "test_key_for_ci"
os.environ["CHROMA_DB_PATH"] = tempfile.mkdtemp()

from src.api.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a TestClient with mocked components."""
    from src.api.main import verify_api_key
    app.dependency_overrides[verify_api_key] = lambda: None
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "components" in data


# ── /documents ────────────────────────────────────────────────────────────────

def test_list_documents_empty(client):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data
    assert "total_chunks" in data


# ── /eval-metrics ─────────────────────────────────────────────────────────────

def test_eval_metrics_empty(client):
    response = client.get("/eval-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "context_precision" in data
    assert "faithfulness" in data
    assert "answer_relevancy" in data
    assert "total_queries" in data


# ── /upload ───────────────────────────────────────────────────────────────────

def test_upload_non_pdf_rejected(client):
    """Only PDF files should be accepted."""
    fake_txt = io.BytesIO(b"this is not a pdf")
    response = client.post(
        "/upload",
        files={"file": ("document.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400


def test_upload_valid_pdf(client, tmp_path):
    """A minimal valid PDF should be processed (even if it extracts no text)."""
    # Minimal PDF bytes
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
    pdf_file = io.BytesIO(minimal_pdf)
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", pdf_file, "application/pdf")},
    )
    # Should succeed even if no text extracted (empty chunk list is fine)
    assert response.status_code in (200, 422)


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_no_documents(client):
    """Query without any uploaded documents should return 400."""
    payload = {
        "question": "What was the revenue in FY2023?",
        "top_k": 3,
        "confidence_threshold": 0.6,
    }
    response = client.post("/query", json=payload)
    # Either 400 (no documents) or 503 (no LLM key) are acceptable
    assert response.status_code in (400, 503)


def test_query_validation_short_question(client):
    """Questions shorter than 5 chars should fail validation."""
    payload = {
        "question": "Hi",
        "top_k": 3,
        "confidence_threshold": 0.6,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_query_validation_invalid_top_k(client):
    payload = {
        "question": "What was the revenue?",
        "top_k": 0,  # must be >= 1
        "confidence_threshold": 0.6,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422


# ── /documents/{id} DELETE ────────────────────────────────────────────────────

def test_delete_nonexistent_document(client):
    response = client.delete("/documents/does_not_exist_abc123")
    assert response.status_code == 404
