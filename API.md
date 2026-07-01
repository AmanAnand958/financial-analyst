# API Reference
## FinDoc Intelligence — FastAPI Backend

**Base URL (local):** `http://localhost:8000`  
**Base URL (Azure):** `https://<app>.azurecontainerapps.io`  
**Interactive Docs:** `GET /docs` (Swagger UI)

---

## Authentication

No authentication required for local development.  
For production, integrate Azure API Management with OAuth2.

---

## Endpoints

### 1. Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "vector_store": "ready",
    "total_chunks": 12500,
    "llm": "ready"
  }
}
```

---

### 2. Upload Document

```
POST /upload
Content-Type: multipart/form-data
```

**Request body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | PDF file | ✅ | Financial PDF (≤ 50 MB) |

**Success Response (200):**
```json
{
  "success": true,
  "document_id": "abc123def456",
  "document_name": "reliance_2023.pdf",
  "chunks_created": 125,
  "total_pages": 500
}
```

**Error Responses:**
| Status | Reason |
|--------|--------|
| 400 | Non-PDF file uploaded |
| 413 | File exceeds 50 MB |
| 422 | PDF text extraction failed |
| 500 | Internal error during indexing |

---

### 3. Query (Main RAG Endpoint)

```
POST /query
Content-Type: application/json
```

**Request body:**
```json
{
  "question": "What was the revenue in FY2023?",
  "top_k": 3,
  "confidence_threshold": 0.6
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Financial question (min 5 chars) |
| `top_k` | int | 3 | Number of source chunks (1-10) |
| `confidence_threshold` | float | 0.6 | Min score to proceed with LLM (0-1) |

**Success Response (200):**
```json
{
  "answer": "The consolidated revenue from operations was ₹2,40,000 Crore in FY2023, as reported in reliance_2023.pdf on page 45.",
  "confidence": "HIGH",
  "retrieval_score": 0.89,
  "source_chunks": [
    {
      "document": "reliance_2023.pdf",
      "page": 45,
      "section": "Income Statement",
      "excerpt": "Consolidated total revenue from operations for the financial year ended March 31, 2023 was ₹2,40,000 Crore…",
      "relevance_score": 0.89
    }
  ],
  "latency_ms": 2345
}
```

**Confidence Values:**
- `HIGH` — retrieval score ≥ threshold, LLM answer generated
- `LOW` — retrieval score < threshold, abstention response returned

**Error Responses:**
| Status | Reason |
|--------|--------|
| 400 | No documents uploaded |
| 422 | Validation error (question too short, invalid top_k) |
| 500 | LLM API failure |
| 503 | LLM not configured (missing GROQ_API_KEY) |

---

### 4. Evaluation Metrics

```
GET /eval-metrics
```

**Response:**
```json
{
  "context_precision": 0.84,
  "faithfulness": 0.91,
  "answer_relevancy": 0.87,
  "hallucination_rate": 0.04,
  "avg_latency_ms": 2100.5,
  "total_queries": 45,
  "last_evaluated": "2025-01-15T10:30:00.000Z",
  "source": "heuristic"
}
```

**Metric Explanations:**

| Metric | Description | Target |
|--------|-------------|--------|
| `context_precision` | Avg relevance score of retrieved chunks | ≥ 0.85 |
| `faithfulness` | Fraction of HIGH-confidence answers | ≥ 0.90 |
| `answer_relevancy` | Fraction of non-abstain answers | ≥ 0.85 |
| `hallucination_rate` | Fraction of LOW-confidence (abstain) answers | ≤ 0.05 |
| `avg_latency_ms` | Mean end-to-end query time | < 5000 |

---

### 5. List Documents

```
GET /documents
```

**Response:**
```json
{
  "total_documents": 3,
  "total_chunks": 12500,
  "documents": [
    {
      "document_id": "abc123def456",
      "document_name": "reliance_2023.pdf",
      "chunk_count": 5200,
      "uploaded_at": "2025-01-15T09:00:00.000Z"
    }
  ]
}
```

---

### 6. Delete Document

```
DELETE /documents/{document_id}
```

**Path Parameters:**
| Parameter | Description |
|-----------|-------------|
| `document_id` | The document_id returned by `/upload` |

**Success Response (200):**
```json
{
  "success": true,
  "document_id": "abc123def456",
  "deleted_chunks": 125
}
```

**Error Responses:**
| Status | Reason |
|--------|--------|
| 404 | Document not found |

---

### 7. Run Full RAGAs Evaluation

```
POST /eval/run-ragas
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_dataset_path` | string | optional | Path to eval JSON file |

Requires `OPENAI_API_KEY` in environment. Uses `data/processed/eval_dataset.json` by default.

**Response:** Same format as `/eval-metrics` with `"source": "ragas"`.

---

## Example: cURL

```bash
# Health check
curl http://localhost:8000/health

# Upload PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@reliance_2023.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue in FY2023?", "top_k": 3}'

# Get metrics
curl http://localhost:8000/eval-metrics

# List documents
curl http://localhost:8000/documents
```

## Example: Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload
with open("reliance_2023.pdf", "rb") as f:
    resp = requests.post(f"{BASE_URL}/upload", files={"file": f})
doc_id = resp.json()["document_id"]

# Query
resp = requests.post(
    f"{BASE_URL}/query",
    json={"question": "What was the revenue in FY2023?"}
)
print(resp.json()["answer"])
print(resp.json()["source_chunks"][0])
```
