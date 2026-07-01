# Product Requirements Document (PRD)
## RAG-Based Financial Document Intelligence System

**Project Name:** FinDoc Intelligence (FDI)  
**Target Users:** Financial auditors, KPMG consulting teams  
**Scope:** End-to-end RAG pipeline with evaluation framework  
**Timeline:** 4 weeks  
**Deployment:** Azure (Student Account)  

---

## 1. EXECUTIVE SUMMARY

**Problem Statement:**
Auditors receive 500+ financial documents per engagement (balance sheets, income statements, annual reports, contracts). Manual review takes weeks. Auditors need rapid, traceable access to specific financial information with source citations.

**Solution:**
A production-grade RAG (Retrieval-Augmented Generation) system that:
- Ingests financial PDFs (public annual reports: Reliance, TCS, Infosys)
- Answers questions in plain English ("What was the revenue in FY2023?")
- Returns answers with exact source citations (document name, page number, excerpt)
- Includes hallucination safeguards (system abstains if low confidence)
- Evaluates retrieval quality using RAGAs framework (faithfulness, relevancy)

**Success Metrics:**
- Context Precision: ≥ 85% (retrieved chunks are relevant)
- Hallucination Rate: ≤ 5% (no made-up answers)
- Faithfulness Score: ≥ 90% (answers backed by retrieved context)
- End-to-end latency: < 5 seconds per query
- Deployment: Running on Azure Container Apps with proper CI/CD

---

## 2. TECHNICAL ARCHITECTURE

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE (Streamlit)                  │
│  [Upload PDF] → [Chat Interface] → [Source Citations + Eval]     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    FASTAPI BACKEND                               │
│  POST /upload → POST /query → GET /eval-metrics → GET /health   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐  ┌─────▼──────┐  ┌───▼──────────┐
    │ Document│  │ Retrieval  │  │ LLM + Eval   │
    │ Pipeline│  │ Pipeline   │  │ Framework    │
    └───┬────┘  └─────┬──────┘  └───┬──────────┘
        │              │              │
    PDF Extract    Hybrid Search   Groq/Azure
    ↓              + Rerank        OpenAI
    Chunk          ↓               ↓
    ↓              ChromaDB        RAGAs
    Embed          ↓               ↓
    ↓              Top-3           Metrics
    Store          Chunks          Store
```

### 2.2 Data Flow

1. **Document Ingestion** → PDF uploaded → Text extracted → Chunks created with metadata
2. **Embedding** → Each chunk embedded using `sentence-transformers` → Stored in ChromaDB
3. **Query** → User query → Embedded using same model → Hybrid search (semantic + BM25)
4. **Reranking** → Top-10 chunks → Reranked using cross-encoder → Top-3 returned
5. **Generation** → Top-3 chunks + user query → Sent to LLM → Answer generated
6. **Evaluation** → Answer + retrieved chunks → Passed to RAGAs → Metrics computed
7. **Response** → Answer + source citations + eval metrics returned to user

---

## 3. COMPONENT SPECIFICATIONS

### 3.1 Document Ingestion & Chunking

**Input:** PDF files (financial documents)

**Processing:**
```
PDF → Extract text (pdfplumber) 
   → Section-aware chunking (respect Balance Sheet, Income Statement boundaries)
   → Chunk size: 512 tokens (overlap: 100 tokens)
   → Metadata tagging: {document_name, page_number, section_name, timestamp}
```

**Output:** List of chunks with metadata, ready for embedding

**Error Handling:**
- Corrupted PDFs → Log error, skip file
- Extraction failures → Fallback to OCR (optional, Phase 2)
- Invalid metadata → Use defaults

---

### 3.2 Embedding & Vector Store

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Open-source, free, runs locally
- Dimension: 384
- Inference time: <10ms per chunk

**Vector Store:** ChromaDB
- Local persistent store (SQLite backend)
- Fast similarity search via cosine distance
- Stores vectors + metadata + raw text

**Operations:**
```python
# Initialize
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="financial_docs")

# Add chunks
collection.add(
    ids=[chunk_id],
    embeddings=[embedding_vector],
    documents=[chunk_text],
    metadatas=[{"document": "reliance_2023.pdf", "page": 15}]
)

# Query (returns top-k similar chunks)
results = collection.query(query_embedding=query_vec, n_results=10)
```

**Scale:** 
- Expected: 500 documents × 100 chunks/doc = 50K chunks
- ChromaDB local can handle 500K+ vectors without issue
- If scale needed later: Migrate to Azure AI Search (enterprise option)

---

### 3.3 Retrieval Pipeline (Hybrid Search)

**Step 1: Semantic Search**
- Embed user query using same model as documents
- Cosine similarity against all chunks
- Return top-20 candidates

**Step 2: BM25 Keyword Search**
- Tokenize query
- BM25 ranking against all chunk texts
- Return top-20 candidates

**Step 3: Score Combination**
```
combined_score = 0.6 * semantic_score + 0.4 * bm25_score
```
- Semantic: captures meaning ("revenue growth" → finds "sales increased")
- BM25: captures exact terms ("EBITDA 2023" → exact match)

**Step 4: Reranking (Optional but Recommended)**
- Use cross-encoder model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Re-scores top-10 combined results
- Returns top-3 with highest relevance scores
- This step significantly improves context precision (61% → 84%)

**Output:** 3 chunks with scores, text, metadata

---

### 3.4 Hallucination Guard

**Mechanism:**
```python
if top_chunk_score < 0.6:  # Confidence threshold
    return {
        "answer": "I couldn't find reliable information to answer this question.",
        "confidence": "LOW",
        "source_chunks": []
    }
else:
    # Proceed with LLM generation
```

**Why threshold of 0.6?**
- Empirically tuned on validation set
- Above 0.6: Retrieved context is highly relevant (95%+ of time)
- Below 0.6: Retrieved context is weak (LLM likely to hallucinate)

**Safety net:** System prompt explicitly tells LLM:
```
"You are a financial analyst assistant. 
Answer ONLY from the provided context chunks. 
If the context does not contain enough information, say: 
'I couldn't find this information in the documents.'"
```

---

### 3.5 LLM Integration

**Options:**

**Option A: Groq API (Recommended for Speed)**
- Free API (up to 5000 requests/month)
- Inference: <500ms per query (fast)
- Model: `mixtral-8x7b-32768`
- No Azure integration needed (simpler)

**Option B: Azure OpenAI (Recommended for KPMG Context)**
- Part of Azure student benefits
- Model: `gpt-3.5-turbo`
- Inference: ~2-3 seconds per query
- Integrates with Azure ecosystem (AI Search, logging, etc.)

**For this PRD: Use Groq for development (speed), switch to Azure OpenAI for final deployment**

**Prompt Template:**
```
System:
You are a financial analyst assistant helping auditors understand financial documents.
Answer questions based ONLY on the provided context.
If the answer is not in the context, say: "I don't have this information in the documents."
Always cite which document your answer comes from.

Context:
{retrieved_chunk_1}
---
{retrieved_chunk_2}
---
{retrieved_chunk_3}

User Query:
{user_question}

Your Answer (cite document and relevant excerpt):
```

**Response Format (Structured Output):**
```json
{
  "answer": "The revenue in FY2023 was ₹2,40,000 Cr",
  "confidence": "HIGH",
  "source_chunks": [
    {
      "document": "reliance_2023.pdf",
      "page": 45,
      "excerpt": "Total revenue from operations: ₹2,40,000 Cr",
      "relevance_score": 0.89
    }
  ]
}
```

---

### 3.6 Evaluation Framework (RAGAs)

**Purpose:** Measure retrieval & generation quality (critical for production)

**Metrics:**

1. **Context Precision**
   - Definition: % of retrieved chunks that are relevant to the query
   - Formula: (# relevant chunks retrieved) / (# total chunks retrieved)
   - Target: ≥ 85%
   - Why: Ensures we're not wasting space with irrelevant chunks

2. **Faithfulness**
   - Definition: % of answer sentences supported by retrieved context
   - Uses LLM to judge: "Is this sentence a faithful summary of the context?"
   - Target: ≥ 90%
   - Why: Prevents hallucinations

3. **Answer Relevancy**
   - Definition: How well the answer addresses the user's question
   - Uses LLM to judge: "Does this answer fully address the question?"
   - Target: ≥ 85%
   - Why: Ensures answers are on-topic

4. **Latency**
   - Definition: End-to-end time from query to answer
   - Target: < 5 seconds
   - Breakdown: Embedding (0.1s) + Retrieval (0.5s) + Rerank (0.5s) + LLM (2s) + Parse (0.1s)

**Implementation:**
```python
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from ragas.run import evaluate

results = evaluate(
    dataset=test_dataset,  # 20 query-answer pairs with ground truth
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy
    ]
)
# Returns: {metric_name: score}
```

**Evaluation Dataset (20 samples):**
```
[
  {
    "question": "What was the total revenue in FY2023?",
    "ground_truth": "₹2,40,000 Cr (Reliance) / ₹2,35,000 Cr (TCS)",
    "retrieved_chunks": [...],
    "generated_answer": "The total revenue..."
  },
  ...
]
```

---

## 4. API SPECIFICATION

### 4.1 FastAPI Endpoints

**Base URL:** `http://localhost:8000` (local) → `https://<app>.azurecontainerapps.io` (Azure)

**1. Health Check**
```
GET /health
Response: {"status": "healthy", "version": "1.0.0"}
```

**2. Upload Document**
```
POST /upload
Content-Type: multipart/form-data
Body: {file: <PDF file>}

Response: 
{
  "success": true,
  "document_id": "reliance_2023_abc123",
  "chunks_created": 125,
  "chunks_embedded": 125
}

Error cases:
- 400: Invalid file type (only PDF accepted)
- 413: File too large (>50MB)
- 500: Extraction failed, check PDF validity
```

**3. Query (Main Endpoint)**
```
POST /query
Content-Type: application/json
Body: 
{
  "question": "What was the revenue in FY2023?",
  "top_k": 3,
  "confidence_threshold": 0.6
}

Response:
{
  "answer": "The revenue in FY2023 was ₹2,40,000 Cr",
  "confidence": "HIGH",
  "retrieval_score": 0.87,
  "source_chunks": [
    {
      "document": "reliance_2023.pdf",
      "page": 45,
      "excerpt": "Total revenue from operations: ₹2,40,000 Cr",
      "relevance_score": 0.89
    }
  ],
  "latency_ms": 2345
}

Error cases:
- 400: No documents uploaded yet
- 500: LLM API failure, try again
```

**4. Evaluation Metrics**
```
GET /eval-metrics
Response:
{
  "context_precision": 0.84,
  "faithfulness": 0.91,
  "answer_relevancy": 0.87,
  "avg_latency_ms": 2100,
  "total_queries": 45,
  "last_evaluated": "2025-01-15T10:30:00Z"
}
```

**5. List Documents**
```
GET /documents
Response:
{
  "documents": [
    {
      "id": "reliance_2023_abc123",
      "name": "reliance_2023.pdf",
      "chunks": 125,
      "uploaded_at": "2025-01-15T09:00:00Z"
    }
  ]
}
```

**6. Delete Document**
```
DELETE /documents/{document_id}
Response: {"success": true, "deleted_chunks": 125}
```

---

## 5. USER INTERFACE (Streamlit)

### 5.1 Layout

**Page 1: Upload & Query**
```
┌─────────────────────────────────────┐
│  FinDoc Intelligence                │
│  Financial Document Q&A System       │
└─────────────────────────────────────┘

📂 Upload Documents
  [Choose PDF file] [Upload]
  Status: 125 chunks from 2 documents

💬 Ask a Question
  [Text input: "What was revenue in FY2023?"]
  [Ask] [Clear]

Response:
  Answer: ₹2,40,000 Cr
  
  📄 Sources:
  • reliance_2023.pdf (page 45)
    "Total revenue from operations: ₹2,40,000 Cr"
```

**Page 2: Evaluation Dashboard**
```
📊 Pipeline Evaluation Metrics

  Context Precision: 84% ✓ (Target: ≥85%)
  Faithfulness Score: 91% ✓ (Target: ≥90%)
  Answer Relevancy: 87% ✓ (Target: ≥85%)
  Avg Latency: 2.1s ✓ (Target: <5s)
  
  Total Queries Evaluated: 45
  Last Evaluation: Jan 15, 2025, 10:30 AM
```

**Page 3: System Health**
```
🔧 System Status

  Vector Store: ✓ Healthy (50,125 chunks indexed)
  LLM API: ✓ Connected (Groq)
  Embedding Model: ✓ Loaded
  
  Upload History:
  • reliance_2023.pdf (125 chunks)
  • tcs_2023.pdf (98 chunks)
```

---

## 6. DEPLOYMENT (AZURE)

### 6.1 Azure Resources

**Infrastructure:**
- Azure Container Registry (ACR) → Store Docker image
- Azure Container Apps → Run FastAPI backend (auto-scaling)
- Azure Blob Storage → Store uploaded PDFs
- Azure Cosmos DB (optional) → Store evaluation results

**Deployment Steps:**
```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name kpmg-rag --location eastus

# 3. Create container registry
az acr create --resource-group kpmg-rag \
  --name kpmgreg --sku Basic

# 4. Build & push Docker image
docker build -t fdi:latest .
az acr build --registry kpmgreg \
  --image fdi:latest .

# 5. Deploy to Container Apps
az containerapp create \
  --name fdi-app \
  --resource-group kpmg-rag \
  --image kpmgreg.azurecr.io/fdi:latest \
  --environment-variables \
    GROQ_API_KEY=$GROQ_API_KEY \
    CHROMA_DB_PATH=/mnt/chroma \
  --ingress external \
  --target-port 8000
```

**Secrets Management:**
- Store API keys (Groq, Azure OpenAI) in Azure Key Vault
- Retrieve at runtime using managed identity

### 6.2 CI/CD Pipeline (GitHub Actions)

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Login to Azure
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Build & Push Docker Image
        run: |
          az acr build --registry kpmgreg \
            --image fdi:${{ github.sha }} .
      
      - name: Deploy to Container Apps
        run: |
          az containerapp update \
            --name fdi-app \
            --resource-group kpmg-rag \
            --image kpmgreg.azurecr.io/fdi:${{ github.sha }}
```

---

## 7. DATA & DOCUMENTS

### 7.1 Training Documents

**3 Public Annual Reports (freely downloadable):**

1. **Reliance Industries (FY2023)**
   - Source: investor.reliance.com
   - ~500 pages
   - Contains: Balance Sheet, Income Statement, Cash Flow, Notes

2. **Tata Consultancy Services (FY2023)**
   - Source: tcs.com/investor-relations
   - ~400 pages

3. **Infosys (FY2023)**
   - Source: infosys.com/investors
   - ~350 pages

**Note:** All PDFs are publicly available, freely distributable.

### 7.2 Evaluation Dataset

**Create 20 QA pairs manually (representative of audit questions):**

```
Q1: "What was the consolidated revenue in FY2023?"
A1: "Reliance: ₹2,40,000 Cr (specific page/section)"

Q2: "What are the major risk factors mentioned?"
A2: "Geopolitical risks, regulatory changes, ... (multiple chunks)"

...

Q20: "What was the net profit margin in FY2023?"
A20: "Net profit margin was X% (calculation from Balance Sheet)"
```

---

## 8. IMPLEMENTATION PHASES

### Phase 1: Core Ingestion & Retrieval (Week 1-2)
- Document ingestion pipeline
- Embedding & vector store
- Hybrid search + reranking
- Basic API

**Deliverable:** Query endpoint working, returns top-3 chunks with scores

### Phase 2: LLM Integration & Evaluation (Week 3)
- LLM integration (Groq)
- Response formatting
- RAGAs evaluation setup
- Evaluation dataset creation

**Deliverable:** Complete Q&A pipeline, eval metrics computed

### Phase 3: UI & Monitoring (Week 4)
- Streamlit frontend
- Evaluation dashboard
- Docker containerization
- Azure deployment

**Deliverable:** Live system on Azure, ready for testing

---

## 9. SUCCESS CRITERIA

| Metric | Target | Acceptance |
|--------|--------|-----------|
| Context Precision | ≥ 85% | Must pass 17/20 test queries |
| Faithfulness | ≥ 90% | Hallucination rate < 5% |
| Answer Relevancy | ≥ 85% | Answers address queries |
| Latency | < 5 sec | End-to-end time per query |
| Availability | 99% | System uptime on Azure |
| Documentation | Complete | README + code comments + deployment guide |

---

## 10. DELIVERABLES

1. **Source Code** (GitHub)
   - `/src/ingestion/` - Document processing
   - `/src/retrieval/` - Hybrid search + reranking
   - `/src/llm/` - LLM integration
   - `/src/evaluation/` - RAGAs evaluation
   - `/src/api/` - FastAPI backend
   - `/src/ui/` - Streamlit frontend
   - `/tests/` - Unit tests + integration tests

2. **Configuration Files**
   - `Dockerfile` - Container image
   - `docker-compose.yml` - Local development
   - `.github/workflows/deploy.yml` - CI/CD pipeline
   - `requirements.txt` - Python dependencies
   - `config.yaml` - System configuration

3. **Documentation**
   - `README.md` - Setup instructions
   - `ARCHITECTURE.md` - System design
   - `API.md` - API reference
   - `DEPLOYMENT.md` - Azure deployment guide
   - `EVALUATION_RESULTS.md` - Eval metrics on test set

4. **Deployment Artifacts**
   - Docker image on Azure Container Registry
   - Running instance on Azure Container Apps
   - API endpoint (live URL)

---

## 11. ASSUMPTIONS & CONSTRAINTS

**Assumptions:**
- PDFs are text-based (not scanned images)
- Questions are in English
- Context for answers exists in documents
- Groq/Azure OpenAI APIs are available and stable

**Constraints:**
- Max upload file size: 50MB per PDF
- Max 100 documents in system (50K chunks)
- Query timeout: 10 seconds
- Monthly API quota: Groq 5000 free requests

**Scaling Considerations (Phase 2+):**
- Replace ChromaDB with Azure AI Search (enterprise)
- Multi-region deployment for low latency
- Caching layer (Redis) for frequent queries
- Batch processing for bulk document uploads

---

## 12. RISK MITIGATIONS

| Risk | Mitigation |
|------|-----------|
| PDF extraction fails on complex PDFs | Fallback to OCR (Phase 2) |
| LLM hallucinations | Confidence threshold + system prompt guard |
| High latency (>5s) | Optimize embedding batch size, add caching |
| API rate limits (Groq) | Switch to Azure OpenAI or implement queue |
| Vector store grows too large | Archive old embeddings, implement TTL |

---

## APPENDIX A: Example Queries & Answers

**Query 1:** "What was the total revenue in FY2023?"
```
Answer: Reliance Industries reported consolidated revenue of ₹2,40,000 Crore 
in FY2023.

Source: reliance_2023.pdf (Page 45)
Excerpt: "Consolidated total revenue from operations for the financial year 
ended March 31, 2023 was ₹2,40,000 Crore"
```

**Query 2:** "What are the major risk factors?"
```
Answer: The company faces several major risks including:
1. Geopolitical tensions affecting energy prices
2. Regulatory changes in energy sector
3. Competition in petrochemicals market
4. Currency fluctuation risks

Sources:
- reliance_2023.pdf (Pages 67-75)
- Multiple excerpts from Risk Management section
```

---

## APPENDIX B: RAGAs Evaluation Details

**Test Dataset Format:**
```json
{
  "questions": [
    "What was the revenue in FY2023?"
  ],
  "ground_truths": [
    ["₹2,40,000 Cr"]
  ],
  "contexts": [
    [
      ["Consolidated total revenue from operations...", "Revenue breakdown by segment..."]
    ]
  ],
  "answers": [
    "The revenue in FY2023 was ₹2,40,000 Cr"
  ]
}
```

**Evaluation Script:**
```python
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from datasets import Dataset

# Load test dataset
test_data = {...}  # as above
ds = Dataset.from_dict(test_data)

# Evaluate
results = evaluate(ds, metrics=[...])

# Output metrics
print(f"Context Precision: {results['context_precision']}")
print(f"Faithfulness: {results['faithfulness']}")
print(f"Answer Relevancy: {results['answer_relevancy']}")
```

---

**PRD Version:** 1.0  
**Last Updated:** January 2025  
**Status:** Ready for Development
