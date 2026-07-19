# FinDoc Intelligence — Interview Questions & Answers

> **How to use this file:** Read through each section before your interview. The questions are grouped by topic and ordered from basic → advanced within each section. Practice answering aloud — don't memorize word-for-word, understand the *why* behind each answer.

---

## Table of Contents

1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [RAG Architecture & Design Decisions](#2-rag-architecture--design-decisions)
3. [Document Ingestion Pipeline](#3-document-ingestion-pipeline)
4. [Retrieval & Hybrid Search](#4-retrieval--hybrid-search)
5. [LLM Integration & Generation](#5-llm-integration--generation)
6. [Evaluation Framework (RAGAs)](#6-evaluation-framework-ragas)
7. [API Design (FastAPI)](#7-api-design-fastapi)
8. [Frontend (Streamlit UI)](#8-frontend-streamlit-ui)
9. [DevOps, Docker & Deployment](#9-devops-docker--deployment)
10. [Testing Strategy](#10-testing-strategy)
11. [System Design & Scalability](#11-system-design--scalability)
12. [Security & Production Readiness](#12-security--production-readiness)
13. [Machine Learning Concepts](#13-machine-learning-concepts)
14. [HR / Behavioral Questions](#14-hr--behavioral-questions)

---

## 1. Project Overview & Motivation

### Q1. Tell me about your project. What problem does it solve?

**Answer:**

FinDoc Intelligence (FDI) is a production-grade **Retrieval-Augmented Generation (RAG)** system built for financial auditors — specifically designed for teams like KPMG. Auditors typically deal with 500+ financial documents per engagement (balance sheets, annual reports, income statements). Manually reviewing them takes weeks.

My system lets auditors upload financial PDFs and ask plain-English questions like "What was the consolidated revenue in FY2023?" The system returns an answer with **exact source citations** (document name, page number, excerpt), and has built-in **hallucination safeguards** — if the system isn't confident, it abstains from answering rather than making something up.

---

### Q2. Why did you choose RAG over fine-tuning an LLM?

**Answer:**

Three key reasons:

1. **Data freshness** — Financial reports change every quarter. Fine-tuning would require retraining the model each time. With RAG, I just ingest the new PDF and the system can immediately answer questions about it.
2. **Traceability** — Auditors need source citations. RAG naturally provides this because every answer is grounded in specific retrieved chunks with page numbers. A fine-tuned model cannot point to where it learned something.
3. **Cost** — Fine-tuning a 70B parameter model is extremely expensive. RAG uses the LLM as-is (via API) and only adds a retrieval layer, which is much cheaper to maintain.
4. **Hallucination control** — With RAG, I can measure retrieval confidence and refuse to answer if the context is insufficient. Fine-tuned models hallucinate silently.

---

### Q3. Who are the target users and how would they use this system?

**Answer:**

The primary target users are **financial auditors and analysts** at firms like KPMG. They would:

1. **Upload** annual reports / financial PDFs through the Streamlit UI or API.
2. **Query** the system in natural language — e.g., "What were the risk factors in the FY2023 report?"
3. **Review** the answer along with source citations (document, page, excerpt) to verify accuracy.
4. **Evaluate** the pipeline's quality using the built-in RAGAs evaluation dashboard.

The system saves them hours of manual searching through hundreds of pages.

---

### Q4. What are the success metrics for this project?

**Answer:**

| Metric | Target | Achieved |
|--------|--------|----------|
| Context Precision@3 | ≥ 60% | 62.6% (Strict numeric-semantic verification) |
| Hallucination Rate | 0.0% | 0.0% (Reduced from 13.0% under baseline) |
| System Abstention Rate | - | 39.1% (Strict thresholding safety fallback) |
| End-to-end Latency | < 5 seconds | 1.99s avg (Groq Llama-3.3 + local models) |

These metrics were evaluated against an expanded test suite of 66 ground-truth financial QA pairs on our multi-page mock annual report.

---

### Q5. What is the tech stack of your project?

**Answer:**

- **Backend:** FastAPI (Python 3.11)
- **Frontend:** Streamlit with Plotly charts
- **Vector DB:** ChromaDB (persistent, file-based)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Keyword Search:** BM25 via `rank-bm25`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L6-v2`
- **LLM:** Groq API with `llama-3.3-70b-versatile`
- **PDF Parsing:** `pdfplumber` + `PyMuPDF` + `pytesseract` (OCR fallback)
- **Evaluation:** RAGAs framework + custom heuristic metrics
- **Deployment:** Docker + Docker Compose + Azure Container Apps + GitHub Actions CI/CD
- **Logging:** Loguru

---

## 2. RAG Architecture & Design Decisions

### Q6. Explain the end-to-end architecture of your RAG system.

**Answer:**

The system has four main pipelines:

1. **Ingestion Pipeline:** PDF → text extraction (pdfplumber, OCR fallback) → table detection → section-aware sliding-window chunking (512 tokens, 100 overlap) → embed using sentence-transformers → store in ChromaDB + in-memory BM25 index.

2. **Retrieval Pipeline:** User query → embed query → semantic search (top-20 from ChromaDB) + BM25 keyword search (top-20) → score fusion (60% semantic + 40% BM25) → cross-encoder reranking → return top-3 chunks.

3. **Generation Pipeline:** Top-3 chunks + user question → check confidence threshold (≥ 0.6) → if passes, send to Groq LLM with structured prompt → parse response with source citations.

4. **Evaluation Pipeline:** Record heuristic metrics per query (latency, context precision proxy, faithfulness proxy) + optional full RAGAs evaluation using the 20-pair ground-truth dataset.

---

### Q7. Why did you choose ChromaDB over Pinecone / Weaviate / FAISS?

**Answer:**

- **Simplicity:** ChromaDB is file-based and requires no external server or cloud account. Perfect for a project that needs to run locally and in containers.
- **Python-native:** It integrates seamlessly with Python — no REST API overhead.
- **Persistence:** It supports persistent storage to disk, so data survives restarts.
- **Metadata filtering:** ChromaDB supports metadata queries, which I use to filter by document ID when deleting documents.
- **Cost:** Free and open-source. Pinecone requires a paid plan for production workloads. FAISS doesn't provide persistence or metadata filtering out of the box.

If this were a true production system with millions of documents, I'd consider migrating to Pinecone or Weaviate for scalability.

---

### Q8. Why did you use Groq instead of OpenAI?

**Answer:**

- **Speed:** Groq uses custom LPU (Language Processing Unit) hardware that delivers inference speeds 10-18x faster than GPU-based alternatives. For a financial Q&A system where latency matters, this is a huge advantage — my average query latency is 0.30 seconds.
- **Cost:** Groq offers a free tier with generous rate limits, making it ideal for development and demos.
- **Model quality:** Groq supports `llama-3.3-70b-versatile`, which is competitive with GPT-4 for structured Q&A tasks, especially with good retrieval context.
- **Fallback design:** My architecture is provider-agnostic — the system can switch to Azure OpenAI by changing config, since the LLM interaction is abstracted in `generator.py`.

---

### Q9. What is the data flow when a user asks a question?

**Answer:**

```
User Question → Check Query Cache → [Cache Hit] → Return cached response
                      ↓
                [Cache Miss]
                      ↓
         Embed query using sentence-transformers
                      ↓
         Semantic search (ChromaDB, top-20) + BM25 search (in-memory, top-20)
                      ↓
         Score fusion: combined = 0.6 × semantic + 0.4 × BM25
                      ↓
         Cross-encoder reranking (top-20 → top-3)
                      ↓
         Confidence check: Is top score ≥ 0.6?
              ↓ No                    ↓ Yes
       Return "LOW CONFIDENCE"   Build prompt with context
       abstain response           → Send to Groq LLM
                                  → Parse response
                                  → Cache result
                                  → Record eval metrics
                                  → Return answer + citations
```

---

### Q10. How does the caching layer work?

**Answer:**

I implemented a **thread-safe, in-memory query cache** using Python's `threading.Lock`:

- When a query comes in, the system first checks the cache using the query string as the key.
- On a cache hit, the cached response is returned immediately (0ms LLM cost).
- On a cache miss, the full RAG pipeline runs, and the result is stored in the cache.
- **Cache invalidation:** When a document is uploaded or deleted, the entire cache is cleared because the knowledge base has changed and cached answers may be stale.

This is a simple but effective strategy for a single-node deployment. For distributed systems, I'd use Redis.

---

## 3. Document Ingestion Pipeline

### Q11. How do you extract text from PDFs?

**Answer:**

I use a **multi-strategy approach:**

1. **Primary: `pdfplumber`** — Extracts text page-by-page. It handles most text-based PDFs well and preserves layout.
2. **OCR Fallback: `PyMuPDF` + `pytesseract`** — If a page returns fewer than 50 characters of text (likely a scanned image), the system renders the page as a PNG using PyMuPDF and runs OCR using pytesseract.
3. **Table Extraction:** pdfplumber can detect tables. When tables are found, I convert them to **Markdown table format** before chunking. This preserves row/column relationships that would otherwise be lost in plain text.

---

### Q12. What is your chunking strategy and why?

**Answer:**

I use **section-aware, sliding-window, token-based chunking:**

- **Token-based (not character-based):** I split on whitespace to count tokens. This gives better semantic units than fixed character counts.
- **Chunk size: 512 tokens, Overlap: 100 tokens:** This ensures enough context per chunk for the embedding model, while the overlap prevents information loss at chunk boundaries.
- **Section-aware:** I detect financial section headers (Balance Sheet, Income Statement, Cash Flow, etc.) using a predefined keyword list. When a new section starts, I break the chunk there to avoid mixing content from different financial sections.
- **Cross-page:** Chunks can span across pages. I track the page range in metadata so citations remain accurate.

**Why sliding window over recursive splitting?**  
Financial documents have specific structure. Recursive splitting (like LangChain's) doesn't respect section boundaries and can produce chunks that mix balance sheet data with income statement data, confusing the retriever.

---

### Q13. How do you handle metadata for each chunk?

**Answer:**

Each chunk carries rich metadata:

- `document_id` — MD5 hash of the filename for unique identification.
- `document_name` — Original filename.
- `page_number` / `page_range` — Which page(s) the chunk spans.
- `section_name` — Detected financial section (e.g., "balance sheet").
- `chunk_index` — Position in the overall document.
- `timestamp` — When the document was processed.

This metadata is stored alongside the embeddings in ChromaDB and is used for source citations in the final answer.

---

### Q14. How do you handle corrupted or malformed PDFs?

**Answer:**

Multiple layers of error handling:

1. **File validation:** Check file extension (`.pdf` only) and file size (max 50MB configurable).
2. **Try-except around extraction:** If `pdfplumber` crashes on a page, the error is logged and that page is skipped — the rest of the document still gets processed.
3. **OCR fallback:** If text extraction returns near-empty content, OCR is attempted.
4. **Empty document check:** If zero chunks are produced after processing, the system returns an appropriate error rather than storing an empty document.

---

### Q15. What is the OCR fallback mechanism?

**Answer:**

When pdfplumber extracts fewer than 50 characters from a page (indicating it's likely a scanned image):

1. **Render page to image:** Using `PyMuPDF` (fitz), the page is rendered at 300 DPI as a PNG pixmap.
2. **Convert to PIL Image:** The pixmap bytes are loaded into a PIL Image object using `io.BytesIO`.
3. **Run OCR:** `pytesseract.image_to_string()` extracts text from the image.
4. **Merge results:** The OCR text replaces the empty pdfplumber output for that page.

This ensures scanned financial documents are also searchable.

---

## 4. Retrieval & Hybrid Search

### Q16. Why hybrid search instead of just semantic search?

**Answer:**

Pure semantic search has blind spots, especially for financial documents:

- **Exact terms matter:** If an auditor searches for "EBITDA margin Q3 FY2023", semantic search might return chunks about revenue or profit margins because they're semantically similar. BM25 ensures exact keyword matches rank high.
- **Numerical precision:** Semantic embeddings don't encode exact numbers well. BM25 gives an exact match advantage for queries with specific figures or codes.
- **Research-backed:** Multiple papers (e.g., from Microsoft's Bing team) show that hybrid search consistently outperforms either method alone for information retrieval.

The 60/40 weighting gives semantic search a slight edge (for understanding intent) while keeping BM25's precision for exact matches.

---

### Q17. Explain how score fusion works in your system.

**Answer:**

After both search methods return their top-20 candidates:

1. **Semantic scores:** ChromaDB returns cosine similarity distances. I convert them to similarity scores (1 - distance).
2. **BM25 scores:** Raw BM25 scores can be any positive number. I **normalize** them by dividing by the maximum score (capped at 10.0 to handle outliers), bringing them into the [0, 1] range.
3. **Fusion formula:** `combined_score = 0.6 × semantic_score + 0.4 × bm25_score`
4. **Deduplication:** If the same chunk appears in both result sets, I take the higher individual scores before fusion.
5. **Sort by combined score** and pass the top candidates to the reranker.

---

### Q18. What is a cross-encoder reranker and why do you use it?

**Answer:**

A **cross-encoder** is a transformer model that takes a (query, passage) pair as input and outputs a single relevance score. Unlike bi-encoders (sentence-transformers) that encode query and passage independently:

- **Cross-encoders see both together** — they can model fine-grained token interactions between the query and passage.
- **Much more accurate** but also slower (can't pre-compute embeddings).

I use `cross-encoder/ms-marco-MiniLM-L6-v2` as the reranker. It takes the top-20 fused candidates (obtained from score fusion of 60% semantic similarity and 40% BM25 keyword matching) and reranks them based on joint query-passage token-level attention. This achieved a **62.6% Context Precision@3 under our strict programmatic verification**, ensuring that exact financial data or dense footnotes are ranked first.

The trade-off is speed: reranking 20 candidates adds an overhead of **~232.6ms**. However, with a highly optimized pipeline and Groq's fast inference, the total query latency remains well under the 5-second SLA, averaging **~1.99s**.

---

### Q19. How does the BM25 index stay in sync with ChromaDB?

**Answer:**

The BM25 index is **in-memory** (not persisted to disk). To keep it in sync:

1. **On document upload:** After chunks are stored in ChromaDB, the same chunk texts are added to the BM25 index. I maintain parallel lists of chunk IDs and tokenized texts.
2. **On document deletion:** Chunks are removed from both ChromaDB (by metadata filter) and the BM25 index (by matching IDs). The BM25 index is then rebuilt from the remaining texts.
3. **On startup:** The `_rebuild_bm25_from_chroma()` method queries all documents from ChromaDB and rebuilds the BM25 index. This ensures the BM25 index survives server restarts.
4. **Thread safety:** All BM25 mutations are protected by a `threading.Lock` to prevent race conditions from concurrent uploads.

---

### Q20. Why top-20 candidates initially and then top-3 finally?

**Answer:**

- **Top-20 initial candidates** gives the retriever a wide enough pool to capture relevant chunks that might rank lower individually in one search method but high in the other. It's the "recall" stage — we want to make sure we don't miss anything relevant.
- **Top-3 final chunks** is the "precision" stage. We send only the most relevant chunks to the LLM because:
  - LLMs have context window limits.
  - More irrelevant context = more hallucination risk.
  - 3 chunks of 512 tokens each = ~1,536 tokens of context, which is optimal for the LLM to synthesize a focused answer.

---

### Q21. What is the embedding model and why did you choose it?

**Answer:**

I use **`all-MiniLM-L6-v2`** from sentence-transformers:

- **384-dimensional** embeddings (compact, fast).
- **Fast inference:** ~14,000 sentences/sec on GPU, still fast on CPU.
- **Good quality:** Trained on 1B+ sentence pairs, ranks well on MTEB benchmarks for its size.
- **Small footprint:** ~80MB model, easy to deploy in containers.
- **Cosine similarity optimized:** Trained specifically for semantic similarity tasks.

For a production system with larger budgets, I'd consider `all-mpnet-base-v2` (768-dim, higher quality) or OpenAI's `text-embedding-3-small`.

---

## 5. LLM Integration & Generation

The hallucination guarding mechanism uses a two-tiered defense: **Retrieval relevance thresholding combined with Deterministic Numeric Citation-Verification (NCV)**.

Here is exactly how this two-tiered guard works:

1. **Retrieval-Score Thresholding:** After retrieval and reranking, we check the cross-encoder relevance score of the top chunk. If it is **below 0.6**, the system immediately abstains from calling the LLM and returns a safe response.
2. **Deterministic Numeric Citation-Verification (NCV):** If the retrieval score is high enough, the LLM generates an answer. Before returning this answer to the auditor, the NCV guard parses all significant numbers (floats, and integers with a length of 3 or more, such as "45,000", "2,40,000", "14.28") from the generated response. It normalizes formatting (removing currency symbols like ₹/$, commas, and terms like "crore") and converts these figures to float values. It then extracts all numerical figures from the retrieved context chunks and checks if every number in the LLM's answer is present as a float value in the context.
3. **Safe Abstention:** If a number in the generated response cannot be verified in the source context, the NCV guard aborts the response to prevent a numeric hallucination. It overrides the output with a safe, standardized message: *"I couldn't find reliable information to answer this question in the provided documents. Please try rephrasing your query or upload additional relevant documents."* and sets `"confidence": "LOW"`.

**Quantifiable Impact:** In testing against our 66-query evaluation suite, a raw LLM baseline had a **12.5% hallucination rate** (generating incorrect figures when retrieval returned irrelevant context or when answering outside documents). Applying this guard successfully reduced the hallucination rate to **0.0%**, shifting the failures to safe, audit-compliant abstentions.

---

### Q23. Explain the prompt engineering in your system.

**Answer:**

I use a **two-part prompt structure:**

**System Prompt** — Sets the persona and rules:

- "You are a financial analyst assistant helping KPMG auditors"
- Rule 1: Answer ONLY from provided context chunks
- Rule 2: If the answer isn't in context, say "I don't have this information"
- Rule 3: Always cite document name and page number
- Rule 4: Be concise, factual, professional
- Rule 5: Never make up numbers
- Rule 6: Distinguish different sources if they conflict

**User Prompt Template** — Structured format:
```
Context:
[chunk 1 with metadata]
[chunk 2 with metadata]
[chunk 3 with metadata]

Question: {user_question}

Provide a clear, factual answer citing the source document(s) and page number(s):
```

**Why this works:** The system prompt constrains the model's behavior, and the structured user prompt makes it easy for the model to locate and cite information.

---

### Q24. Why is the temperature set to 0.1?

**Answer:**

Temperature controls the randomness of LLM outputs:

- **0.0** = deterministic, always picks the highest probability token
- **1.0** = maximum randomness

I set **0.1** because:

- Financial Q&A requires **factual precision** — we don't want creative or varied answers.
- A small non-zero value (0.1 vs 0.0) allows slight variation to avoid repetitive phrasing, but keeps responses highly focused.
- For tasks like creative writing, temperature 0.7-0.9 is typical. For factual extraction, 0.0-0.2 is standard.

---

### Q25. How do you handle the context window limitation of the LLM?

**Answer:**

Several strategies:

1. **Limited chunks:** Only 3 chunks (each ~512 tokens) are sent to the LLM, totaling ~1,536 tokens of context — well within the 128K context window of `llama-3.3-70b-versatile`.
2. **Relevance filtering:** The cross-encoder reranker ensures only the most relevant chunks make it to the LLM.
3. **Max tokens cap:** I limit the generated response to 1,024 tokens to prevent verbose outputs.
4. **Structured prompt:** The prompt template is concise, adding minimal token overhead.

Even with system prompt + context + question + response, the total is under 4,000 tokens — well under the limit.

---

### Q26. What happens if the Groq API is down?

**Answer:**

Currently, the system returns an HTTP 500 error with a descriptive message. The error is caught in the `/query` endpoint's try-except block, logged via Loguru, and returned as a JSON error response.

For production improvements, I would add:

1. **Retry logic** with exponential backoff (3 retries with 1s, 2s, 4s delays).
2. **Fallback provider** — Switch to Azure OpenAI or a local model.
3. **Circuit breaker pattern** — After N consecutive failures, stop calling the API for a cooldown period.
4. **Graceful degradation** — Return retrieved chunks without LLM-generated answer, so the user still gets relevant context.

---

## 6. Evaluation Framework (RAGAs)

### Q27. What is RAGAs and why did you use it?

**Answer:**

**RAGAs** (Retrieval-Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines. I used it because:

1. **Standard metrics:** It provides well-defined metrics specifically designed for RAG:
   - **Context Precision:** Are the retrieved chunks actually relevant to the query?
   - **Faithfulness:** Is the answer supported by the retrieved context (not hallucinated)?
   - **Answer Relevancy:** Does the answer actually address the question asked?

2. **Automated evaluation:** Unlike human evaluation, RAGAs can evaluate hundreds of queries automatically.

3. **Industry standard:** RAGAs is widely recognized in the AI/ML community for RAG evaluation, making my results comparable to other systems.

---

### Q28. Explain each RAGAs metric in detail.

**Answer:**

| Metric | What it measures | How it's calculated | Target |
|--------|-----------------|---------------------|--------|
| **Context Precision** | Are retrieved chunks relevant? | Uses an LLM to judge if each retrieved chunk contains info needed to answer the question. Ratio of relevant chunks / total chunks. | ≥ 85% |
| **Faithfulness** | Is the answer grounded in context? | Breaks the answer into individual claims/statements. Uses an LLM to verify each claim against the retrieved context. Ratio of supported claims / total claims. | ≥ 90% |
| **Answer Relevancy** | Does the answer address the question? | Generates N hypothetical questions from the answer, then measures semantic similarity between these questions and the original question. | ≥ 85% |

---

### Q29. How does your heuristic evaluation work (without RAGAs)?

**Answer:**

For every query, I record **lightweight heuristic metrics** that don't require additional LLM calls:

1. **Latency:** `time.time()` before and after the full pipeline.
2. **Context Precision proxy:** I use the cross-encoder reranking score of the top chunk. If the reranker gives a high score, the context is likely precise.
3. **Faithfulness proxy:** I use the confidence level from the hallucination guard. If the system didn't abstain, the answer is likely faithful.
4. **Query metadata:** Question, answer, number of retrieved chunks, source documents — all logged to a JSON file.

This lets me track system performance in real-time without the cost of running full RAGAs evaluation on every query.

---

### Q30. What is the evaluation dataset and how did you create it?

**Answer:**

The evaluation dataset is **20 ground-truth financial QA pairs** stored in `data/processed/eval_dataset.json`. Each entry has:

- A financial question
- The expected ground-truth answer
- The source document context

These pairs were created from publicly available annual reports (Reliance, TCS, Infosys FY2023) by:

1. Reading the actual financial documents.
2. Formulating questions that auditors would typically ask.
3. Writing ground-truth answers with exact figures from the documents.

During RAGAs evaluation, the system dynamically generates answers using the RAG pipeline and compares them against these ground-truth pairs.

---

## 7. API Design (FastAPI)

### Q31. What are the API endpoints and their purpose?

**Answer:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check — returns system status, component health, document count |
| `/upload` | POST | Upload and process a financial PDF — returns chunk count and document ID |
| `/query` | POST | RAG query — returns answer, source citations, confidence, and eval metrics |
| `/eval-metrics` | GET | Returns aggregated evaluation metrics (avg precision, faithfulness, etc.) |
| `/documents` | GET | List all ingested documents with metadata |
| `/documents/{id}` | DELETE | Delete a document and all its chunks from ChromaDB and BM25 |

---

### Q32. How do you handle concurrent requests?

**Answer:**

Multiple mechanisms:

1. **Thread-safe BM25 index:** All BM25 mutations (add/remove documents) are protected by a `threading.Lock` to prevent race conditions.
2. **Thread-safe cache:** The query cache uses its own `threading.Lock` for read/write operations.
3. **FastAPI async:** FastAPI supports async request handling. I/O-bound operations (like Groq API calls) don't block other requests.
4. **Rate limiting:** I implemented a sliding-window rate limiter using `collections.deque` that limits requests per IP to prevent abuse.
5. **Uvicorn workers:** In Docker, I run with `--workers 2` for parallel request processing.

---

### Q33. Why FastAPI over Flask or Django?

**Answer:**

- **Performance:** FastAPI is one of the fastest Python web frameworks, built on Starlette and Uvicorn (ASGI).
- **Automatic docs:** Swagger/OpenAPI docs are generated automatically at `/docs` — huge time saver.
- **Type validation:** Pydantic models validate request/response schemas automatically. Invalid inputs get clear error messages.
- **Async support:** Native async/await support for non-blocking I/O.
- **Modern Python:** Uses type hints throughout, making the code self-documenting.

Flask would work but requires manual validation. Django is too heavy for an API-only service.

---

### Q34. How does the lifespan event work in your FastAPI app?

**Answer:**

I use FastAPI's `@asynccontextmanager` lifespan pattern:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Initialize all components
    global processor, retriever, generator, evaluator
    processor = DocumentProcessor(...)
    retriever = HybridRetriever(...)
    generator = FinancialAnswerGenerator(...)
    evaluator = RAGAsEvaluator(...)
    
    yield  # Application runs here
    
    # SHUTDOWN: Cleanup resources
    logger.info("Shutting down...")
```

This ensures all heavy components (ML models, DB connections) are loaded **once** at startup and shared across all requests as singletons, rather than being created per-request.

---

### Q35. How do you handle file uploads securely?

**Answer:**

1. **File type validation:** Only `.pdf` extensions are accepted. The MIME type is also checked.
2. **File size limit:** Configurable max size (50MB default) prevents memory exhaustion.
3. **Temporary storage:** Uploaded files are saved to a temporary directory using `tempfile.NamedTemporaryFile`, processed, and then the temp file is cleaned up.
4. **Filename sanitization:** The original filename is preserved for metadata, but the system uses generated document IDs (MD5 hashes) internally.
5. **Error handling:** If processing fails partway, any partially stored chunks are cleaned up.

---

## 8. Frontend (Streamlit UI)

### Q36. Describe the Streamlit UI and its features.

**Answer:**

The Streamlit UI has multiple tabs:

1. **Upload Tab:** Drag-and-drop PDF upload with progress indicator. Shows processing status and chunk count.
2. **Chat Interface:** Natural language query input. Displays the answer with expandable source citation cards showing document name, page number, and relevant excerpt.
3. **Evaluation Dashboard:** Plotly charts showing Context Precision, Faithfulness, Answer Relevancy, and Latency over time.
4. **System Health:** Shows API status, document count, and component health.

---

### Q37. How does the Streamlit frontend communicate with the FastAPI backend?

**Answer:**

Via **HTTP requests** using the `requests` library:

- The Streamlit app is configured with the `API_URL` environment variable (default: `http://localhost:8000`, or `http://api:8000` in Docker).
- File uploads use `requests.post(f"{API_URL}/upload", files=...)`.
- Queries use `requests.post(f"{API_URL}/query", json={"question": ...})`.
- Metrics use `requests.get(f"{API_URL}/eval-metrics")`.

In Docker, the UI container talks to the API container via Docker's internal DNS (`http://api:8000`), defined in `docker-compose.yml`.

---

## 9. DevOps, Docker & Deployment

### Q38. Explain your Docker setup.

**Answer:**

I have **two Dockerfiles:**

1. **`Dockerfile` (API):**
   - Base: `python:3.11-slim`
   - Installs system deps for PDF processing (libmupdf-dev, build-essential)
   - Layer caching: `requirements.txt` is copied and installed before source code, so code changes don't invalidate the dependency layer.
   - Creates data directories, exposes port 8000, runs uvicorn with 2 workers.
   - Built-in healthcheck: `curl -f http://localhost:8000/health`

2. **`Dockerfile.streamlit` (UI):**
   - Similar base, installs Streamlit dependencies.
   - Exposes port 8501, runs `streamlit run`.

3. **`docker-compose.yml`:**
   - Defines both services with networking.
   - UI `depends_on` API with `condition: service_healthy`.
   - Volume mounts for hot-reload during development.
   - Environment variables from `.env` file.

---

### Q39. Describe the CI/CD pipeline.

**Answer:**

The **GitHub Actions workflow** (`.github/workflows/deploy.yml`) has three stages:

1. **Test:** On every push/PR to `main` — installs deps, runs `pytest` with a test API key.
2. **Build & Push:** Only on `main` branch merge — builds both Docker images, tags with commit SHA + `latest`, pushes to Azure Container Registry (ACR).
3. **Deploy:** Updates Azure Container Apps with the new image SHA, then prints the live URLs.

This ensures that:

- No broken code gets deployed (tests gate the build).
- Every deployment is traceable to a specific commit (SHA tagging).
- Rollback is easy — just redeploy the previous SHA.

---

### Q40. Why Azure Container Apps?

**Answer:**

- **Serverless scaling:** Azure Container Apps scales to zero when idle, reducing costs for a student/demo deployment.
- **Built-in HTTPS:** Automatic TLS certificates and ingress routing.
- **Docker-native:** Just push a Docker image and it runs — no Kubernetes YAML needed.
- **Environment integration:** Easy to connect to Azure Key Vault for secrets, Azure Monitor for logging.
- **Cost:** The free tier is generous enough for a demo project.

---

## 10. Testing Strategy

### Q41. What is your testing strategy?

**Answer:**

I have **36 tests** across three test files:

1. **`test_ingestion.py`** — Tests document processing:
   - Chunk size and overlap validation
   - Text extraction from mock PDFs
   - Section detection
   - Metadata correctness
   - Edge cases (empty PDF, corrupted file)

2. **`test_retrieval.py`** — Tests hybrid search:
   - Semantic search returns relevant results
   - BM25 search works with keyword queries
   - Score fusion combines scores correctly
   - Cross-encoder reranking improves ordering
   - Thread safety of BM25 index

3. **`test_api.py`** — Tests FastAPI endpoints:
   - Health check returns 200
   - Upload accepts PDFs and rejects non-PDFs
   - Query returns structured response
   - Delete removes document from system
   - Rate limiting works
   - Error handling for missing documents

All tests use **mocking** (`unittest.mock`) to avoid needing real API keys or ML models during CI.

---

### Q42. How do you mock external dependencies in tests?

**Answer:**

I use `unittest.mock.patch` to mock:

- **Groq API calls** — Mock the `Groq` client to return predefined responses without making real API calls.
- **Sentence-transformers** — Mock `SentenceTransformer.encode()` to return random embeddings.
- **ChromaDB** — Mock the ChromaDB collection methods (`add`, `query`, `delete`).
- **File operations** — Use `tempfile` for temporary PDFs in tests.

Example:
```python
@patch("src.llm.generator.Groq")
def test_query_returns_answer(mock_groq):
    mock_groq.return_value.chat.completions.create.return_value = MockResponse(...)
    result = generator.generate(question="What was revenue?", chunks=[...])
    assert "revenue" in result["answer"].lower()
```

This ensures tests are **fast, deterministic, and don't require API keys**.

---

## 11. System Design & Scalability

### Q43. How would you scale this system for 10,000 concurrent users?

**Answer:**

Several changes needed:

1. **Vector DB:** Migrate from ChromaDB (single-node) to **Pinecone or Weaviate** (managed, distributed, handles billions of vectors).
2. **Caching:** Replace in-memory cache with **Redis** for distributed caching across multiple API instances.
3. **Load balancing:** Put API behind **nginx or AWS ALB** with multiple uvicorn workers across containers.
4. **Async processing:** Move PDF ingestion to a **background task queue** (Celery + Redis) since it's CPU-intensive.
5. **Kubernetes:** Replace Docker Compose with **Kubernetes** for auto-scaling, health checks, and rolling deployments.
6. **CDN:** Serve static UI assets via CDN.
7. **Rate limiting:** Move from in-memory to Redis-based distributed rate limiting.

---

### Q44. How would you handle multi-tenant (multiple companies) support?

**Answer:**

1. **Namespace isolation:** Each tenant gets a separate ChromaDB collection (e.g., `company_a_documents`, `company_b_documents`).
2. **Authentication:** Add JWT-based auth. Each token carries a `tenant_id` claim.
3. **Data isolation:** API middleware extracts `tenant_id` from the token and routes to the correct collection.
4. **Separate BM25 indices:** One per tenant to prevent cross-tenant data leakage.
5. **Resource quotas:** Limit documents, queries, and storage per tenant.

---

### Q45. What are the bottlenecks in your current architecture?

**Answer:**

1. **Single-node BM25:** In-memory BM25 index can't be shared across multiple API instances. Solution: Use Elasticsearch for BM25.
2. **Synchronous PDF processing:** Large PDFs block the API thread. Solution: Background task queue.
3. **No embedding cache:** Re-embedding the same query wastes computation. Solution: Cache query embeddings.
4. **ChromaDB limits:** Not designed for millions of documents. Solution: Migrate to a distributed vector DB.
5. **No auth:** Currently open. Solution: Add API key or JWT authentication.

---

### Q46. How would you add real-time document updates (streaming)?

**Answer:**

1. **WebSocket endpoint** in FastAPI for real-time processing status.
2. **Server-Sent Events (SSE)** for streaming LLM responses token-by-token.
3. **File watcher:** Use `watchdog` to monitor a directory for new PDFs and auto-ingest.
4. **Incremental indexing:** Instead of re-indexing everything, only process and embed new/changed pages.

---

## 12. Security & Production Readiness

### Q47. What security measures have you implemented?

**Answer:**

1. **CORS middleware:** Restricts which origins can call the API.
2. **Rate limiting:** Sliding-window per-IP rate limiter prevents abuse.
3. **Input validation:** Pydantic models validate all request payloads.
4. **File type restriction:** Only PDFs accepted, with size limits.
5. **Environment variables:** API keys stored in `.env` files, never hardcoded.
6. **Health checks:** Docker healthchecks ensure only healthy containers receive traffic.

**What I'd add for production:**

- JWT authentication with role-based access control
- API key rotation
- Request/response encryption (HTTPS via Azure)
- Input sanitization against prompt injection
- Audit logging for compliance

---

### Q48. How do you handle prompt injection attacks?

**Answer:**

Currently, there's basic protection through the **system prompt** which instructs the model to only answer from context. However, for production I'd add:

1. **Input sanitization:** Strip suspicious patterns like "ignore previous instructions" or "you are now...".
2. **Output validation:** Check if the LLM response contains leaked system prompt content.
3. **Separate context and instructions:** Use delimiters (`<<<CONTEXT>>>`) that make it harder for injected text to override system behavior.
4. **Content filtering:** Use a secondary classifier to detect injection attempts before sending to the LLM.

---

## 13. Machine Learning Concepts

### Q49. What is cosine similarity and why do you use it for embeddings?

**Answer:**

Cosine similarity measures the angle between two vectors, regardless of their magnitude:

```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```

- Range: [-1, 1] (1 = identical direction, 0 = orthogonal, -1 = opposite)
- For normalized embeddings, it equals the dot product.

**Why cosine over Euclidean distance?**

- Cosine similarity captures **semantic direction** — two sentences with the same meaning point in the same direction even if their embedding magnitudes differ.
- It's **scale-invariant**, so longer/shorter texts are compared fairly.
- `all-MiniLM-L6-v2` is specifically trained to optimize cosine similarity for semantic tasks.

---

### Q50. What is BM25 and how does it work?

**Answer:**

**BM25 (Best Matching 25)** is a probabilistic ranking function for keyword search. It extends TF-IDF with two improvements:

1. **Term Frequency Saturation:** TF-IDF scores keep increasing with term frequency. BM25 uses a saturation function so after a term appears enough times, additional occurrences have diminishing returns. Controlled by parameter `k1` (typically 1.2-2.0).

2. **Document Length Normalization:** Longer documents naturally have more term matches. BM25 normalizes by document length so short, focused documents aren't disadvantaged. Controlled by parameter `b` (typically 0.75).

Formula:
```
Score(Q, D) = Σ IDF(qi) × [f(qi, D) × (k1 + 1)] / [f(qi, D) + k1 × (1 - b + b × |D|/avgdl)]
```

In my system, I use `BM25Okapi` from the `rank-bm25` library which implements this standard algorithm.

---

### Q51. What is a bi-encoder vs a cross-encoder?

**Answer:**

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|-----------|---------------|
| **Input** | Encodes query and document **separately** | Encodes (query, document) **together** |
| **Output** | Two independent embedding vectors → compute similarity | Single relevance score |
| **Speed** | Fast (can pre-compute document embeddings) | Slow (must process each pair) |
| **Accuracy** | Good | Better (sees token interactions) |
| **Use case** | Initial retrieval (thousands of docs) | Reranking (tens of candidates) |

In my system:

- **Bi-encoder** (`all-MiniLM-L6-v2`) → Used for initial semantic search over all chunks in ChromaDB.
- **Cross-encoder** (`ms-marco-MiniLM-L6-v2`) → Used to rerank the top-20 candidates for maximum precision.

This is the **retrieve-then-rerank** pattern, standard in production search systems.

---

### Q52. Explain the concept of embeddings in simple terms.

**Answer:**

Embeddings convert text into a list of numbers (a vector) that captures the **meaning** of the text. Think of it as coordinates in a high-dimensional "meaning space":

- "What was the company's revenue?" → [0.23, -0.45, 0.12, ...]  (384 numbers)
- "How much money did the firm earn?" → [0.22, -0.44, 0.13, ...]  (very similar vector)
- "What color is the sky?" → [0.89, 0.12, -0.67, ...]  (very different vector)

The embedding model learns this mapping from training on billions of text pairs. Texts with similar meaning end up close together in this space, making similarity search possible.

---

### Q53. What is the Transformer architecture (briefly)?

**Answer:**

The Transformer is the neural network architecture behind all modern LLMs:

1. **Self-Attention:** Each token looks at every other token in the sequence to understand context. "Bank" in "river bank" vs "bank account" gets different representations because attention captures the surrounding context.
2. **Multi-Head Attention:** Multiple attention heads capture different types of relationships (syntactic, semantic, positional).
3. **Feed-Forward Networks:** After attention, each position is processed through a neural network.
4. **Layer stacking:** Transformers stack many such layers (LLaMA-3 has 80 layers for 70B parameters).

Key advantage over RNNs: **Parallelization** — all tokens are processed simultaneously instead of sequentially, enabling training on massive datasets.

---

### Q54. What is the difference between semantic search and keyword search?

**Answer:**

| Aspect | Semantic Search | Keyword Search (BM25) |
|--------|----------------|----------------------|
| **Mechanism** | Compares meaning via embeddings | Matches exact words |
| **"Revenue" vs "Income"** | Understands they're related ✅ | Misses the connection ❌ |
| **"EBITDA Q3 2023"** | May match similar but wrong periods | Exact match precision ✅ |
| **Typos** | Somewhat resilient | Strict match fails |
| **Speed** | Requires embedding + vector search | Simple inverted index |
| **Best for** | Intent understanding | Exact term matching |

That's why hybrid (combining both) works best — it gets the best of both worlds.

---

## 14. HR / Behavioral Questions

### Q55. What was the most challenging part of this project?

**Answer:**

The **hybrid search score fusion** was the trickiest part. The challenge was that semantic search and BM25 return scores on completely different scales — cosine similarity is [0,1] while BM25 scores can be any positive number. 

I had to design a normalization strategy that makes them comparable. I tried min-max normalization first, but outlier BM25 scores compressed the useful range. I settled on dividing by the max score with a cap of 10.0, which handles outliers gracefully. Then finding the right 60/40 weight ratio required experimentation with the evaluation dataset.

The cross-encoder reranking was also challenging — understanding when it helps vs. when it adds latency without benefit required careful benchmarking.

---

### Q56. What would you do differently if you started over?

**Answer:**

1. **Use LangChain or LlamaIndex** for the retrieval pipeline — I built everything from scratch which was a great learning experience, but these frameworks would have saved time in production.
2. **Add Elasticsearch** instead of in-memory BM25 for scalability and persistence.
3. **Implement streaming responses** — currently the user waits for the full answer. Streaming token-by-token feels much faster.
4. **Add user authentication** from day one — retrofitting auth is always harder.
5. **Use a proper experiment tracking tool** (MLflow or Weights & Biases) for systematic evaluation instead of JSON logs.

---

### Q57. How did you decide the chunk size and overlap?

**Answer:**

Through **empirical testing** and research:

- I tested chunk sizes of 256, 512, and 1024 tokens.
- **256** was too small — financial figures often span multiple sentences, and chunks lacked context.
- **1024** was too large — chunks contained mixed topics, reducing retrieval precision.
- **512** was the sweet spot — enough context for meaningful retrieval without noise.

For overlap:

- **0 overlap** caused information loss at boundaries (e.g., a revenue figure at the end of one chunk with its context at the start of the next).
- **100 tokens (~20% of chunk size)** ensures boundary information is captured in both adjacent chunks.

This aligns with common recommendations in the RAG literature.

---

### Q58. How do you stay updated with AI/ML developments?

**Answer:**

- **Papers:** I follow arXiv papers on RAG improvements (e.g., RAPTOR, Self-RAG, Corrective RAG).
- **Communities:** Active on Reddit r/MachineLearning, Twitter/X AI accounts, and Discord communities.
- **Hands-on:** I build projects (like this one) to apply new concepts.
- **Courses:** I've completed courses on LLMs, RAG systems, and vector databases.
- **Benchmarks:** I track MTEB leaderboard for embedding models and Chatbot Arena for LLMs.

---

### Q59. How do you handle disagreements with team members about technical decisions?

**Answer:**

I follow a structured approach:

1. **Listen first** — Understand their perspective and reasoning fully.
2. **Data-driven discussion** — Propose A/B testing or benchmarking both approaches. For example, if we disagreed on embedding model choice, I'd benchmark both on our evaluation dataset.
3. **Prototype** — Build a quick proof of concept for both approaches rather than arguing theoretically.
4. **Prioritize the user** — The approach that better serves the end user (auditor) should win.
5. **Document the decision** — Record why we chose option A over B for future reference.

---

### Q60. Where do you see this project going in the future?

**Answer:**

1. **Multi-modal support** — Handle charts, images, and infographics in annual reports using vision models.
2. **Comparative analysis** — "Compare Reliance vs TCS revenue growth over 3 years" — query across multiple documents.
3. **Agent capabilities** — Let the system perform multi-step reasoning: "Calculate the debt-to-equity ratio from the balance sheet data."
4. **Fine-tuned domain model** — Fine-tune a smaller model specifically on financial terminology for faster, cheaper inference.
5. **Real-time data** — Integrate with Bloomberg/Reuters APIs for live financial data alongside document Q&A.
6. **Compliance checker** — Auto-check if financial reports comply with regulatory standards (SEBI, SEC).

---

## Bonus: Deep-Dive Technical Questions

### Q61. How does ChromaDB store and index vectors internally?

**Answer:**

ChromaDB uses **HNSW (Hierarchical Navigable Small World)** graphs for approximate nearest neighbor search:

- Vectors are organized in a multi-layer graph structure.
- Each layer is a "skip list" of connections — upper layers have long-range connections for fast navigation, lower layers have short-range connections for precision.
- Search starts at the top layer, greedily navigates to the nearest region, then descends to lower layers for fine-grained search.
- Trade-off: Slightly approximate results (might miss the absolute nearest neighbor) but orders of magnitude faster than brute-force search. Typically 95-99% recall.

---

### Q62. What is the difference between `all-MiniLM-L6-v2` and `all-mpnet-base-v2`?

**Answer:**

| Aspect | MiniLM-L6 | MPNet-Base |
|--------|-----------|------------|
| Dimensions | 384 | 768 |
| Parameters | ~22M | ~110M |
| Speed | ~14K sent/sec | ~2.8K sent/sec |
| MTEB Score | Lower | Higher |
| Use case | Speed-sensitive, resource-constrained | Quality-critical |

I chose MiniLM for its speed and small footprint — important for containerized deployment where memory is limited. The quality difference is partially compensated by the cross-encoder reranker.

---

### Q63. Explain the difference between synchronous and asynchronous processing in your FastAPI app.

**Answer:**

- **Synchronous (current):** When a `/query` request comes in, the handler function runs the full pipeline (embed → search → rerank → LLM call) sequentially. The worker thread is blocked until completion.
- **Asynchronous (partial):** FastAPI itself uses ASGI (async server), so it can handle multiple connections concurrently. However, my CPU-bound operations (embedding, reranking) are synchronous and block the event loop.

**Improvement:** I should use `asyncio.to_thread()` or `run_in_executor()` to offload CPU-bound ML operations to a thread pool, freeing the event loop for other requests.

---

### Q64. How would you implement Retrieval-Augmented Fine-Tuning (RAFT)?

**Answer:**

RAFT combines RAG with fine-tuning:

1. **Training data:** Create (question, correct_chunks, distractor_chunks, answer) tuples.
2. **Fine-tune:** Train the LLM to generate answers from a mix of relevant and irrelevant chunks, teaching it to identify and use only the relevant ones.
3. **Inference:** At query time, still retrieve chunks (RAG-style) but the fine-tuned model is much better at ignoring irrelevant retrieved chunks.

This would improve my faithfulness metric since the model would be trained specifically on financial document Q&A with real retrieval noise.

---

### Q65. What is the difference between dense retrieval and sparse retrieval?

**Answer:**

- **Dense retrieval** (my semantic search): Converts text to dense, continuous vectors (e.g., 384 floats). Captures semantic meaning. Works well for paraphrases and conceptual matching.
- **Sparse retrieval** (my BM25): Represents text as sparse vectors in a vocabulary-sized space. Most dimensions are zero. Only non-zero where specific words appear. Works well for exact term matching.

My hybrid approach combines both — dense for meaning, sparse for precision. This is the current best practice, used by systems like Bing, Google, and Amazon search.

---

### Q66. How do you handle document versioning?

**Answer:**

Currently, each upload creates a new `document_id` based on the MD5 hash of the filename. If the same file is uploaded again, it creates duplicate chunks.

**Improvement plan:**

1. **Content hashing:** Hash the PDF content (not filename) to detect true duplicates.
2. **Version tagging:** Add a `version` field to metadata. When re-uploading the same document name, increment the version.
3. **Diff detection:** Compare new chunks against existing ones. Only add/update changed chunks.
4. **Version queries:** Allow users to query specific document versions or always use the latest.

---

### Q67. What is the time complexity of your retrieval pipeline?

**Answer:**

For a collection with N chunks:

| Stage | Time Complexity |
|-------|----------------|
| Query embedding | O(1) — fixed-length computation |
| Semantic search (ChromaDB/HNSW) | O(log N) — approximate NN search |
| BM25 search | O(N × Q) — N chunks, Q query terms |
| Score fusion | O(K) — K = 20 candidates |
| Cross-encoder reranking | O(K × T) — K pairs, T = model inference time |
| LLM generation | O(1) — API call, fixed context |

The BM25 O(N) component is the bottleneck for very large collections. Elasticsearch would reduce this to O(log N) via inverted indices.

---

### Q68. Explain the concept of HNSW (Hierarchical Navigable Small World) graphs.

**Answer:**

HNSW is the algorithm ChromaDB uses for fast vector similarity search:

1. **Build phase:** Vectors are inserted into a multi-layer graph. Higher layers have fewer nodes but long-range connections. Lower layers have all nodes with local connections.
2. **Search phase:**
   - Start at a random entry point in the top layer.
   - Greedily navigate to the nearest neighbor at this layer.
   - Move down to the next layer and repeat.
   - At the bottom layer, explore local neighborhood for the final k nearest neighbors.

Think of it like searching a city: first navigate between continents (top layer), then between cities (middle layer), then between streets (bottom layer).

**Parameters:**

- `M` = max connections per node (affects memory and accuracy)
- `ef` = beam width during search (affects speed vs. accuracy trade-off)

---

### Q69. What are the trade-offs between accuracy and latency in your system?

**Answer:**

| Decision | Accuracy Impact | Latency Impact |
|----------|----------------|----------------|
| More candidates (top-20 vs top-5) | ↑ better recall | ↑ slower fusion |
| Cross-encoder reranking | Optimizes candidate order | ↑ +232.6ms |
| Larger embedding model | ↑ better embeddings | ↑ slower encoding |
| More chunks to LLM (3 vs 10) | Mixed (more context vs noise) | ↑ slower generation |
| Higher confidence threshold (0.6 → 0.8) | ↑ fewer hallucinations | ~ same, but more abstentions |
| BM25 weight (0.4 → 0.7) | Depends on query type | ~ same |

I chose a balanced configuration that meets all targets: 62.6% context precision@3, 0.0% hallucination rate, and 1.99s average latency.

---

### Q70. How would you add support for non-English financial documents?

**Answer:**

1. **Multilingual embedding model:** Replace `all-MiniLM-L6-v2` with `paraphrase-multilingual-MiniLM-L12-v2` (supports 50+ languages, same architecture).
2. **Multilingual BM25:** Use language-specific tokenizers (e.g., Jieba for Chinese, MeCab for Japanese).
3. **Multilingual LLM:** LLaMA-3 already supports multiple languages, but I'd test quality for the target language.
4. **OCR language support:** Configure pytesseract with the appropriate language pack (e.g., `lang='hin'` for Hindi).
5. **UI internationalization:** Add language selector to the Streamlit UI.

---

### Q71. What is Reciprocal Rank Fusion (RRF) and how does it differ from your weighted fusion?

**Answer:**

**RRF** is an alternative fusion method that doesn't need score normalization:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `k` is a constant (typically 60) and `rank_i(d)` is the document's rank in system `i`.

**Comparison:**

| Aspect | Weighted Fusion (mine) | RRF |
|--------|----------------------|-----|
| Needs score normalization | Yes | No (uses ranks) |
| Sensitive to score distribution | Yes | No |
| Tuning | Weight ratio (0.6/0.4) | k constant |
| Better when | Scores are well-calibrated | Scores are on different scales |

I could switch to RRF to eliminate the BM25 normalization step. However, my weighted approach works well because I explicitly handle normalization.

---

### Q72. How do you handle tables in financial documents?

**Answer:**

Financial tables (balance sheets, income statements) need special handling because:

1. **Detection:** pdfplumber's `page.extract_tables()` detects tabular structures.
2. **Markdown conversion:** Detected tables are converted to Markdown format to preserve row/column relationships:
   ```
   | Item | FY2023 | FY2022 |
   |------|--------|--------|
   | Revenue | 240,000 Cr | 210,000 Cr |
   ```

3. **Merge with text:** The Markdown table is appended to the page's text before chunking.
4. **Why Markdown?** LLMs understand Markdown tables well — they can correctly identify specific cells. Plain text concatenation of table data loses structure entirely.

---

### Q73. What observability tools would you add for production?

**Answer:**

1. **OpenTelemetry** — Already in dependencies. Would add distributed tracing to track a query through embed → search → rerank → LLM stages.
2. **Prometheus + Grafana** — Expose metrics endpoint with counters (query count, error count), histograms (latency distribution), and gauges (cache hit ratio, document count).
3. **Structured logging** — Export Loguru logs to ELK stack (Elasticsearch, Logstash, Kibana) for search and alerting.
4. **Alerting** — PagerDuty/OpsGenie alerts for error rate spikes, latency degradation, or LLM API failures.
5. **Dashboard:** Query volume, P95 latency, cache hit rate, confidence score distribution.

---

### Q74. How would you implement A/B testing for different retrieval strategies?

**Answer:**

1. **Feature flags:** Use a config flag to route a percentage of traffic to the new strategy.
2. **Parallel execution:** Run both strategies simultaneously, log results for both, but only show one to the user.
3. **Metrics collection:** Record per-strategy metrics (precision, latency, user satisfaction) in a structured format.
4. **Statistical testing:** After sufficient data, run a Mann-Whitney U test or bootstrap confidence intervals to determine if the difference is statistically significant.
5. **Gradual rollout:** Start with 5% traffic on the new strategy, increase to 50% if metrics look good, then 100%.

---

### Q75. What is the Cold Start problem in your system and how do you handle it?

**Answer:**

**Cold start** happens when the system starts with no documents:

- **Semantic search:** ChromaDB returns empty results.
- **BM25:** Empty index, returns nothing.
- **LLM:** Gets empty context, can't generate useful answers.

**How I handle it:**

1. **Graceful empty state:** The `/query` endpoint checks if any documents exist. If not, returns a user-friendly message: "No documents uploaded yet."
2. **UI guidance:** The Streamlit UI prompts users to upload documents first.
3. **Health check:** The `/health` endpoint reports `documents_count: 0` so monitoring can detect this state.

---

### Q76. Explain Thread Safety in your application.

**Answer:**

Thread safety is critical because FastAPI can handle concurrent requests:

1. **BM25 Index:** Protected by `threading.Lock`. Any operation that reads or mutates the BM25 index acquires the lock first, preventing data corruption from concurrent uploads/deletes.
2. **Query Cache:** Has its own `threading.Lock`. Concurrent reads/writes to the cache dict are serialized.
3. **Rate Limiter:** Uses `collections.deque` which is thread-safe for append/popleft operations in CPython.
4. **ChromaDB:** Handles its own thread safety internally.
5. **Groq Client:** Stateless HTTP client — each request is independent, naturally thread-safe.

---

### Q77. What is the difference between L1, L2, and Cosine distance for vector search?

**Answer:**

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **L1 (Manhattan)** | Σ\|a_i - b_i\| | Sparse features, outlier-robust |
| **L2 (Euclidean)** | √(Σ(a_i - b_i)²) | When magnitude matters |
| **Cosine** | 1 - cos(θ) | When direction matters, not magnitude |

For text embeddings, **cosine** is standard because:

- Two sentences with the same meaning should be "similar" regardless of how long they are.
- Cosine measures the angle between vectors, ignoring magnitude.
- Most embedding models are trained to optimize cosine similarity.

ChromaDB uses cosine distance by default, which is what I use.

---

### Q78. How do you handle edge cases in query processing?

**Answer:**

1. **Empty query:** Validation rejects empty strings with a 422 error.
2. **Very long query:** Truncated to model's max input length before embedding.
3. **Non-English query:** The embedding model handles basic multilingual content; for full support, I'd switch to a multilingual model.
4. **SQL injection-style:** Not applicable (no SQL database), but I sanitize inputs via Pydantic models.
5. **Repeated queries:** Served from cache without re-running the pipeline.
6. **No relevant chunks found:** Confidence threshold triggers the hallucination guard, returning a safe abstention response.

---

### Q79. What is Retrieval-Augmented Generation vs. Knowledge Graphs? Which would you choose?

**Answer:**

| Aspect | RAG | Knowledge Graphs |
|--------|-----|-----------------|
| Data format | Unstructured text (PDFs) | Structured triples (entity-relation-entity) |
| Setup complexity | Lower (embed + search) | Higher (entity extraction, graph construction) |
| Query type | Natural language Q&A | Structured queries, relationship traversal |
| Updates | Easy (re-embed new docs) | Complex (maintain graph consistency) |
| Reasoning | Limited to retrieved context | Multi-hop reasoning possible |

For this project, **RAG is the right choice** because:

- Financial documents are primarily unstructured text.
- Auditors ask natural language questions, not graph queries.
- Documents change quarterly — RAG handles updates easily.

For a system that needs to answer "Which subsidiaries of Reliance had revenue > 5000 Cr AND are based in Gujarat?", a knowledge graph would be better.

---

### Q80. If you had unlimited budget and time, what would the ideal version of this system look like?

**Answer:**

1. **Multi-modal ingestion:** Parse charts, graphs, and images using GPT-4V or Gemini Vision.
2. **Graph RAG:** Build a knowledge graph alongside vector search for multi-hop reasoning.
3. **Agentic RAG:** Let the system decompose complex queries into sub-queries, search multiple times, and synthesize.
4. **Fine-tuned financial LLM:** Train on millions of financial documents for domain expertise.
5. **Real-time streaming:** WebSocket-based streaming answers with live source highlights.
6. **Regulatory compliance engine:** Auto-check reports against SEBI/SEC/IFRS standards.
7. **Collaborative features:** Multiple auditors can annotate, share queries, and build on each other's findings.
8. **Global deployment:** Multi-region Kubernetes deployment with <100ms latency worldwide.
9. **Continuous evaluation:** Automated RAGAs runs on every deployment, blocking releases that degrade metrics.
10. **Feedback loop:** User thumbs-up/down feeds back into retrieval and generation improvement.

---

### Q81. How does your RAG system handle numerical math or calculations (e.g., Year-over-Year revenue growth)?

**Answer:**

Standard RAG systems are notorious for failing at math. If a user asks "What was the percentage change in net profit between FY2022 and FY2023?", the retriever will fetch the chunks containing both years' profits, but the LLM will try to calculate the percentage growth. Since LLMs are next-token predictors (not calculators), they often hallucinate arithmetic results.

**Follow-up: How do you prevent calculation errors in production?**

1. **Explicit instructions:** In the system prompt, instruct the LLM: *"If a question requires calculation and the exact calculation is not in the text, do not calculate it yourself. Explain the math steps needed and state that you cannot execute arithmetic."*
2. **Tool-Use (Function Calling):** Implement an agentic workflow where the LLM parses the calculation query and generates a Python expression (e.g., `(240000 - 210000) / 210000`). This expression is executed in a sandboxed environment, and the result is returned to the user.
3. **Structured extraction:** Extract the financial tables into a structured SQL database or pandas DataFrames and use Text-to-SQL or code generation to calculate figures programmatically.

---

### Q82. What happens if a critical statement is cut in half by your 512-token chunk boundaries? How do you solve chunk fragmentation?

**Answer:**

This is the **chunk fragmentation** problem. If a sentence like *"We expect revenue to double, unless [boundary] our primary supplier goes bankrupt"* is split, Chunk A is overly optimistic and Chunk B is highly critical, and neither gets the full context.

**Follow-up: What strategies solve this?**

1. **Overlapping:** Using a 100-token overlap ensures that the boundaries of adjacent chunks share context, capturing statements that cross the line.
2. **Parent-Child Retrieval (Small-to-Large Chunking):**
   - We split the document into small chunks (e.g., 128 tokens) for vector search, because small chunks are semantically dense and retrieve better.
   - Each small chunk is linked to a larger "parent" chunk (e.g., 1024 tokens) or the full page.
   - When a small chunk is retrieved, we feed the **parent** chunk to the LLM. This provides full surrounding context.
3. **Sentence-Window Retrieval:** Retrieve a specific sentence, but feed that sentence along with `N` sentences before and after it to the LLM.

---

### Q83. When the system gives a wrong answer, how do you diagnose if it's a retrieval failure or a generation failure?

**Answer:**

This is the core diagnostic question in RAG engineering.

- **Retrieval Failure:** The retriever failed to find the chunks containing the answer. The LLM never had a chance.
- **Generation Failure:** The retriever found the correct chunks, but the LLM hallucinated, misread the numbers, ignored the facts, or was confused by distractor chunks.

**Follow-up: How do you measure and isolate this programmatically?**

1. **Context Recall (Retrieval Metric):** Check if the ground-truth information is present in the top-K retrieved chunks. We can calculate this by matching sentences or using an LLM evaluator to check if the retrieved context contains the ground truth.
2. **Faithfulness (Generation Metric):** Check if the generated answer can be strictly derived from the retrieved context. If Context Recall is 100% but Faithfulness is low, it's a generation failure.
3. **Human-in-the-loop audit logs:** Store the retrieved chunks alongside the query and generated answer. When a user reports a bad answer, look at the logs to see if the correct numbers were in the "Sources" card.

---

### Q84. Financial tables can be extremely wide or span multiple pages. How does your table parsing handle this?

**Answer:**

This is a major issue in PDF parsing.

- If a table is wider than the page, text extraction tools like `pdfplumber` might extract columns out of order or merge cells vertically.
- If a table spans multiple pages, the column headers are on Page 1, but Page 2 has only rows of raw numbers. Chunks from Page 2 will lose all column header context.

**Follow-up: How do you handle multi-page tables?**

1. **Header Injection:** When a multi-page table is detected during parsing, we extract the table header from Page 1 and prepend it to the table segments of Page 2, Page 3, etc., before chunking.
2. **Vision-based parsing (OCR / Table Detection):** Use deep-learning-based table parsers like TableTransformer, layout-parser, or Azure Document Intelligence, which reconstruct the physical grid layout of the table into structured JSON/HTML before feeding it to the pipeline.
3. **Markdown Representation:** Ensure table rows are fully written as complete Markdown table lines so that row and column semantics are maintained.

---

### Q85. How do you implement Role-Based Access Control (RBAC) in RAG to prevent data leaks?

**Answer:**

In enterprise RAG (e.g., at KPMG), security is paramount. A user querying the system should only retrieve documents they are authorized to see.

**Follow-up: How is this implemented at the database level?**

1. **Metadata Filtering:** When ingesting documents, we add security tags in the metadata (e.g., `{"allowed_roles": ["auditor_reliance", "admin"], "client_id": "reliance_2023"}`).
2. **Query-Time Filtering:** When a user queries the system, the API intercepts the request, retrieves the user's role/tenant claims from their JWT token, and passes them as metadata filters to ChromaDB:
   `collection.query(query_embeddings=[...], where={"client_id": user_client_id})`
   This ensures ChromaDB *only* performs vector search on allowed vectors, mathematically guaranteeing that unauthorized chunks are never even evaluated.

3. **Separate Collections (Physical Isolation):** For strict compliance, host separate vector DB collections or separate physical instances for different clients.

---

### Q86. What is the impact of frequent document insertions/deletions on your Vector Database (HNSW index)?

**Answer:**

ChromaDB uses HNSW (Hierarchical Navigable Small World) for vector indexing. HNSW builds a graph structure.

- **Deletions:** Deleting a document doesn't immediately rebuild the graph. It usually "tombstones" the nodes, which can leave holes in the HNSW graph, degrading query recall over time.
- **Insertions:** Frequently adding single documents causes index fragmentation. Every insertion updates graph nodes, which can block read queries or lead to high write latency.

**Follow-up: How do you manage this in production?**

1. **Batching:** Instead of updating the vector DB for every single PDF page upload in real-time, batch updates during low-traffic periods.
2. **Index Rebuilding:** Schedule periodic index optimization/rebuilds (compaction) to clean up deleted/tombstoned nodes and restore recall rates.
3. **Read-Write separation:** Use a primary-replica database pattern where writes are sent to a primary writer, and read queries are served from read-replicas. Replicate the updated index asynchronously.

---

### Q87. What are the specific failure modes of your RAG pipeline in production, and how do you handle them?

**Answer:**

Any production RAG system has inherent edge cases. In a financial auditing domain, the principal failure modes are:

1. **Table-Heavy Footnotes:** Small, fine-print disclosures or tables inside footnotes (e.g., related party tables or lease reconciliation rows) are often poorly captured by standard tokenizers and markdown converters, leading to retrieval failures.
   - *Mitigation:* We use a page-aware, section-aware PDF parser that detects tables, extracts them as structured markdown tables, and maps them to their respective footnote references in the metadata.
2. **Multi-Page Tables Spanning OCR/Page Boundaries:** When a large table spans multiple pages, column headers are printed on Page 1 but omitted on subsequent pages. Chunks extracted from Page 2 and Page 3 lose their column mapping context, causing retrieval and synthesis failure.
   - *Mitigation:* We implement a header-injection mechanism during ingestion that detects multi-page tables, extracts the column headers from the start of the table, and prepends them to every segmented table chunk on successive pages.
3. **Sub-pixel or Poor OCR on Scanned Footnotes:** Scanned PDFs or scanned signature pages can result in distorted numbers (e.g., mistaking "8" for "3" or "0" for "8"), which can bypass semantic checks but lead to auditing errors.
   - *Mitigation:* The Deterministic NCV guard serves as a failsafe here. If OCR errors lead to a mismatch between what the LLM prints and what is in the text corpus, the NCV guard flags it and abstains.

---

### Q88. How does this system tie into Fraud Detection and Financial Auditing (specifically matching PayU's risk and fraud domains)?

**Answer:**

While built as an assistant for auditors, this RAG system can be directly mapped to automated **fraud, anomaly, and risk detection** — which is a short conceptual step from PayU's payment fraud and credit risk domain.

1. **Cross-Statement Reconciliation (Footnote-to-Table Discrepancy):** Fraud in corporate reports often manifests as discrepancies between the main financial statements (which are highly visible) and the dense footnotes (where risks are buried). By querying the RAG system to reconcile figures (e.g., "Reconcile the total lease liabilities reported in the Balance Sheet with the lease liability commitments listed in Note 5.3"), we can build an automated discrepancy checker. If the numbers do not reconcile, it triggers a fraud red-flag.
2. **Related Party Anomaly Detection:** In payment and credit fraud, related party transactions (e.g., transactions between shell companies or un-consolidated subsidiaries) are high-risk. A query like "Identify all related party advances in Note 5.2 and verify if they are approved as arm's length by the audit committee" allows auditors to instantly highlight credit risk and potential asset siphoning.
3. **Credit Risk Evaluation (Interest Coverage & Leverage):** PayU evaluates merchants for business viability and credit risk. By querying and calculating financial ratios directly from the audited RAG data (e.g., EBITDA of ₹74,000 Crore / interest expense of ₹4,000 Crore = 18.5x coverage), risk analysts can evaluate a merchant's solvency and leverage ratios using verified audit-trail facts, preventing defaults.

---

### Q89. Why not just fine-tune an LLM instead of building a RAG pipeline for financial auditing?

**Answer:**

This is a classic architectural question. In financial auditing and compliance, **fine-tuning is never a replacement for RAG**; instead, they serve complementary purposes. Here is why we anchor the system on RAG:

1. **Hard Factual Grounding (Auditable Citations):** Auditors require a strict audit trail. A fine-tuned LLM generates responses from its weights (parametric memory), which cannot provide verifiable source citations (page numbers, exact document excerpts). RAG provides exact citations from the source PDF.
2. **Elimination of Hallucinations:** Fine-tuned models still suffer from memory drift and hallucinate figures. In auditing, a hallucinated number (e.g. stating lease liability is ₹3,500 Cr instead of ₹3,200 Cr) is a critical failure. RAG allows us to place deterministic guards (like NCV) on top of the context chunks to guarantee zero hallucinations.
3. **Real-time Data Updates:** Financial documents change constantly. If we rely on fine-tuning, we must retrain the model every time a new quarterly report, annexure, or audit note is released. With RAG, updating the system is as simple as parsing the new document and upserting it into ChromaDB—taking seconds rather than hours of GPU training.
4. **Access Control & Security:** Financial statements often contain sensitive, pre-release, or siloed figures (e.g., executive remuneration, pending litigation). RAG allows us to enforce document-level metadata filtering at query time, ensuring users only retrieve information they have permissions to see. With a fine-tuned model, all training data is baked into the model weights, making access control impossible.

---

### Q90. How did you design your fine-tuning dataset and cloud-based QLoRA training loop?

**Answer:**

To show how fine-tuning complements RAG, I implemented a cloud-based fine-tuning pipeline:

1. **Synthetic Dataset Generation:** I wrote an automated generator (`generate_ft_dataset.py`) that extracts page-by-page text from the mock PDF and queries an LLM in smaller batches (5 pairs per call) to generate 90 high-quality training pairs in Alpaca format (`instruction`, `input`, `output`). Batching prevents string truncation and ensures well-formed JSON.
2. **OpenAI JSONL Formatting:** To avoid downloading heavy model weights locally and overloading the host, I wrote a converter (`cloud_fine_tune.py`) that structures the training pairs into OpenAI's messages format:
   ```json
   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Context: [input]\n\nQuestion: [instruction]"}, {"role": "assistant", "content": "[output]"}]}
   ```
3. **Cloud-Based Training:** The dataset is uploaded via OpenAI's SDK, launching a fine-tuning job on `gpt-4o-mini` in the cloud. This handles model training entirely on cloud infrastructure, keeping the local environment clean and fast.
4. **Local Generator Integration:** I updated the RAG generator (`generator.py`) to detect if a fine-tuned model ID (e.g. `ft:gpt-4o-mini...`) is configured, seamlessly routing queries to the custom cloud-tuned model.

---

### Q91. What is the value of this fine-tuned model in the RAG pipeline?

**Answer:**

While RAG handles **knowledge retrieval**, the fine-tuned model is optimized for **behavioral alignment**:
- **Structured Auditing Tone:** The fine-tuned model learns to write answers in a conservative, evidence-based, formal auditing style (strictly quoting Note numbers and pages).
- **Format Adherence:** It learns to output tables, bullet points, and key-value pairs in a consistent, parser-friendly layout.
- **Improved Context Synthesis:** It learns to better synthesize and compare numbers across multiple retrieved chunks (for example, reconciling table figures against footnote disclosures).
- **Safe Refusal (Abstention):** It is trained on out-of-context training samples to strictly output a standardized low-confidence refusal ("I couldn't find reliable information...") when the context lacks sufficient evidence, serving as a secondary layer of hallucination defense.

---

## Quick Reference: One-Liners for Common Questions

| Question | Quick Answer |
|----------|-------------|
| Why RAG over fine-tuning? | Data freshness, traceability, cost, hallucination control |
| Why hybrid search? | Semantic captures meaning, BM25 captures exact terms — best of both |
| Why cross-encoder reranking? | Optimizes candidate order (achieving 62.6% Context Precision@3 under strict verification) |
| Why Groq over OpenAI? | 10x faster inference, free tier, LLaMA-3 quality |
| Why ChromaDB? | Simple, file-based, no server needed, Python-native |
| Why 512 chunk size? | Sweet spot between context richness and retrieval precision |
| Why 0.6 confidence threshold? | Part of a two-tiered guard (Thresholding + NCV) that reduces hallucination rate from 13.0% to 0.0% |
| Why temperature 0.1? | Financial Q&A needs factual precision, not creativity |
| Why FastAPI? | Async, auto-docs, Pydantic validation, high performance |
| Why Docker? | Consistent environments, easy deployment, CI/CD compatible |
| How to handle math in RAG? | Explicit prompts, tool-use (Python execution), or structured SQL data extraction |
| How to handle wide tables? | Header injection, vision-based parsing, markdown representation |
| How to enforce RAG security? | Metadata filtering at query time or physical collection isolation |

---

> **Good luck with your interview! 🚀**  
> Remember: Interviewers value understanding *why* you made decisions, not just *what* you built. Always explain the trade-offs.
