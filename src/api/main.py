"""
FinDoc Intelligence — FastAPI Backend

Endpoints:
  GET  /health              - Health check
  POST /upload              - Upload and process a financial PDF
  POST /query               - RAG query with hallucination guard
  GET  /eval-metrics        - Aggregated evaluation metrics
  GET  /documents           - List all ingested documents
  DELETE /documents/{id}   - Delete a document and its chunks
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Load .env first so components can read API keys
load_dotenv()

from src.evaluation.ragas_evaluator import RAGAsEvaluator
from src.ingestion.document_processor import DocumentProcessor
from src.llm.generator import FinancialAnswerGenerator
from src.retrieval.hybrid_search import HybridRetriever


# ── Singletons (initialised at startup via lifespan) ──────────────────────────

processor: DocumentProcessor
retriever: HybridRetriever
generator: FinancialAnswerGenerator
evaluator: RAGAsEvaluator


# ── Caching & Rate Limiting Classes ───────────────────────────────────────────

class QueryCache:
    """Thread-safe in-memory cache for query responses."""
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}

    def get(self, question: str, top_k: int, confidence_threshold: float) -> Optional[dict]:
        key = (question.strip().lower(), top_k, confidence_threshold)
        with self._lock:
            return self._cache.get(key)

    def set(self, question: str, top_k: int, confidence_threshold: float, val: dict) -> None:
        key = (question.strip().lower(), top_k, confidence_threshold)
        with self._lock:
            self._cache[key] = val

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("Query cache invalidated.")

query_cache = QueryCache()


class InMemoryRateLimiter:
    """Thread-safe in-memory request rate limiter."""
    def __init__(self, requests_limit: int, window_seconds: int):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.history = collections.defaultdict(list)
        self._lock = threading.Lock()
        
    def is_rate_limited(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            self.history[key] = [t for t in self.history[key] if now - t < self.window_seconds]
            if len(self.history[key]) >= self.requests_limit:
                return True
            self.history[key].append(now)
            return False

# Limit to 30 queries/minute and 5 uploads/minute per client IP
query_limiter = InMemoryRateLimiter(requests_limit=30, window_seconds=60)
upload_limiter = InMemoryRateLimiter(requests_limit=5, window_seconds=60)


# ── Authentication Dependency ──────────────────────────────────────────────────

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Enforces API Key check if API_KEY is set in environment."""
    expected_key = os.getenv("API_KEY")
    if expected_key:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key (X-API-Key header required).",
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy components once at startup."""
    global processor, retriever, generator, evaluator

    logger.info("Starting FinDoc Intelligence API …")
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=100)
    retriever = HybridRetriever(
        chroma_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        semantic_weight=0.6,
        bm25_weight=0.4,
    )
    # Generator is optional if GROQ_API_KEY is missing (fail gracefully)
    try:
        generator = FinancialAnswerGenerator()
    except EnvironmentError as exc:
        logger.warning(str(exc))
        generator = None  # type: ignore[assignment]

    evaluator = RAGAsEvaluator()
    logger.info("All components initialised.")
    yield
    logger.info("Shutting down FinDoc Intelligence API.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinDoc Intelligence",
    description="RAG-Based Financial Document Intelligence System for KPMG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Financial question to answer")
    top_k: int = Field(3, ge=1, le=10, description="Number of source chunks to retrieve")
    confidence_threshold: float = Field(
        0.6, ge=0.0, le=1.0, description="Min retrieval score to proceed with LLM"
    )

class SourceChunk(BaseModel):
    document: str
    page: str | int
    section: str
    excerpt: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    retrieval_score: float
    source_chunks: List[SourceChunk]
    latency_ms: int

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    document_name: str
    chunks_created: int
    total_pages: int

class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    uploaded_at: str

class EvalMetrics(BaseModel):
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    hallucination_rate: float
    avg_latency_ms: float
    total_queries: int
    last_evaluated: Optional[str]
    source: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Returns system health and component statuses."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "vector_store": "ready",
            "total_chunks": retriever.chunk_count(),
            "llm": "ready" if generator else "no_api_key",
        },
    }


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a financial PDF and ingest it into the RAG pipeline.

    - Validates file type (PDF only) and size (≤ 50 MB)
    - Persists original PDF to local folder data/raw/
    - Cleans up any existing index for the same document name to prevent orphans
    - Extracts text/tables, creates chunks with metadata
    - Embeds chunks and stores in ChromaDB
    """
    # Rate limit check
    ip = getattr(request, "client", None)
    client_ip = ip.host if ip else "unknown"
    if upload_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many document uploads. Limit is 5 uploads per minute.",
        )

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    # Check file size (50 MB cap)
    MAX_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is 50 MB.",
        )

    # Resolve paths & create directories
    doc_name = file.filename
    doc_id = processor._generate_doc_id(doc_name)
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    pdf_path = os.path.join(raw_dir, doc_name)

    # Overwrite if exists, but delete from database first to avoid orphaned chunks
    retriever.delete_document(doc_id)

    # Persist raw PDF
    with open(pdf_path, "wb") as f:
        f.write(content)

    try:
        logger.info(f"Processing upload: {doc_name}")
        result = processor.process_document(pdf_path, doc_name)

        if not result["success"]:
            # Delete persisted raw file on processing failure
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF processing failed: {result.get('error', 'Unknown error')}",
            )

        # Add chunks to retriever
        success = retriever.add_chunks(result["chunks"])
        if not success:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to index document chunks.",
            )

        # Invalidate response cache
        query_cache.clear()

        return UploadResponse(
            success=True,
            document_id=result["document_id"],
            document_name=result["document_name"],
            chunks_created=result["total_chunks"],
            total_pages=result["total_pages"],
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    dependencies=[Depends(verify_api_key)],
)
async def query_documents(request: Request, query_req: QueryRequest):
    """
    Answer a financial question using RAG.

    Steps:
    1. Check query cache
    2. Hybrid search (semantic + BM25) → top-k chunks
    3. Hallucination guard (confidence threshold check)
    4. Groq LLM generation with structured prompt
    5. Return answer + source citations + latency
    """
    # Rate limit check
    ip = getattr(request, "client", None)
    client_ip = ip.host if ip else "unknown"
    if query_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many queries. Limit is 30 queries per minute.",
        )

    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not available. Set GROQ_API_KEY in your .env file.",
        )

    if retriever.chunk_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been uploaded yet. Please upload a PDF first.",
        )

    # 1. Check Query Cache
    cached_res = query_cache.get(query_req.question, query_req.top_k, query_req.confidence_threshold)
    if cached_res:
        logger.info(f"Query Cache Hit: '{query_req.question}'")
        return QueryResponse(**cached_res)

    # 2. Retrieve
    context_chunks = retriever.hybrid_search(
        query_req.question, top_k=query_req.top_k
    )

    # 3. Generate
    try:
        response = generator.generate(
            question=query_req.question,
            context_chunks=context_chunks,
            confidence_threshold=query_req.confidence_threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    # Record for evaluation
    evaluator.record_query(
        question=query_req.question,
        answer=response["answer"],
        source_chunks=response["source_chunks"],
        confidence=response["confidence"],
        latency_ms=response["latency_ms"],
    )

    # Cache response
    query_cache.set(query_req.question, query_req.top_k, query_req.confidence_threshold, response)

    return QueryResponse(**response)


@app.get(
    "/eval-metrics",
    response_model=EvalMetrics,
    tags=["Evaluation"],
    dependencies=[Depends(verify_api_key)],
)
async def get_eval_metrics():
    """Return aggregated evaluation metrics for all queries processed so far."""
    metrics = evaluator.get_metrics()
    return EvalMetrics(**metrics)


@app.get(
    "/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)
async def list_documents():
    """List all documents currently indexed in the system."""
    docs = retriever.list_documents()
    return {
        "total_documents": len(docs),
        "total_chunks": retriever.chunk_count(),
        "documents": docs,
    }


@app.delete(
    "/documents/{document_id}",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)
async def delete_document(document_id: str):
    """Delete a document and all its associated chunks from the system."""
    # Find matching document name in list of documents to delete raw file
    docs = retriever.list_documents()
    doc_name = None
    for d in docs:
        if d["document_id"] == document_id:
            doc_name = d["document_name"]
            break

    deleted_chunks = retriever.delete_document(document_id)
    if deleted_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found.",
        )

    # Delete raw PDF
    if doc_name:
        pdf_path = os.path.join("data", "raw", doc_name)
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.info(f"Deleted raw PDF file: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to delete raw PDF {pdf_path}: {e}")

    # Invalidate cache
    query_cache.clear()

    return {"success": True, "document_id": document_id, "deleted_chunks": deleted_chunks}


@app.post(
    "/eval/run-ragas",
    tags=["Evaluation"],
    dependencies=[Depends(verify_api_key)],
)
async def run_full_ragas(test_dataset_path: Optional[str] = None):
    """
    Run the full RAGAs evaluation suite.

    Populates questions with contexts and answers dynamically from retriever and generator
    before evaluating. Requires GROQ_API_KEY or OPENAI_API_KEY.
    """
    dataset_path = test_dataset_path or "./data/processed/eval_dataset.json"
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation dataset not found: {dataset_path}"
        )

    try:
        with open(dataset_path) as f:
            dataset = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load dataset: {str(e)}"
        )

    questions = dataset.get("questions", [])
    ground_truths = dataset.get("ground_truths", [])

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation dataset is empty."
        )

    if retriever.chunk_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents indexed. Evaluation requires uploaded reports."
        )

    # 1. Populate contexts and answers dynamically from retriever and generator
    populated_contexts = []
    populated_answers = []

    logger.info(f"Generating answers and contexts for {len(questions)} evaluation questions...")
    for q in questions:
        # Retrieve
        chunks = retriever.hybrid_search(q, top_k=3)
        chunk_texts = [c["text"] for c in chunks]
        populated_contexts.append(chunk_texts)

        # Generate
        if generator:
            try:
                gen_res = generator.generate(
                    question=q,
                    context_chunks=chunks,
                    confidence_threshold=0.0  # bypass guard to always evaluate answers
                )
                populated_answers.append(gen_res["answer"])
            except Exception as e:
                logger.error(f"Gen failed for eval question '{q}': {e}")
                populated_answers.append("Error generating answer.")
        else:
            populated_answers.append("LLM not available.")

    # Save populated dataset
    populated_data = {
        "questions": questions,
        "ground_truths": ground_truths,
        "contexts": populated_contexts,
        "answers": populated_answers
    }

    populated_path = "./data/processed/eval_dataset_populated.json"
    with open(populated_path, "w") as f:
        json.dump(populated_data, f, indent=2)

    logger.info(f"Populated dataset saved to {populated_path}")

    # 2. Run evaluator on populated dataset
    metrics = evaluator.run_ragas(populated_path)
    if metrics is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "RAGAs evaluation failed. Ensure GROQ_API_KEY or OPENAI_API_KEY is set."
            ),
        )
    return metrics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
