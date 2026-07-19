# Yeh file FastAPI backend web server and API endpoints ko define karti hai.
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

# Dynamic type checking annotations support imports.
from __future__ import annotations

# Thread structures rate limiting caching checks collections package.
import collections
# Input output parameters json format parse module.
import json
# File check routes operations environment path os module.
import os
# Directories copy and delete management module.
import shutil
# Temporary storage files directories builders package.
import tempfile
# System threads locks implementations module.
import threading
# System timing track measurements module.
import time
# Lifecycle context managers FastAPI lifespan support.
from contextlib import asynccontextmanager
# Data type arrays list typing check annotations parameters.
from typing import List, Optional

# Load env file configurations.
from dotenv import load_dotenv
# FastAPI frameworks imports.
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Request, Header, Depends
# Cross-Origin resource settings middleware FastAPI config.
from fastapi.middleware.cors import CORSMiddleware
# Console logging visual tracking logger.
from loguru import logger
# Pydantic schemas builders parameters.
from pydantic import BaseModel, Field

# Pehle env configurations variables load karega.
load_dotenv()

# Internal components module packages loading.
# Accuracy checks evaluator interface class load.
from src.evaluation.ragas_evaluator import RAGAsEvaluator
# File parsing ingestion class processor load.
from src.ingestion.document_processor import DocumentProcessor
# LLM generation framework answer classes load.
from src.llm.generator import FinancialAnswerGenerator
# Vector DB and keywords search retrievers class load.
from src.retrieval.hybrid_search import HybridRetriever


# ── Singletons (initialised at startup via lifespan) ──────────────────────────

# Global instance reference variable declarations processor.
processor: DocumentProcessor
# Retriever class global instance reference.
retriever: HybridRetriever
# Answer generator class global instance.
generator: FinancialAnswerGenerator
# Ragas evaluator class global instance.
evaluator: RAGAsEvaluator


# ── Caching & Rate Limiting Classes ───────────────────────────────────────────

# Safe caching memory data structures class.
class QueryCache:
    """Thread-safe in-memory cache for query responses."""
    # Initialization setup values.
    def __init__(self):
        # Synchronization lock threads setup.
        self._lock = threading.Lock()
        # Storage dictionary mapping cache key definitions.
        self._cache = {}

    # Fetch cached entries functions checks.
    def get(self, question: str, top_k: int, confidence_threshold: float) -> Optional[dict]:
        # Formulate query keys tuple format checks.
        key = (question.strip().lower(), top_k, confidence_threshold)
        # Lock thread resource block.
        with self._lock:
            # Fetch mappings inside collection.
            return self._cache.get(key)

    # Insert cache entry properties configs functions.
    def set(self, question: str, top_k: int, confidence_threshold: float, val: dict) -> None:
        # Key configurations.
        key = (question.strip().lower(), top_k, confidence_threshold)
        # Lock thread resources.
        with self._lock:
            # Set cache map values keys.
            self._cache[key] = val

    # Clear memory dictionary records cache mappings.
    def clear(self) -> None:
        # Lock threads.
        with self._lock:
            # Clear all dictionary keys values.
            self._cache.clear()
            # Inform logger confirmations.
            logger.info("Query cache invalidated.")

# Global cache instance setup.
query_cache = QueryCache()


# Request counts limiter class definition.
class InMemoryRateLimiter:
    """Thread-safe in-memory request rate limiter."""
    # Initialization configure properties setups.
    def __init__(self, requests_limit: int, window_seconds: int):
        # Limit boundary config.
        self.requests_limit = requests_limit
        # Interval window timing sizes.
        self.window_seconds = window_seconds
        # Collections logs mapping lists setup default lists.
        self.history = collections.defaultdict(list)
        # Synchronization threads locks parameters.
        self._lock = threading.Lock()
        
    # Check limit boundaries validations criteria rules.
    def is_rate_limited(self, key: str) -> bool:
        # Time track monotonic setups.
        now = time.time()
        # Lock resources.
        with self._lock:
            # Filter history records timeline outside active intervals.
            self.history[key] = [t for t in self.history[key] if now - t < self.window_seconds]
            # Verify total matches limits.
            if len(self.history[key]) >= self.requests_limit:
                # Returns True (limited).
                return True
            # Append current record.
            self.history[key].append(now)
            # Returns False (allowed).
            return False

# Limit limits instances configs.
query_limiter = InMemoryRateLimiter(requests_limit=30, window_seconds=60)
# Document uploads limiter.
upload_limiter = InMemoryRateLimiter(requests_limit=5, window_seconds=60)


# ── Authentication Dependency ──────────────────────────────────────────────────

# Header API key check validation function.
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Enforces API Key check if API_KEY is set in environment."""
    # Load expected API key string from configurations.
    expected_key = os.getenv("API_KEY")
    # Verify expected key config setup.
    if expected_key:
        # Key comparison validations check rules filters.
        if not x_api_key or x_api_key != expected_key:
            # Raise exception 401 unauthorized.
            raise HTTPException(
                # Code unauthorized status mapping.
                status_code=status.HTTP_401_UNAUTHORIZED,
                # Description error logs.
                detail="Invalid or missing API Key (X-API-Key header required).",
            )  # Raise end.


# App lifespan startup shutdown hooks wrapper.
@asynccontextmanager
# Lifespan execute wrapper.
async def lifespan(app: FastAPI):
    """Initialise heavy components once at startup."""
    # Declare globals scope access.
    global processor, retriever, generator, evaluator

    # Logging trace notifications.
    logger.info("Starting FinDoc Intelligence API …")
    # Initialize processor instances setups.
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=100)
    # Initialize retriever databases collection parameters setup paths.
    retriever = HybridRetriever(
        # Database path config.
        chroma_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        # Weight mapping search.
        semantic_weight=0.6,
        # BM25 factors.
        bm25_weight=0.4,
    )  # Retriever init end.
    # LLM Generator initialization attempts checks errors.
    try:
        # Generator initialization.
        generator = FinancialAnswerGenerator()
    # Catch environmental setup failures errors exceptions.
    except EnvironmentError as exc:
        # Logging warnings consoles messages.
        logger.warning(str(exc))
        # Re-assign generators references empty formats.
        generator = None  # type: ignore[assignment]

    # Initialize evaluator interfaces metrics trackers.
    evaluator = RAGAsEvaluator()
    # Logging messages status confirmations.
    logger.info("All components initialised.")
    # Yield control FastAPI lifecycles run states.
    yield
    # Shutdown logging alerts tracking messages console print.
    logger.info("Shutting down FinDoc Intelligence API.")


# ── App ────────────────────────────────────────────────────────────────────────

# Instantiate app structures configuration variables parameters.
app = FastAPI(
    # Name details.
    title="FinDoc Intelligence",
    # Text details.
    description="RAG-Based Financial Document Intelligence System for KPMG",
    # Version markers.
    version="1.0.0",
    # Lifetime hooks link options configurations.
    lifespan=lifespan,
)

# CORS setting policies mapping configurations.
app.add_middleware(
    # CORSMiddleware classes.
    CORSMiddleware,
    # Origins lists wildcard setups.
    allow_origins=["*"],
    # Credentials setups.
    allow_credentials=True,
    # Methods headers wildcards.
    allow_methods=["*"],
    # Headers maps.
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

# Query API input request parameters schemas BaseModel validation.
class QueryRequest(BaseModel):
    # Query text query parameter constraints.
    question: str = Field(..., min_length=5, description="Financial question to answer")
    # Limit candidates outputs selection count constraints.
    top_k: int = Field(3, ge=1, le=10, description="Number of source chunks to retrieve")
    # Confidence limits validations.
    confidence_threshold: float = Field(
        0.6, ge=0.0, le=1.0, description="Min retrieval score to proceed with LLM"
    )


# Source chunk formats metadata representation structures models.
class SourceChunk(BaseModel):
    # Parent filename tags.
    document: str
    # Reference page number tags.
    page: str | int
    # Category section.
    section: str
    # Text excerpts.
    excerpt: str
    # Matches values.
    relevance_score: float


# Query endpoint output response structure properties.
class QueryResponse(BaseModel):
    # Output answers string details.
    answer: str
    # Validation status parameters.
    confidence: str
    # Matches scores values metrics.
    retrieval_score: float
    # Sources list payload.
    source_chunks: List[SourceChunk]
    # Latency tracking.
    latency_ms: int


# Upload endpoint response status parameters formats schemas.
class UploadResponse(BaseModel):
    # Boolean success trace parameters.
    success: bool
    # File unique hashes.
    document_id: str
    # Filename tag properties maps.
    document_name: str
    # Total segments index counts.
    chunks_created: int
    # Total file pages.
    total_pages: int


# Document metadata summaries info representation format schema.
class DocumentInfo(BaseModel):
    # Unique database doc index strings properties.
    document_id: str
    # Filename indicators.
    document_name: str
    # Chunk sizes counts metrics.
    chunk_count: int
    # Timestamps track details format configurations path settings.
    uploaded_at: str


# Evaluation metrics analytics summary outputs schemas structures configs.
class EvalMetrics(BaseModel):
    # Context scores.
    context_precision: float
    # Ground truth comparisons.
    faithfulness: float
    # Relevancy indicators.
    answer_relevancy: float
    # Abstain tracking ratios.
    hallucination_rate: float
    # Average speed latency metrics.
    avg_latency_ms: float
    # Total count records processed.
    total_queries: int
    # Timestamp updates references tracking logs.
    last_evaluated: Optional[str]
    # Source tagging.
    source: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

# Health check route path definition FastAPI get method parameters.
@app.get("/health", tags=["System"])
# Health checks function.
async def health_check():
    """Returns system health and component statuses."""
    # Returns status dictionary payload information updates details.
    return {
        # Standard status.
        "status": "healthy",
        # Version numbers.
        "version": "1.0.0",
        # Sub components maps lists validations indicators setups.
        "components": {
            # Vector database status.
            "vector_store": "ready",
            # Total chunks size indicators trackers configurations.
            "total_chunks": retriever.chunk_count(),
            # LLM status checks flags validations options definitions.
            "llm": "ready" if generator else "no_api_key",
        },  # Components map end.
    }  # Return end.


# Upload API post endpoint definitions properties path setup options.
@app.post(
    # Path mappings.
    "/upload",
    # Out formats schema structures.
    response_model=UploadResponse,
    # HTTP code status maps checks.
    status_code=status.HTTP_200_OK,
    # Visual categorised labels.
    tags=["Documents"],
    # Authorization validation hooks configurations parameters.
    dependencies=[Depends(verify_api_key)],
)  # Upload routes path end.
# Ingest controller execution function mapping details uploads.
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a financial PDF and ingest it into the RAG pipeline.
    """
    # Rate limit check
    # Fetch client IP addresses headers records details.
    ip = getattr(request, "client", None)
    # Validate IP address mappings check strings.
    client_ip = ip.host if ip else "unknown"
    # Rate limit check triggers validations options.
    if upload_limiter.is_rate_limited(client_ip):
        # Limit exception raise codes alert mapping.
        raise HTTPException(
            # Code HTTP 429.
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            # Description text logs messages.
            detail="Too many document uploads. Limit is 5 uploads per minute.",
        )  # Rate limit exception end.

    # Validate file type
    # File format type matching validations check rules sequence.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        # Format exception raise status 400.
        raise HTTPException(
            # HTTP 400.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Error logs messages print.
            detail="Only PDF files are accepted.",
        )  # File type validation check end.

    # Check file size (50 MB cap)
    # Define sizes limit.
    MAX_SIZE = 50 * 1024 * 1024
    # Read binary bytes content.
    content = await file.read()
    # Check limit check validations.
    if len(content) > MAX_SIZE:
        # Size exception raise status 413.
        raise HTTPException(
            # HTTP 413.
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            # Size messages descriptions.
            detail=f"File too large. Maximum size is 50 MB.",
        )  # Size check end.

    # Resolve paths & create directories
    # Filename references setups.
    doc_name = file.filename
    # Generate stable IDs hashes.
    doc_id = processor._generate_doc_id(doc_name)
    # Save folder directory paths setups.
    raw_dir = os.path.join("data", "raw")
    # Directory builders paths os levels setups options.
    os.makedirs(raw_dir, exist_ok=True)
    # File targets paths configs mappings.
    pdf_path = os.path.join(raw_dir, doc_name)

    # Overwrite if exists, but delete from database first to avoid orphaned chunks
    # Database previous records checks delete triggers.
    retriever.delete_document(doc_id)

    # Persist raw PDF
    # Open target file writes stream.
    with open(pdf_path, "wb") as f:
        # Write contents payload bytes.
        f.write(content)

    # Ingestion pipeline process loops catches exception errors routes.
    try:
        # Logging trace writes console.
        logger.info(f"Processing upload: {doc_name}")
        # Parse processors triggers execution functions logic updates.
        result = processor.process_document(pdf_path, doc_name)

        # Process results status checks.
        if not result["success"]:
            # Delete physical raw file check paths exists.
            if os.path.exists(pdf_path):
                # Delete files.
                os.remove(pdf_path)
            # Unprocessable entities status exception raise 422.
            raise HTTPException(
                # HTTP 422.
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                # Details logging descriptions.
                detail=f"PDF processing failed: {result.get('error', 'Unknown error')}",
            )  # Failure check end.

        # Add chunks to retriever
        # Vector database write execution indexing.
        success = retriever.add_chunks(result["chunks"])
        # Database status check confirms.
        if not success:
            # Clean local files traces check paths setup options.
            if os.path.exists(pdf_path):
                # Delete file.
                os.remove(pdf_path)
            # Internal server error exception raise 500.
            raise HTTPException(
                # HTTP 500.
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                # Error descriptions mapping logs.
                detail="Failed to index document chunks.",
            )  # Index check end.

        # Invalidate response cache
        # Reset cache structures.
        query_cache.clear()

        # Success upload responses schema mapping objects returns parameters.
        return UploadResponse(
            # Flag set True.
            success=True,
            # Document ID.
            document_id=result["document_id"],
            # Filename.
            document_name=result["document_name"],
            # Chunks count.
            chunks_created=result["total_chunks"],
            # Page limits.
            total_pages=result["total_pages"],
        )  # Response end.

    # HTTPException catches forward handlers.
    except HTTPException:
        # Reraise exception.
        raise
    # Generic anomalies catches traces cleanups.
    except Exception as e:
        # Verify files existence tracks cleanup.
        if os.path.exists(pdf_path):
            # Delete file.
            os.remove(pdf_path)
        # Server error status raises.
        raise HTTPException(
            # HTTP 500.
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            # Error mapping string messages details.
            detail=f"Upload failed: {str(e)}"
        )  # Exception raise end.


# RAG query post API route definitions parameters path setup configurations.
@app.post(
    # Path coordinates.
    "/query",
    # Outputs validations schemas.
    response_model=QueryResponse,
    # Tags markers.
    tags=["RAG"],
    # Credentials dependencies mapping checks configurations.
    dependencies=[Depends(verify_api_key)],
)  # Query path end.
# Answers generation controller execution mapping function details query.
async def query_documents(request: Request, query_req: QueryRequest):
    """
    Answer a financial question using RAG.
    """
    # Rate limit check
    # Fetch client IP.
    ip = getattr(request, "client", None)
    # Check IP strings.
    client_ip = ip.host if ip else "unknown"
    # Limit check validations code.
    if query_limiter.is_rate_limited(client_ip):
        # Raise HTTP 429 status.
        raise HTTPException(
            # HTTP 429.
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            # Messages alert details logs console.
            detail="Too many queries. Limit is 30 queries per minute.",
        )  # Limit check end.

    # LLM client availability checks indicators verify.
    if generator is None:
        # LLM not set status raises.
        raise HTTPException(
            # Service unavailable 503.
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            # Log warnings configuration instructions.
            detail="LLM not available. Set GROQ_API_KEY in your .env file.",
        )  # LLM missing check end.

    # Indexed database count validation check code.
    if retriever.chunk_count() == 0:
        # Empty database check status raises.
        raise HTTPException(
            # HTTP 400.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Details instructions updates.
            detail="No documents have been uploaded yet. Please upload a PDF first.",
        )  # Empty check end.

    # 1. Check Query Cache
    # Fetch memory cache response if matches criteria.
    cached_res = query_cache.get(query_req.question, query_req.top_k, query_req.confidence_threshold)
    # Cache hit indicators verify.
    if cached_res:
        # Logging trace hit confirmations console messages updates.
        logger.info(f"Query Cache Hit: '{query_req.question}'")
        # Returns parsed query responses layouts properties.
        return QueryResponse(**cached_res)

    # 2. Retrieve
    # Matches context search vectors algorithms execute.
    context_chunks = retriever.hybrid_search(
        # Question query text.
        query_req.question, top_k=query_req.top_k
    )  # Retrieval end.

    # 3. Generate
    # Model APIs connections execute catch failures blocks.
    try:
        # Generation methods call.
        response = generator.generate(
            # Questions text.
            question=query_req.question,
            # Retrieval context chunks.
            context_chunks=context_chunks,
            # Threshold score validations overrides.
            confidence_threshold=query_req.confidence_threshold,
        )  # Generator end.
    # Run time errors exceptions catch parameters.
    except RuntimeError as exc:
        # Server errors raise configurations.
        raise HTTPException(
            # HTTP 500.
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )  # Run time error check end.

    # Record for evaluation
    # Logs evaluation metrics recording update parameters.
    evaluator.record_query(
        # Question.
        question=query_req.question,
        # Answer content.
        answer=response["answer"],
        # Sources.
        source_chunks=response["source_chunks"],
        # Confidence label.
        confidence=response["confidence"],
        # Duration speed.
        latency_ms=response["latency_ms"],
    )  # Record end.

    # Cache response
    # Save results inside memory cache mappings dictionaries.
    query_cache.set(query_req.question, query_req.top_k, query_req.confidence_threshold, response)

    # Outputs returns.
    return QueryResponse(**response)


# Evaluation metrics get route FastAPI endpoint mapping path options setup.
@app.get(
    # Path mappings.
    "/eval-metrics",
    # Outputs models schemas.
    response_model=EvalMetrics,
    # Tags.
    tags=["Evaluation"],
    # Credentials validations configurations.
    dependencies=[Depends(verify_api_key)],
)  # Eval path end.
# Evaluation metrics fetch controller execution function.
async def get_eval_metrics():
    """Return aggregated evaluation metrics for all queries processed so far."""
    # Compute heuristic metric status averages properties mappings.
    metrics = evaluator.get_metrics()
    # Returns formatted evaluations models details.
    return EvalMetrics(**metrics)


# List documents get route FastAPI endpoint path definitions options.
@app.get(
    # Path.
    "/documents",
    # Tags labels.
    tags=["Documents"],
    # Authorization checks.
    dependencies=[Depends(verify_api_key)],
)  # Documents route path end.
# List documents indexed controller execution functions details.
async def list_documents():
    """List all documents currently indexed in the system."""
    # Fetch details lists from vector search database retriever interfaces.
    docs = retriever.list_documents()
    # Return formatted payload map lists confirmations logs.
    return {
        # Doc count total metrics.
        "total_documents": len(docs),
        # Chunks count.
        "total_chunks": retriever.chunk_count(),
        # Document items arrays collections structures data maps.
        "documents": docs,
    }  # Return end.


# Delete document endpoint FastAPI delete mapping paths parameters setup.
@app.delete(
    # Path parameters ID bindings.
    "/documents/{document_id}",
    # Tags label.
    tags=["Documents"],
    # Auth validations.
    dependencies=[Depends(verify_api_key)],
)  # Delete paths end.
# Delete document controller execution function details mappings.
async def delete_document(document_id: str):
    """Delete a document and all its associated chunks from the system."""
    # Find matching document name in list of documents to delete raw file
    # Load indexes.
    docs = retriever.list_documents()
    # Name track initialization placeholder.
    doc_name = None
    # Iteration search loop matching index values.
    for d in docs:
        # ID check verification matches properties.
        if d["document_id"] == document_id:
            # Reassign filename indicator maps tags.
            doc_name = d["document_name"]
            # Escape loops.
            break

    # Database updates triggers deletions routines methods.
    deleted_chunks = retriever.delete_document(document_id)
    # Deletion results validation checks status.
    if deleted_chunks == 0:
        # HTTP 404 Exception raise if not matches found.
        raise HTTPException(
            # HTTP 404.
            status_code=status.HTTP_404_NOT_FOUND,
            # Error logs messages console prints.
            detail=f"Document with id '{document_id}' not found.",
        )  # Missing check end.

    # Delete raw PDF
    # Verify filename references.
    if doc_name:
        # File paths constructs.
        pdf_path = os.path.join("data", "raw", doc_name)
        # Verify path checks parameters.
        if os.path.exists(pdf_path):
            # Try file deletion operations routines checking blocks.
            try:
                # Remove file.
                os.remove(pdf_path)
                # Logging details confirmations status.
                logger.info(f"Deleted raw PDF file: {pdf_path}")
            # Catch deletion exceptions logs parameters updates trackers.
            except Exception as e:
                # Log console warning tracks.
                logger.error(f"Failed to delete raw PDF {pdf_path}: {e}")

    # Invalidate cache
    # Reset caches mapping arrays structures.
    query_cache.clear()

    # Confirmations output map.
    return {"success": True, "document_id": document_id, "deleted_chunks": deleted_chunks}


# Ragas evaluation automation query post API route mapping settings paths.
@app.post(
    # Path targets.
    "/eval/run-ragas",
    # Tags markers.
    tags=["Evaluation"],
    # Authorization checks configuration parameters options dependencies.
    dependencies=[Depends(verify_api_key)],
)  # Run Ragas route end.
# Evaluation runs controllers.
async def run_full_ragas(test_dataset_path: Optional[str] = None):
    """
    Run the full RAGAs evaluation suite.
    """
    # Dataset path evaluation.
    dataset_path = test_dataset_path or "./data/processed/eval_dataset.json"
    # Verify file paths exists constraints.
    if not os.path.exists(dataset_path):
        # Raise HTTP 404.
        raise HTTPException(
            # HTTP 404.
            status_code=status.HTTP_404_NOT_FOUND,
            # Messages descriptions.
            detail=f"Evaluation dataset not found: {dataset_path}"
        )  # Path check end.

    # Parse JSON datasets files streams catch exceptions codes.
    try:
        # Open data file.
        with open(dataset_path) as f:
            # Parse load datasets values.
            dataset = json.load(f)
    # Catch load anomalies.
    except Exception as e:
        # HTTP 400 error status raises.
        raise HTTPException(
            # HTTP 400.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Descriptions details updates trackings.
            detail=f"Failed to load dataset: {str(e)}"
        )  # Dataset load end.

    # Question list extract options configuration parameters mappings values.
    questions = dataset.get("questions", [])
    # Ground truth list extracts mappings setups configs.
    ground_truths = dataset.get("ground_truths", [])

    # Empty validation arrays filters.
    if not questions:
        # HTTP 400 error raises.
        raise HTTPException(
            # HTTP 400.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Alert detail descriptions.
            detail="Evaluation dataset is empty."
        )  # Questions count check end.

    # Vector database count limit validations checks.
    if retriever.chunk_count() == 0:
        # Empty collection check status raises.
        raise HTTPException(
            # HTTP 400.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Messages console descriptions.
            detail="No documents indexed. Evaluation requires uploaded reports."
        )  # Database count check end.

    # 1. Populate contexts and answers dynamically from retriever and generator
    # Context trackers list placeholder.
    populated_contexts = []
    # Answer list trackers.
    populated_answers = []

    # Logger tracks progress writes.
    logger.info(f"Generating answers and contexts for {len(questions)} evaluation questions...")
    # Iteration list loop sequence calculations updates.
    for q in questions:
        # Retrieve
        # Vector database search candidates.
        chunks = retriever.hybrid_search(q, top_k=3)
        # Texts extraction maps checks.
        chunk_texts = [c["text"] for c in chunks]
        # Append lists.
        populated_contexts.append(chunk_texts)

        # Generate
        # LLM client checker check setup settings.
        if generator:
            # Generation blocks.
            try:
                # Answer generator triggers calls configurations parameters.
                gen_res = generator.generate(
                    # Question text.
                    question=q,
                    # Context elements.
                    context_chunks=chunks,
                    # Bypass confidence guard threshold checks.
                    confidence_threshold=0.0
                )  # Generation end.
                # Save answer elements.
                populated_answers.append(gen_res["answer"])
            # Catch LLM connection error trace.
            except Exception as e:
                # Logger console updates errors warnings details.
                logger.error(f"Gen failed for eval question '{q}': {e}")
                # Save error placeholder string.
                populated_answers.append("Error generating answer.")
        # Fallback LLM unavailable scenario setups.
        else:
            # Set placeholder indicators.
            populated_answers.append("LLM not available.")

    # Save populated dataset
    # Formulate populated dataset schemas data values.
    populated_data = {
        # Questions list.
        "questions": questions,
        # Ground truths list.
        "ground_truths": ground_truths,
        # Context list maps.
        "contexts": populated_contexts,
        # Answers list.
        "answers": populated_answers
    }  # Data schema end.

    # Set save file target coordinates.
    populated_path = "./data/processed/eval_dataset_populated.json"
    # Open write file stream.
    with open(populated_path, "w") as f:
        # Dump mappings formats spacing indentation.
        json.dump(populated_data, f, indent=2)

    # Success logging confirmations path parameters.
    logger.info(f"Populated dataset saved to {populated_path}")

    # 2. Run evaluator on populated dataset
    # Trigger full Ragas suite calculations methods executions.
    metrics = evaluator.run_ragas(populated_path)
    # Check evaluation results validity flags.
    if metrics is None:
        # Failures status exception raises HTTP 500.
        raise HTTPException(
            # HTTP 500.
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            # Descriptions logs messages.
            detail=(
                "RAGAs evaluation failed. Ensure GROQ_API_KEY or OPENAI_API_KEY is set."
            ),  # Message details end.
        )  # Exception end.
    # Returns calculation outputs mappings objects.
    return metrics


# Entrypoint check.
if __name__ == "__main__":
    # Import uvicorn server package.
    import uvicorn
    # Execute uvicorn server mapping setups options host ports reloading paths.
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
