# Yeh file hybrid retrieval (semantic aur keyword search combine) system set up karti hai.
"""
Hybrid Retriever — Combines semantic (ChromaDB) and BM25 keyword search.

Architecture:
  1. Semantic search  → sentence-transformers embeddings + cosine similarity
  2. BM25 keyword     → rank-bm25 over stored chunk texts
  3. Score fusion     → 60% semantic + 40% BM25
  4. (Optional) Rerank → cross-encoder for precision boost

The retriever maintains an in-memory BM25 index that mirrors what's stored in
ChromaDB. On startup it can reload the BM25 index from ChromaDB so state
survives restarts.
"""

# Future import type annotations support dynamic declarations.
from __future__ import annotations

# Thread lock implementations support thread safety.
import threading
# Dictionary structure parameters typing helper definitions.
from typing import Dict, List, Optional

# Database client library chromadb.
import chromadb
# Math vector metrics processing library numpy.
import numpy as np
# Program flow traces messages tracking.
from loguru import logger
# Text rank algorithm rank-bm25 module.
from rank_bm25 import BM25Okapi
# Sentence encoders and transformers packages loading.
from sentence_transformers import CrossEncoder, SentenceTransformer


# Semantic aur keyword matches run karne ki main class.
class HybridRetriever:
    """
    Hybrid semantic + BM25 retriever backed by ChromaDB.

    Thread-safe: a threading.Lock protects BM25 index mutations.
    """

    # BM25 output ranking score boundary limit parameters.
    _BM25_NORM_CAP = 10.0

    # Retriever instance initialization configurations.
    def __init__(
        self,
        # Chroma vector DB path setup.
        chroma_path: str = "./chroma_db",
        # Vector DB collection group labels.
        collection_name: str = "financial_documents",
        # Default text embeddings model.
        embedding_model: str = "all-MiniLM-L6-v2",
        # Precision reranker setup models definition.
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L6-v2",
        # Semantic search score weight.
        semantic_weight: float = 0.6,
        # BM25 keyword score weight.
        bm25_weight: float = 0.4,
    ):  # Init declaration end.
        # Startup status logs console print track.
        logger.info("Initialising HybridRetriever …")

        # Embedding model loading method initialization.
        self.embedding_model = SentenceTransformer(embedding_model)
        # Log successful embeddings loading messages.
        logger.info(f"Loaded embedding model: {embedding_model}")

        # Reranker model loader setups.
        self.reranker: Optional[CrossEncoder] = None
        # Reranker configure values exists criteria validation check.
        if reranker_model:
            # Model loader attempt catches errors.
            try:
                # Class mapping.
                self.reranker = CrossEncoder(reranker_model)
                # Confirm loader status logging console.
                logger.info(f"Loaded reranker: {reranker_model}")
            # Catch initialization failures exception alerts prints.
            except Exception as exc:
                # Warning logs write tracks.
                logger.warning(f"Could not load reranker ({exc}). Skipping reranking.")

        # Chroma DB client load setups.
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        # Access collection object database parameters maps.
        self.collection = self.chroma_client.get_or_create_collection(
            # Collection name labels configs.
            name=collection_name,
            # Similarity evaluation space metric configs.
            metadata={"hnsw:space": "cosine"},
        )  # Collection init end.
        # Log information confirmation of Chroma initialization.
        logger.info(
            f"ChromaDB collection '{collection_name}' ready "
            f"({self.collection.count()} existing chunks)"
        )  # Print log end.

        # BM25 internals settings trackers thread synchronization lock setup.
        self._lock = threading.Lock()
        # Chunks raw text records collection placeholder list.
        self._bm25_texts: List[str] = []
        # Chunks hash string IDs tracker list.
        self._bm25_ids: List[str] = []
        # Chunks properties map configurations list.
        self._bm25_metadatas: List[Dict] = []
        # Active index objects maps setup.
        self._bm25_index: Optional[BM25Okapi] = None
        # Save score weights properties assignments.
        self.semantic_weight = semantic_weight
        # Save keyword weights settings configs.
        self.bm25_weight = bm25_weight

        # Rebuild indexes from existing data storage on startup.
        self._rebuild_bm25_from_chroma()

    # ── Public API ────────────────────────────────────────────────────────────

    # Insert document chunks methods definitions.
    def add_chunks(self, chunks: List[Dict]) -> bool:
        """
        Add document chunks to both ChromaDB and the BM25 index.
        """
        # Empty array checks validations.
        if not chunks:
            # Returns positive status signal.
            return True

        # Process databases writes inside try catch exceptions checks blocks.
        try:
            # Extract texts collections.
            texts = [c["text"] for c in chunks]
            # Extract IDs collections.
            ids = [c["id"] for c in chunks]
            # Extract metadata dictionaries elements.
            metadatas = [c["metadata"] for c in chunks]

            # Ingest embeddings compute stage.
            logger.info(f"Embedding {len(chunks)} chunks …")
            # Text strings vector translations encoder execution.
            embeddings = self.embedding_model.encode(
                # Text inputs progress tracking batch size options.
                texts, show_progress_bar=True, batch_size=32
            ).tolist()  # Encoder end.

            # Vector collection writes.
            self.collection.upsert(
                # IDs strings mappings.
                ids=ids,
                # Computed vectors.
                embeddings=embeddings,
                # String contents.
                documents=texts,
                # Properties dictionaries maps.
                metadatas=metadatas,
            )  # Upsert end.
            # Success trace console log.
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB")

            # Thread sync updates BM25 collections list items logs.
            with self._lock:
                # Add text records.
                self._bm25_texts.extend(texts)
                # Add ID strings.
                self._bm25_ids.extend(ids)
                # Add metadata maps.
                self._bm25_metadatas.extend(metadatas)
                # Recompile indices logic calls.
                self._rebuild_bm25()

            # Output confirmations.
            return True

        # Exception catches trace failures.
        except Exception as exc:
            # Write errors log database console prints.
            logger.error(f"add_chunks failed: {exc}")
            # Returns negative checks signals.
            return False

    # Main search hybrid matching algorithm retrieve methods configurations.
    def hybrid_search(
        self,
        # Question text query.
        query: str,
        # Outputs segments count limit parameters.
        top_k: int = 3,
        # Candidate selections limits check boundaries properties.
        n_candidates: int = 20,
        # precision rerank execution checks.
        use_reranker: bool = True,
    ) -> List[Dict]:  # Search method end.
        """
        Perform hybrid retrieval and return the top-k most relevant chunks.
        """
        # Empty collection check validations logic exits.
        if self.collection.count() == 0:
            # Warning traces prints.
            logger.warning("Collection is empty — no chunks to retrieve.")
            # Returns empty list.
            return []

        # Candidate bounds adjustments.
        n_candidates = min(n_candidates, self.collection.count())
        # Target final bounds calculations.
        top_k = min(top_k, n_candidates)

        # ── Step 1: Semantic search ──────────────────────────────────────────
        # Encode query string vector translations mappings.
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        # Query vector database match retrieval filters configurations.
        semantic_results = self.collection.query(
            # Target query vector.
            query_embeddings=[query_embedding],
            # Matches selections limit.
            n_results=n_candidates,
            # Properties inclusions.
            include=["documents", "metadatas", "distances"],
        )  # Chroma query end.

        # Candidates temporary storage maps mappings.
        candidates: Dict[str, Dict] = {}
        # Iterate matches elements indices loop properties updates.
        for i, doc_id in enumerate(semantic_results["ids"][0]):
            # Vector distances measure checks values.
            distance = semantic_results["distances"][0][i]
            # Convert cosine distance maps to similarity score formulas.
            semantic_score = float(1.0 - distance)
            # Candidate structure updates configs mappings keys assignments.
            candidates[doc_id] = {
                # Document string texts.
                "text": semantic_results["documents"][0][i],
                # Document properties.
                "metadata": semantic_results["metadatas"][0][i],
                # Similarity parameters.
                "semantic_score": semantic_score,
                # Default empty keyword score weights setup.
                "bm25_score": 0.0,
            }  # Candidates items end.

        # ── Step 2: BM25 search ──────────────────────────────────────────────
        # Thread sync locks checks BM25 memory searches operations.
        with self._lock:
            # Verify index availability status verification code.
            if self._bm25_index is not None:
                # Text parse tokenizer matching check evaluate.
                bm25_scores = self._bm25_index.get_scores(query.split())
                # Range limits scale normalizations calculations.
                min_score = bm25_scores.min()
                # Max scale limits.
                max_score = bm25_scores.max()
                # Boundary comparisons checks filters logic structure.
                if max_score > min_score:
                    # Norm score values conversions mappings scale formulas.
                    bm25_scores_norm = (bm25_scores - min_score) / (max_score - min_score)
                # Static score scenarios options mappings.
                else:
                    # All match check verification flags set variables.
                    bm25_scores_norm = np.ones_like(bm25_scores) if max_score != 0.0 else np.zeros_like(bm25_scores)

                # Sorting candidates indices highest first.
                top_bm25_indices = np.argsort(bm25_scores_norm)[::-1][:n_candidates]
                # Iterate candidates indices map configs collections list properties.
                for idx in top_bm25_indices:
                    # Bounds checking safety verify rules filters.
                    if idx >= len(self._bm25_ids):
                        # Out of bounds escape skips.
                        continue
                    # Unique chunk ID fetch.
                    doc_id = self._bm25_ids[idx]
                    # Candidates exists verification filters.
                    if doc_id not in candidates:
                        # Append new document values configurations setups.
                        candidates[doc_id] = {
                            # Texts segments.
                            "text": self._bm25_texts[idx],
                            # Metadata tags.
                            "metadata": self._bm25_metadatas[idx],
                            # Default semantic weight properties setup.
                            "semantic_score": 0.0,
                            # Default keyword weight setup.
                            "bm25_score": 0.0,
                        }  # Candidates insert end.
                    # Update keyword normal score mappings outputs configurations.
                    candidates[doc_id]["bm25_score"] = float(bm25_scores_norm[idx])

        # ── Step 3: Score fusion ─────────────────────────────────────────────
        # Merged data sets arrays trackers.
        fused_results = []
        # Merge iterations loops properties evaluations.
        for doc_id, data in candidates.items():
            # Combine weights calculations metrics formulas.
            combined = (
                # Semantic weight factors.
                self.semantic_weight * data["semantic_score"]
                # Keyword weight factors.
                + self.bm25_weight * data["bm25_score"]
            )  # Combined weight end.
            # Append result dictionary mappings parameters options details inside arrays.
            fused_results.append(
                {
                    # IDs strings.
                    "doc_id": doc_id,
                    # Texts.
                    "text": data["text"],
                    # Precision round combined scores properties parameters.
                    "score": round(combined, 4),
                    # Metadata configs references.
                    "metadata": data["metadata"],
                }
            )  # Fused inserts.

        # Sort combined results descending score parameters rules.
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        # Select best matches candidate limits check.
        top_candidates = fused_results[:n_candidates]

        # ── Step 4: Optional reranking ───────────────────────────────────────
        # Check reranker credentials availability flags validations.
        if use_reranker and self.reranker and len(top_candidates) > 1:
            # Precision reranker score mapping execution triggers method logic.
            top_candidates = self._rerank(query, top_candidates, top_k)
        # Fallback normal segment select crop bounds setups.
        else:
            # Crop bounds candidates size arrays.
            top_candidates = top_candidates[:top_k]

        # Log completion checks details mapping.
        logger.info(
            f"hybrid_search → returning {len(top_candidates)} chunks "
            f"(top score={top_candidates[0]['score'] if top_candidates else 0:.3f})"
        )  # Log end.
        # Returns candidate collections lists.
        return top_candidates

    # Document chunks delete updates methods configurations metrics definitions.
    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document from ChromaDB and BM25 index.
        """
        # Search targets records inside databases matching document IDs.
        results = self.collection.get(
            # ID match checks.
            where={"document_id": document_id},
            # Properties inclusions.
            include=["documents", "metadatas"],
        )  # Get chunks end.
        # IDs extraction maps.
        chunk_ids = results["ids"]
        # Empty verify check exits.
        if not chunk_ids:
            # Returns 0 deleted.
            return 0

        # Delete database elements lists references.
        self.collection.delete(ids=chunk_ids)
        # Successful logs write trackings configurations.
        logger.info(f"Deleted {len(chunk_ids)} chunks for document_id={document_id}")

        # BM25 state rebuild execution synchronize configurations paths.
        self._rebuild_bm25_from_chroma()
        # Returns total counts.
        return len(chunk_ids)

    # List documents indexed inside database metadata helper method.
    def list_documents(self) -> List[Dict]:
        """
        Return a summary of all documents currently in the collection.
        """
        # Retrieve all items properties from database collection mapping.
        results = self.collection.get(include=["metadatas"])
        # Document maps trace.
        doc_map: Dict[str, Dict] = {}
        # Iterate metadata maps properties list mappings checks parameters.
        for meta in results["metadatas"]:
            # ID checks.
            did = meta.get("document_id", "unknown")
            # Unique ID entries validation filters tracks configurations.
            if did not in doc_map:
                # Add document information formats details.
                doc_map[did] = {
                    # ID field.
                    "document_id": did,
                    # Filename maps.
                    "document_name": meta.get("document_name", "unknown"),
                    # Counter chunks initialization.
                    "chunk_count": 0,
                    # Upload timing mappings flags configurations settings paths.
                    "uploaded_at": meta.get("upload_timestamp", ""),
                }  # doc_map index init end.
            # Increment counts.
            doc_map[did]["chunk_count"] += 1  # count update.
        # Outputs collections list conversions returns.
        return list(doc_map.values())

    # Total counts segments metadata return helper methods.
    def chunk_count(self) -> int:
        """Total number of chunks currently stored."""
        # Database counts return operations logic execution maps.
        return self.collection.count()

    # ── Private helpers ───────────────────────────────────────────────────────

    # Rebuild BM25 index from local text list lists helper methods.
    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from the in-memory text list. Caller must hold lock."""
        # Empty text list validations checks.
        if not self._bm25_texts:
            # Set index null definitions.
            self._bm25_index = None
            # Return codes check sequences exits.
            return
        # Tokenizer formatting split conversions mapping arrays.
        tokenized = [text.split() for text in self._bm25_texts]
        # Re-initialize index builder.
        self._bm25_index = BM25Okapi(tokenized)

    # Rebuild BM25 index from vector database Chroma collections helper method.
    def _rebuild_bm25_from_chroma(self) -> None:
        """
        Load all documents from ChromaDB and rebuild the BM25 index.
        """
        # Count collection elements inside database configs.
        count = self.collection.count()
        # Empty database check validations logic signals.
        if count == 0:
            # Information logging trace.
            logger.info("ChromaDB collection is empty — BM25 index not built.")
            # Returns code sequences.
            return

        # Start rebuilding index log traces prints configs.
        logger.info(f"Rebuilding BM25 index from {count} ChromaDB chunks …")
        # Ingest all records from database collection parameters.
        results = self.collection.get(include=["documents", "metadatas"])
        # Thread sync locks variables check update parameters operations.
        with self._lock:
            # Sync IDs.
            self._bm25_ids = results["ids"]
            # Sync texts string.
            self._bm25_texts = results["documents"]
            # Sync metadata configs.
            self._bm25_metadatas = results["metadatas"]
            # Index compilation execute call.
            self._rebuild_bm25()
        # Information logging confirmation track.
        logger.info("BM25 index ready.")

    # Cross-encoder precision rerank executions logic helper methods.
    def _rerank(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:  # Rerank method end.
        """
        Rerank candidates using a cross-encoder and return top_k.
        """
        # Create input tuple query text passage checks pairs.
        pairs = [(query, c["text"]) for c in candidates]
        # Compute relevance scores using cross encoder predictor model execution.
        ce_scores = self.reranker.predict(pairs)  # type: ignore[union-attr]
        # Re-assign calculated scores inside candidates map settings loop properties.
        for i, score in enumerate(ce_scores):
            # Decimal precision round configuration limits check updates.
            candidates[i]["score"] = round(float(score), 4)
        # Sort candidates descending score parameters rules.
        candidates.sort(key=lambda x: x["score"], reverse=True)
        # Select best matches return.
        return candidates[:top_k]
