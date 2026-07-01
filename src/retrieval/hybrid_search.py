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

from __future__ import annotations

import threading
from typing import Dict, List, Optional

import chromadb
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class HybridRetriever:
    """
    Hybrid semantic + BM25 retriever backed by ChromaDB.

    Thread-safe: a threading.Lock protects BM25 index mutations.
    """

    # BM25 normalisation cap — prevents a single very high BM25 score from
    # dominating the combined ranking.
    _BM25_NORM_CAP = 10.0

    def __init__(
        self,
        chroma_path: str = "./chroma_db",
        collection_name: str = "financial_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L6-v2",
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        logger.info("Initialising HybridRetriever …")

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Loaded embedding model: {embedding_model}")

        # Cross-encoder reranker (optional)
        self.reranker: Optional[CrossEncoder] = None
        if reranker_model:
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info(f"Loaded reranker: {reranker_model}")
            except Exception as exc:
                logger.warning(f"Could not load reranker ({exc}). Skipping reranking.")

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}' ready "
            f"({self.collection.count()} existing chunks)"
        )

        # BM25 index (rebuilt from ChromaDB on init)
        self._lock = threading.Lock()
        self._bm25_texts: List[str] = []
        self._bm25_ids: List[str] = []
        self._bm25_metadatas: List[Dict] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self._rebuild_bm25_from_chroma()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict]) -> bool:
        """
        Add document chunks to both ChromaDB and the BM25 index.

        Args:
            chunks: List of chunk dicts — each must have "id", "text", "metadata".

        Returns:
            True on success, False on failure.
        """
        if not chunks:
            return True

        try:
            texts = [c["text"] for c in chunks]
            ids = [c["id"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            # Embed in batches
            logger.info(f"Embedding {len(chunks)} chunks …")
            embeddings = self.embedding_model.encode(
                texts, show_progress_bar=True, batch_size=32
            ).tolist()

            # Upsert to ChromaDB (handles duplicates gracefully)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB")

            # Update BM25 index (thread-safe)
            with self._lock:
                self._bm25_texts.extend(texts)
                self._bm25_ids.extend(ids)
                self._bm25_metadatas.extend(metadatas)
                self._rebuild_bm25()

            return True

        except Exception as exc:
            logger.error(f"add_chunks failed: {exc}")
            return False

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        n_candidates: int = 20,
        use_reranker: bool = True,
    ) -> List[Dict]:
        """
        Perform hybrid retrieval and return the top-k most relevant chunks.

        Args:
            query:        User query string.
            top_k:        Number of final chunks to return.
            n_candidates: Number of candidates from each search method.
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            List of chunk dicts sorted by relevance score (descending).
            Each dict: {"doc_id", "text", "score", "metadata"}
        """
        if self.collection.count() == 0:
            logger.warning("Collection is empty — no chunks to retrieve.")
            return []

        n_candidates = min(n_candidates, self.collection.count())
        top_k = min(top_k, n_candidates)

        # ── Step 1: Semantic search ──────────────────────────────────────────
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        candidates: Dict[str, Dict] = {}
        for i, doc_id in enumerate(semantic_results["ids"][0]):
            distance = semantic_results["distances"][0][i]
            # Cosine distance → similarity (ChromaDB returns distances)
            semantic_score = float(1.0 - distance)
            candidates[doc_id] = {
                "text": semantic_results["documents"][0][i],
                "metadata": semantic_results["metadatas"][0][i],
                "semantic_score": semantic_score,
                "bm25_score": 0.0,
            }

        # ── Step 2: BM25 search ──────────────────────────────────────────────
        with self._lock:
            if self._bm25_index is not None:
                bm25_scores = self._bm25_index.get_scores(query.split())
                # Normalise BM25 scores to [0, 1]
                min_score = bm25_scores.min()
                max_score = bm25_scores.max()
                if max_score > min_score:
                    bm25_scores_norm = (bm25_scores - min_score) / (max_score - min_score)
                else:
                    # If all documents have the same score, check if there was any match (non-zero score)
                    bm25_scores_norm = np.ones_like(bm25_scores) if max_score != 0.0 else np.zeros_like(bm25_scores)

                # Top-n BM25 candidates
                top_bm25_indices = np.argsort(bm25_scores_norm)[::-1][:n_candidates]
                for idx in top_bm25_indices:
                    if idx >= len(self._bm25_ids):
                        continue
                    doc_id = self._bm25_ids[idx]
                    if doc_id not in candidates:
                        candidates[doc_id] = {
                            "text": self._bm25_texts[idx],
                            "metadata": self._bm25_metadatas[idx],
                            "semantic_score": 0.0,
                            "bm25_score": 0.0,
                        }
                    candidates[doc_id]["bm25_score"] = float(bm25_scores_norm[idx])

        # ── Step 3: Score fusion ─────────────────────────────────────────────
        fused_results = []
        for doc_id, data in candidates.items():
            combined = (
                self.semantic_weight * data["semantic_score"]
                + self.bm25_weight * data["bm25_score"]
            )
            fused_results.append(
                {
                    "doc_id": doc_id,
                    "text": data["text"],
                    "score": round(combined, 4),
                    "metadata": data["metadata"],
                }
            )

        fused_results.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = fused_results[:n_candidates]

        # ── Step 4: Optional reranking ───────────────────────────────────────
        if use_reranker and self.reranker and len(top_candidates) > 1:
            top_candidates = self._rerank(query, top_candidates, top_k)
        else:
            top_candidates = top_candidates[:top_k]

        logger.info(
            f"hybrid_search → returning {len(top_candidates)} chunks "
            f"(top score={top_candidates[0]['score'] if top_candidates else 0:.3f})"
        )
        return top_candidates

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document from ChromaDB and BM25 index.

        Returns:
            Number of chunks deleted.
        """
        # Find matching chunk IDs in ChromaDB
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        chunk_ids = results["ids"]
        if not chunk_ids:
            return 0

        self.collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks for document_id={document_id}")

        # Rebuild BM25 from ChromaDB (simplest correct approach)
        self._rebuild_bm25_from_chroma()
        return len(chunk_ids)

    def list_documents(self) -> List[Dict]:
        """
        Return a summary of all documents currently in the collection.

        Returns:
            List of {document_id, document_name, chunk_count, uploaded_at}
        """
        results = self.collection.get(include=["metadatas"])
        doc_map: Dict[str, Dict] = {}
        for meta in results["metadatas"]:
            did = meta.get("document_id", "unknown")
            if did not in doc_map:
                doc_map[did] = {
                    "document_id": did,
                    "document_name": meta.get("document_name", "unknown"),
                    "chunk_count": 0,
                    "uploaded_at": meta.get("upload_timestamp", ""),
                }
            doc_map[did]["chunk_count"] += 1
        return list(doc_map.values())

    def chunk_count(self) -> int:
        """Total number of chunks currently stored."""
        return self.collection.count()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from the in-memory text list. Caller must hold lock."""
        if not self._bm25_texts:
            self._bm25_index = None
            return
        tokenized = [text.split() for text in self._bm25_texts]
        self._bm25_index = BM25Okapi(tokenized)

    def _rebuild_bm25_from_chroma(self) -> None:
        """
        Load all documents from ChromaDB and rebuild the BM25 index.

        Called on startup and after deletions to ensure consistency.
        """
        count = self.collection.count()
        if count == 0:
            logger.info("ChromaDB collection is empty — BM25 index not built.")
            return

        logger.info(f"Rebuilding BM25 index from {count} ChromaDB chunks …")
        results = self.collection.get(include=["documents", "metadatas"])
        with self._lock:
            self._bm25_ids = results["ids"]
            self._bm25_texts = results["documents"]
            self._bm25_metadatas = results["metadatas"]
            self._rebuild_bm25()
        logger.info("BM25 index ready.")

    def _rerank(
        self, query: str, candidates: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        Rerank candidates using a cross-encoder and return top_k.

        The cross-encoder score replaces the fusion score so that the final
        ordering reflects fine-grained query-passage relevance.
        """
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)  # type: ignore[union-attr]
        for i, score in enumerate(ce_scores):
            candidates[i]["score"] = round(float(score), 4)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
