"""
RAGAs Evaluation Framework — Computes retrieval and generation quality metrics.

Metrics computed:
- Context Precision:  % of retrieved chunks relevant to the query (target ≥ 85%)
- Faithfulness:       % of answer sentences supported by retrieved context (target ≥ 90%)
- Answer Relevancy:   How well the answer addresses the question (target ≥ 85%)
- Latency:            End-to-end query time (target < 5s)

The evaluator stores per-query results and aggregated metrics in memory.
Results can be persisted to disk as JSON.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


class RAGAsEvaluator:
    """
    Lightweight evaluation framework for the FDI RAG pipeline.

    For full RAGAs integration (requires OpenAI API key), call ``run_ragas()``.
    The simpler ``record_query()`` method tracks heuristic metrics without any
    additional API calls and is always available.
    """

    def __init__(self, results_path: str = "./data/processed/eval_results.json"):
        self.results_path = results_path
        self._query_log: List[Dict] = []
        self._load_existing_results()

    # ── Recording (heuristic, always available) ───────────────────────────────

    def record_query(
        self,
        question: str,
        answer: str,
        source_chunks: List[Dict],
        confidence: str,
        latency_ms: int,
    ) -> None:
        """
        Record a query-response pair for running metric computation.

        This uses simple heuristics and does NOT require an external API.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "chunk_count": len(source_chunks),
            "avg_relevance_score": (
                sum(c.get("relevance_score", 0) for c in source_chunks) / len(source_chunks)
                if source_chunks
                else 0.0
            ),
            "is_abstain": confidence == "LOW",
        }
        self._query_log.append(record)
        self._persist()

    def get_metrics(self) -> Dict:
        """
        Compute and return aggregated evaluation metrics from recorded queries.

        Returns heuristic approximations when RAGAs hasn't been run:
        - context_precision:  average relevance score of retrieved chunks
        - faithfulness:       fraction of HIGH-confidence answers
        - answer_relevancy:   fraction of non-abstain answers
        - avg_latency_ms:     mean end-to-end latency
        - total_queries:      total queries recorded
        - hallucination_rate: fraction of abstain responses
        - last_evaluated:     timestamp of most recent query
        """
        if not self._query_log:
            return self._empty_metrics()

        total = len(self._query_log)
        high_conf = sum(1 for r in self._query_log if r["confidence"] == "HIGH")
        abstains = sum(1 for r in self._query_log if r["is_abstain"])
        avg_latency = sum(r["latency_ms"] for r in self._query_log) / total
        avg_relevance = sum(r["avg_relevance_score"] for r in self._query_log) / total

        return {
            "context_precision": round(min(avg_relevance, 1.0), 4),
            "faithfulness": round(high_conf / total, 4),
            "answer_relevancy": round((total - abstains) / total, 4),
            "hallucination_rate": round(abstains / total, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "total_queries": total,
            "last_evaluated": self._query_log[-1]["timestamp"],
            "source": "heuristic",  # distinguish from full RAGAs evaluation
        }

    # ── Full RAGAs evaluation (requires OpenAI key) ───────────────────────────

    def run_ragas(
        self, test_dataset_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Run the full RAGAs evaluation suite on a test dataset.

        Supports both OpenAI and Groq (via LangChain ChatGroq) as evaluation LLMs.

        Args:
            test_dataset_path: Path to JSON file with evaluation dataset.
                               Falls back to self.results_path if None.

        Returns:
            Dict with ragas metric scores, or None if evaluation fails.
        """
        try:
            import datasets.fingerprint
            datasets.fingerprint.generate_fingerprint = lambda *args, **kwargs: "dummy_fingerprint"
            
            import langchain.output_parsers
            import langchain_core.output_parsers
            langchain_core.output_parsers.PydanticOutputParser = langchain.output_parsers.PydanticOutputParser
            
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError:
            logger.error("RAGAs not installed. Run: pip install ragas datasets")
            return None

        dataset_path = test_dataset_path or "./data/processed/eval_dataset.json"
        if not os.path.exists(dataset_path):
            logger.error(f"Evaluation dataset not found: {dataset_path}")
            return None

        try:
            with open(dataset_path) as f:
                data = json.load(f)

            # Map plural keys to singular for RAGAs compatibility
            ragas_data = {
                "question": data.get("questions", data.get("question", [])),
                "contexts": data.get("contexts", []),
                "answer": data.get("answers", data.get("answer", [])),
                "ground_truth": data.get("ground_truths", data.get("ground_truth", [])),
            }

            # Normalize ground_truth to a list of strings
            cleaned_gt = []
            for gt in ragas_data["ground_truth"]:
                if isinstance(gt, list):
                    cleaned_gt.append(gt[0] if len(gt) > 0 else "")
                else:
                    cleaned_gt.append(str(gt))
            ragas_data["ground_truth"] = cleaned_gt

            ds = Dataset.from_dict(ragas_data)
            logger.info(f"Running RAGAs on {len(ds)} samples …")

            metrics = [context_precision, faithfulness, answer_relevancy]

            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not openai_api_key and (nvidia_api_key or groq_api_key):
                logger.info("Configuring RAGAs with LLM and HuggingFace Embeddings...")
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    from ragas.llms import LangchainLLMWrapper
                    from ragas.embeddings import LangchainEmbeddingsWrapper

                    def langchain_llm(llm):
                        return LangchainLLMWrapper(llm)

                    def langchain_embeddings(embeddings):
                        return LangchainEmbeddingsWrapper(embeddings)

                    if nvidia_api_key:
                        logger.info("Using NVIDIA NIM for RAGAs evaluation...")
                        from langchain_community.chat_models import ChatOpenAI
                        llm = ChatOpenAI(
                            model_name="meta/llama-3.1-8b-instruct",
                            openai_api_key=nvidia_api_key,
                            openai_api_base="https://integrate.api.nvidia.com/v1",
                            temperature=0
                        )
                    else:
                        logger.info("Using Groq for RAGAs evaluation...")
                        from langchain_groq import ChatGroq
                        llm = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name="llama-3.1-8b-instant",
                            temperature=0
                        )
                        
                    ragas_llm = langchain_llm(llm)
                    hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    ragas_emb = langchain_embeddings(hf_emb)

                    for metric in metrics:
                        metric.llm = ragas_llm
                        if hasattr(metric, "embeddings"):
                            metric.embeddings = ragas_emb
                except Exception as exc:
                    logger.error(f"Failed to configure Groq LLM for RAGAs: {exc}")

            start = time.monotonic()
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            results = evaluate(
                ds,
                metrics=metrics,
            )
            elapsed = time.monotonic() - start
            logger.info(f"RAGAs evaluation completed in {elapsed:.1f}s")

            # Store full evaluation results
            self._query_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": "Full RAGAs Run",
                "answer": "Evaluated RAGAs metrics",
                "confidence": "HIGH",
                "latency_ms": int(elapsed * 1000),
                "avg_relevance_score": float(results.get("context_precision", 0)),
                "is_abstain": False,
            })
            self._persist()

            metrics_result = {
                "context_precision": round(float(results.get("context_precision", 0)), 4),
                "faithfulness": round(float(results.get("faithfulness", 0)), 4),
                "answer_relevancy": round(float(results.get("answer_relevancy", 0)), 4),
                "avg_latency_ms": self.get_metrics().get("avg_latency_ms", 0),
                "total_queries": self.get_metrics().get("total_queries", 0),
                "last_evaluated": datetime.now(timezone.utc).isoformat(),
                "source": "ragas",
            }
            return metrics_result

        except Exception as exc:
            logger.error(f"RAGAs evaluation failed: {exc}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty_metrics(self) -> Dict:
        return {
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "hallucination_rate": 0.0,
            "avg_latency_ms": 0.0,
            "total_queries": 0,
            "last_evaluated": None,
            "source": "heuristic",
        }

    def _persist(self) -> None:
        """Persist query log to disk (best-effort)."""
        try:
            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
            with open(self.results_path, "w") as f:
                json.dump(self._query_log, f, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Could not persist eval results: {exc}")

    def _load_existing_results(self) -> None:
        """Load prior query log on startup so metrics survive restarts."""
        if os.path.exists(self.results_path):
            try:
                with open(self.results_path) as f:
                    self._query_log = json.load(f)
                logger.info(
                    f"Loaded {len(self._query_log)} existing query records from {self.results_path}"
                )
            except Exception as exc:
                logger.warning(f"Could not load existing eval results: {exc}")
                self._query_log = []
