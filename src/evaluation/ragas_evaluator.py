# RAG framework ke evaluation metrics calculations ke liye framework setup code.
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

# Dynamic type notations enable karne ke liye import.
from __future__ import annotations

# JSON data parse aur format karne ke liye package import.
import json
# File check aur dir structures manipulate karne ke liye os.
import os
# System runtime/elapsed time compute karne ke liye time module.
import time
# Query timestamp formats generate karne ke liye datetime.
from datetime import datetime, timezone
# Generic definitions types check typing supports.
from typing import Any, Dict, List, Optional

# Logging mechanism manage karne ke liye loguru.
from loguru import logger


# RAG pipeline testing metrics calculations ki class.
class RAGAsEvaluator:
    """
    Lightweight evaluation framework for the FDI RAG pipeline.

    For full RAGAs integration (requires OpenAI API key), call ``run_ragas()``.
    The simpler ``record_query()`` method tracks heuristic metrics without any
    additional API calls and is always available.
    """

    # Initialization method results path settings configure.
    def __init__(self, results_path: str = "./data/processed/eval_results.json"):
        # Storage path save parameters logic setup.
        self.results_path = results_path
        # Memory query structures track logs initialize.
        self._query_log: List[Dict] = []
        # Pehle se saved analytics log load karne ka method call.
        self._load_existing_results()

    # ── Recording (heuristic, always available) ───────────────────────────────

    # Single queries metadata logs collect aur save wrapper.
    def record_query(
        self,
        # Client input queries question content.
        question: str,
        # LLM system outputs generated text.
        answer: str,
        # Matches context chunk document sections logs.
        source_chunks: List[Dict],
        # System confidence check HIGH/LOW status markers.
        confidence: str,
        # Process total response duration millisecond value.
        latency_ms: int,
    ) -> None:  # record_query definition line end.
        """
        Record a query-response pair for running metric computation.
        """
        # Dictionary parameters key mappings timestamp values generate setup.
        record = {
            # Current time system timestamp UTC formats track.
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Input query text details.
            "question": question,
            # Output answers string details.
            "answer": answer,
            # Security confidence check indicators.
            "confidence": confidence,
            # Process duration time parameter.
            "latency_ms": latency_ms,
            # Total matches segment length.
            "chunk_count": len(source_chunks),
            # Average score logic compute sequence.
            "avg_relevance_score": (
                # Sum weights divided by lengths validations checks.
                sum(c.get("relevance_score", 0) for c in source_chunks) / len(source_chunks)
                # If matches list exists condition verification.
                if source_chunks
                # Default case return code sequence.
                else 0.0
            ),  # End relevance calculation.
            # Low rating cases indicator flags evaluate definitions.
            "is_abstain": confidence == "LOW",
        }  # Record map end.
        # Collection array items inserts.
        self._query_log.append(record)
        # Disk write sync logic trigger methods execution.
        self._persist()

    # Heuristic calculated average metrics mapping values generator function.
    def get_metrics(self) -> Dict:
        """
        Compute and return aggregated evaluation metrics from recorded queries.
        """
        # If no queries logs exists on storage cache return zeros maps.
        if not self._query_log:
            # Safe default dictionary outputs mapping.
            return self._empty_metrics()

        # Count total elements inside logging collection lists.
        total = len(self._query_log)
        # Filter high confidence responses count loop variables.
        high_conf = sum(1 for r in self._query_log if r["confidence"] == "HIGH")
        # Count total low score abstained queries instances trackers.
        abstains = sum(1 for r in self._query_log if r["is_abstain"])
        # Sum total latency values properties parameters.
        avg_latency = sum(r["latency_ms"] for r in self._query_log) / total
        # Sum average relevance scores outputs metrics parameters.
        avg_relevance = sum(r["avg_relevance_score"] for r in self._query_log) / total

        # Return compiled heuristic results configurations metrics properties.
        return {
            # Average relevance values round mapping formats definition.
            "context_precision": round(min(avg_relevance, 1.0), 4),
            # High ratios evaluate conversions limits configurations details.
            "faithfulness": round(high_conf / total, 4),
            # Non abstain percentages values calculation.
            "answer_relevancy": round((total - abstains) / total, 4),
            # Hallucination values track ratio mappings parameters.
            "hallucination_rate": round(abstains / total, 4),
            # Precision round decimal format latency setup.
            "avg_latency_ms": round(avg_latency, 1),
            # Total items count values mappings properties logs.
            "total_queries": total,
            # Last query timestamp fetch.
            "last_evaluated": self._query_log[-1]["timestamp"],
            # Sources distinction indicator flag setup parameter.
            "source": "heuristic",
        }  # Return end.

    # ── Full RAGAs evaluation (requires OpenAI key) ───────────────────────────

    # RAGAs evaluation automation sequence tool method.
    def run_ragas(
        self, test_dataset_path: Optional[str] = None
    ) -> Optional[Dict]:  # Method declaration.
        """
        Run the full RAGAs evaluation suite on a test dataset.
        """
        # Dynamic check package loading dependencies block validation mappings.
        try:
            # Datasets internals settings.
            import datasets.fingerprint
            # Dummy wrapper assignment configurations.
            datasets.fingerprint.generate_fingerprint = lambda *args, **kwargs: "dummy_fingerprint"
            
            # Langchain internal parsers load compatibility mappings setup.
            import langchain.output_parsers
            # Core framework package overrides.
            import langchain_core.output_parsers
            # Dynamic parser object configuration assignments values override details.
            langchain_core.output_parsers.PydanticOutputParser = langchain.output_parsers.PydanticOutputParser
            
            # Ragas evaluate data structures dependencies.
            from datasets import Dataset
            # Main calculations evaluator call module.
            from ragas import evaluate
            # Quality validation metrics definitions import configurations.
            from ragas.metrics import (
                # Relevancy metric checker.
                answer_relevancy,
                # Context precision metrics.
                context_precision,
                # Faithfulness logic verify.
                faithfulness,
            )  # Metrics import end.
        # Fallback check missing packages handler logs errors warnings.
        except ImportError:
            # Logger console message write status parameters.
            logger.error("RAGAs not installed. Run: pip install ragas datasets")
            # Returns None as output signals indicating failure definitions.
            return None

        # Data path validation checks setup options fallback.
        dataset_path = test_dataset_path or "./data/processed/eval_dataset.json"
        # Verify file file path exists criteria constraints.
        if not os.path.exists(dataset_path):
            # Error log updates tracks message details.
            logger.error(f"Evaluation dataset not found: {dataset_path}")
            # Returns None status mapping details check.
            return None

        # Process datasets records read parsing logic execution checks.
        try:
            # Open source dataset JSON details stream files.
            with open(dataset_path) as f:
                # Load JSON contents definitions details mapping.
                data = json.load(f)

            # Plural keys rename format changes for Ragas standard input schemas.
            ragas_data = {
                # Question key value check setups.
                "question": data.get("questions", data.get("question", [])),
                # Context items mapping path updates.
                "contexts": data.get("contexts", []),
                # LLM answer responses formats setup.
                "answer": data.get("answers", data.get("answer", [])),
                # Ground truths verification details parameters definitions.
                "ground_truth": data.get("ground_truths", data.get("ground_truth", [])),
            }  # Ragas data end.

            # Ground truth type check string sanitization maps sequence loop.
            cleaned_gt = []
            # Ground truth iterations validation checks.
            for gt in ragas_data["ground_truth"]:
                # If element values format represents lists structures.
                if isinstance(gt, list):
                    # Pick index zero standard items if size is non empty.
                    cleaned_gt.append(gt[0] if len(gt) > 0 else "")
                # Single element string parse conversion.
                else:
                    # String conversion variables assignment inserts.
                    cleaned_gt.append(str(gt))  # Cleanup GT inserts.
            # Replace target properties mappings.
            ragas_data["ground_truth"] = cleaned_gt

            # Datasets object conversions parameters configs.
            ds = Dataset.from_dict(ragas_data)
            # Logger tracking prints info notifications.
            logger.info(f"Running RAGAs on {len(ds)} samples …")

            # Metrics items array setup mapping parameters.
            metrics = [context_precision, faithfulness, answer_relevancy]

            # Environment system read variables configurations definitions.
            openai_api_key = os.getenv("OPENAI_API_KEY")
            # Groq system key variables load configs definitions.
            groq_api_key = os.getenv("GROQ_API_KEY")
            # NVIDIA API configurations.
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            # Check setup options for custom LLM configurations overrides logic.
            if not openai_api_key and (nvidia_api_key or groq_api_key):
                # Status configuration logging console print message.
                logger.info("Configuring RAGAs with LLM and HuggingFace Embeddings...")
                # Secondary model packages load retry blocks exception catches.
                try:
                    # Embeddings transformers framework package loading.
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    # Wrapper settings loaders.
                    from ragas.llms import LangchainLLMWrapper
                    # Embeddings wrapper configs.
                    from ragas.embeddings import LangchainEmbeddingsWrapper

                    # Wrapper definition lambda functions mapping helpers.
                    def langchain_llm(llm):
                        # Wrap LLM model.
                        return LangchainLLMWrapper(llm)

                    # Embeddings adapter mappings instructions definition.
                    def langchain_embeddings(embeddings):
                        # Wrap embeddings model.
                        return LangchainEmbeddingsWrapper(embeddings)

                    # Check fallback nvidia NIM configurations availability path.
                    if nvidia_api_key:
                        # Log message confirmation details updates.
                        logger.info("Using NVIDIA NIM for RAGAs evaluation...")
                        # LLM class packages loading interfaces paths.
                        from langchain_community.chat_models import ChatOpenAI
                        # Create client interface mapping parameters options structure.
                        llm = ChatOpenAI(
                            # Llama 3 instruct models target selector parameter.
                            model_name="meta/llama-3.1-8b-instruct",
                            # Credentials variables assigns.
                            openai_api_key=nvidia_api_key,
                            # Endpoints path mapping.
                            openai_api_base="https://integrate.api.nvidia.com/v1",
                            # Temp control.
                            temperature=0
                        )  # ChatOpenAI init.
                    # Fallback path if NVIDIA key is missing use GROQ key configs.
                    else:
                        # Status logging info confirmations.
                        logger.info("Using Groq for RAGAs evaluation...")
                        # LLM Groq packages loading.
                        from langchain_groq import ChatGroq
                        # Create Groq model parameter settings client details.
                        llm = ChatGroq(
                            # Groq setup key parameter variables mappings.
                            groq_api_key=groq_api_key,
                            # Instant version model selection settings.
                            model_name="llama-3.1-8b-instant",
                            # Strict temperature control settings mapping.
                            temperature=0
                        )  # ChatGroq init.
                        
                    # Langchain wrappers setup.
                    ragas_llm = langchain_llm(llm)
                    # Sentence transformers MiniLM model initialization embeds options.
                    hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    # Wrappers configuration.
                    ragas_emb = langchain_embeddings(hf_emb)

                    # Bind model configurations options inside each metrics item.
                    for metric in metrics:
                        # Assign LLM properties details configuration value.
                        metric.llm = ragas_llm
                        # If metrics requires embeddings parameter definitions check.
                        if hasattr(metric, "embeddings"):
                            # Assign embeddings adapters instances options.
                            metric.embeddings = ragas_emb  # Embedding config.
                # Catch configuration exceptions errors during dynamic overrides setup.
                except Exception as exc:
                    # Log updates failure warnings tracking.
                    logger.error(f"Failed to configure Groq LLM for RAGAs: {exc}")

            # Monotonic timeline measure starts.
            start = time.monotonic()
            # Asyncio loop setup options logic mappings.
            import asyncio
            # Load loop references catching exception errors trace path.
            try:
                # Active loop fetch logic.
                loop = asyncio.get_event_loop()
            # Catch handler loop initialize scenarios.
            except RuntimeError:
                # New loop configuration initialization maps.
                loop = asyncio.new_event_loop()
                # Set active thread loop parameters definitions context.
                asyncio.set_event_loop(loop)
                
            # Evaluate framework call trigger metrics computation execution.
            results = evaluate(
                # Datasets input references.
                ds,
                # Metrics array config.
                metrics=metrics,
            )  # Evaluate end.
            # Computations elapsed duration parameter evaluation settings calculation.
            elapsed = time.monotonic() - start
            # Successful performance status logging console prints.
            logger.info(f"RAGAs evaluation completed in {elapsed:.1f}s")

            # Append complete metrics analytics results to persistent memory log.
            self._query_log.append({
                # Current time systems.
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Query label details.
                "question": "Full RAGAs Run",
                # Text answers parameters description.
                "answer": "Evaluated RAGAs metrics",
                # High validation confirmation values.
                "confidence": "HIGH",
                # Timing measure values convert integers.
                "latency_ms": int(elapsed * 1000),
                # Average relevance values extracts mappings properties setups.
                "avg_relevance_score": float(results.get("context_precision", 0)),
                # Is abstain set to False.
                "is_abstain": False,
            })  # Logs appends end.
            # Sync logs on storage path files.
            self._persist()

            # Output results properties key value structures mappings definitions.
            metrics_result = {
                # Context precision calculation parameters values conversions.
                "context_precision": round(float(results.get("context_precision", 0)), 4),
                # Faithfulness conversions mappings.
                "faithfulness": round(float(results.get("faithfulness", 0)), 4),
                # Relevancy results properties.
                "answer_relevancy": round(float(results.get("answer_relevancy", 0)), 4),
                # Average latency retrieve settings check metrics parameters.
                "avg_latency_ms": self.get_metrics().get("avg_latency_ms", 0),
                # Total count elements parameters references track properties.
                "total_queries": self.get_metrics().get("total_queries", 0),
                # Evaluation time settings UTC conversions string path.
                "last_evaluated": datetime.now(timezone.utc).isoformat(),
                # Indicator key tag setup.
                "source": "ragas",
            }  # Result map end.
            # Output variables returns sequence format.
            return metrics_result

        # Catch Ragas execution Exceptions errors block parameters configurations.
        except Exception as exc:
            # Error logger alerts messages displays console paths.
            logger.error(f"RAGAs evaluation failed: {exc}")
            # Returns None fallback.
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    # Empty data configurations setup default dictionary mapping variables.
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        # Empty maps key definitions values parameters configurations setup.
        return {
            "context_precision": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "hallucination_rate": 0.0,
            "avg_latency_ms": 0.0,
            "total_queries": 0,
            "last_evaluated": None,
            "source": "heuristic",
        }  # Return end.

    # File sync write data details persist parameters methods definitions.
    def _persist(self) -> None:
        """Persist query log to disk (best-effort)."""
        # File operations try catch exceptions block parameters sequence.
        try:
            # Create directories paths structures check options configurations.
            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
            # Write open connections files outputs streams.
            with open(self.results_path, "w") as f:
                # Dump collection arrays formatting clean details config metrics.
                json.dump(self._query_log, f, indent=2, default=str)
        # Catch file writing anomalies errors logs console paths trackers.
        except Exception as exc:
            # Warning logger status alert configurations values update.
            logger.warning(f"Could not persist eval results: {exc}")

    # Prior query logs load details read configurations method.
    def _load_existing_results(self) -> None:
        """Load prior query log on startup so metrics survive restarts."""
        # Check files existence criteria constraints.
        if os.path.exists(self.results_path):
            # File reads stream load try catching blocks exception tracks.
            try:
                # Open target paths files.
                with open(self.results_path) as f:
                    # Parsing inputs logs elements load mappings variables.
                    self._query_log = json.load(f)
                # Successful load confirmations log print details updates.
                logger.info(
                    f"Loaded {len(self._query_log)} existing query records from {self.results_path}"
                )  # Info logs.
            # Catch initialization errors file read anomalies traces.
            except Exception as exc:
                # Console prints alert details configuration trackers.
                logger.warning(f"Could not load existing eval results: {exc}")
                # Fallback setup empty logs list structures.
                self._query_log = []
