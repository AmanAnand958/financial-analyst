"""
Financial Answer Generator — Groq LLM integration with hallucination guard.

Responsibilities:
- Build a structured prompt from retrieved context chunks
- Apply a confidence threshold to prevent low-confidence hallucinations
- Call the Groq API (mixtral-8x7b-32768)
- Parse and return structured JSON response with source citations
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from groq import Groq
from loguru import logger


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analyst assistant helping KPMG auditors \
understand financial documents.

Rules you MUST follow:
1. Answer questions based ONLY on the provided context chunks.
2. If the answer is not in the context, respond: \
"I don't have this information in the provided documents."
3. Always cite the exact document name and page number your answer comes from.
4. Be concise, factual, and professional.
5. Never make up numbers, percentages, or financial figures.
6. If multiple documents provide different numbers, clearly distinguish them.
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Provide a clear, factual answer citing the source document(s) and page number(s):"""


class FinancialAnswerGenerator:
    """
    Generates grounded financial answers using Groq LLM with hallucination guard.

    The hallucination guard aborts LLM generation when the top retrieved chunk
    has a relevance score below ``confidence_threshold``, returning a safe
    "I couldn't find reliable information" response instead.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        confidence_threshold: float = 0.6,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and restart the server."
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_confidence_threshold = confidence_threshold
        logger.info(f"FinancialAnswerGenerator ready (model={model})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        context_chunks: List[Dict],
        confidence_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Generate a grounded answer from retrieved context.

        Args:
            question:             The user's financial question.
            context_chunks:       List of retrieved chunks (from HybridRetriever).
            confidence_threshold: Override the instance default threshold.

        Returns:
            {
                "answer": str,
                "confidence": "HIGH" | "LOW",
                "retrieval_score": float,
                "source_chunks": [
                    {
                        "document": str,
                        "page": int | str,
                        "excerpt": str,
                        "relevance_score": float
                    }
                ],
                "latency_ms": int
            }
        """
        start_time = time.monotonic()
        threshold = confidence_threshold or self.default_confidence_threshold

        # ── Hallucination guard ───────────────────────────────────────────────
        if not context_chunks:
            return self._low_confidence_response(0.0, start_time)

        top_score = context_chunks[0].get("score", 0.0)
        if top_score < threshold:
            logger.warning(
                f"Top chunk score {top_score:.3f} < threshold {threshold:.3f}. "
                f"Returning LOW-confidence response."
            )
            return self._low_confidence_response(top_score, start_time)

        # ── Build context string ──────────────────────────────────────────────
        top_chunks = context_chunks[:3]
        context_str = self._build_context(top_chunks)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            context=context_str, question=question
                        ),
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"Groq API error: {exc}. Trying NVIDIA NIM fallback...")
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if nvidia_api_key:
                try:
                    from openai import OpenAI
                    nvidia_client = OpenAI(
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key=nvidia_api_key
                    )
                    completion = nvidia_client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": USER_PROMPT_TEMPLATE.format(
                                    context=context_str, question=question
                                ),
                            },
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    answer = completion.choices[0].message.content.strip()
                    logger.info("Successfully generated answer using NVIDIA NIM fallback.")
                except Exception as nv_exc:
                    logger.error(f"NVIDIA NIM Fallback also failed: {nv_exc}")
                    raise RuntimeError(f"LLM generation failed on Groq and Nvidia: {exc}") from exc
            else:
                logger.error(f"Groq API error: {exc} and NVIDIA_API_KEY is not set.")
                raise RuntimeError(f"LLM generation failed: {exc}") from exc

        latency_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(f"Generated answer in {latency_ms}ms (score={top_score:.3f})")

        return {
            "answer": answer,
            "confidence": "HIGH",
            "retrieval_score": round(top_score, 4),
            "source_chunks": self._build_sources(top_chunks),
            "latency_ms": latency_ms,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a numbered context string for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            doc_name = meta.get("document_name", "Unknown Document")
            page = meta.get("page_number", "N/A")
            section = meta.get("section_name", "")
            header = f"[{i}] Source: {doc_name}, Page {page}"
            if section and section != "General":
                header += f", Section: {section}"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _build_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Build the source citation list for the API response."""
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            sources.append(
                {
                    "document": meta.get("document_name", "Unknown"),
                    "page": meta.get("page_number", "N/A"),
                    "section": meta.get("section_name", "General"),
                    "excerpt": chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""),
                    "relevance_score": round(chunk.get("score", 0.0), 4),
                }
            )
        return sources

    def _low_confidence_response(self, score: float, start_time: float) -> Dict:
        """Return a safe abstention response when confidence is too low."""
        return {
            "answer": (
                "I couldn't find reliable information to answer this question "
                "in the provided documents. Please try rephrasing your query or "
                "upload additional relevant documents."
            ),
            "confidence": "LOW",
            "retrieval_score": round(score, 4),
            "source_chunks": [],
            "latency_ms": int((time.monotonic() - start_time) * 1000),
        }
