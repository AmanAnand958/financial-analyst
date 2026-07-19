# Yeh file financial answers generate karne ke liye LLM framework setup karti hai.
"""
Financial Answer Generator — Groq LLM integration with hallucination guard.

Responsibilities:
- Build a structured prompt from retrieved context chunks
- Apply a confidence threshold to prevent low-confidence hallucinations
- Call the Groq API (mixtral-8x7b-32768)
- Parse and return structured JSON response with source citations
"""

# Type annotations ko runtime support dene ke liye import kiya.
from __future__ import annotations

# OS level operations aur env variables fetch karne ke liye os module.
import os
# System time aur latency compute karne ke liye time module.
import time
# Dict, List aur Optional structures type hinting ke liye imports.
from typing import Dict, List, Optional

# Groq API client load karne ke liye Groq module import kiya.
from groq import Groq
# Project ke logs standard format me console pe print karne ke liye logger.
from loguru import logger


# ── Prompt template ───────────────────────────────────────────────────────────

# System prompt decide karta hai ki AI model ko financial analyst bankar kaise reply dena hai.
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

# User prompt template jo dynamic context aur user question ko formatting framework deta hai.
USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Provide a clear, factual answer citing the source document(s) and page number(s):"""


# Financial answers generate karne ki main class definition start hoti hai.
class FinancialAnswerGenerator:
    """
    Generates grounded financial answers using Groq LLM with hallucination guard.

    The hallucination guard aborts LLM generation when the top retrieved chunk
    has a relevance score below ``confidence_threshold``, returning a safe
    "I couldn't find reliable information" response instead.
    """

    # Class initialization / constructor method variables default values set karega.
    def __init__(
        self,
        # Default LLM model use hoga Llama-3.3.
        model: str = "llama-3.3-70b-versatile",
        # Maximum tokens to generate limit define karega.
        max_tokens: int = 1024,
        # Temperature value model output random behavior trace control karne ke liye.
        temperature: float = 0.1,
        # Confidence score threshold check limits parameter.
        confidence_threshold: float = 0.6,
    ):  # Constructor wrapper end.
        # Environment parameters check settings se keys read karega.
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        # Check if the model is OpenAI or cloud fine-tuned
        self.is_openai_model = model.startswith("ft:") or model.startswith("gpt-")
        
        if self.is_openai_model:
            if not openai_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Cloud fine-tuned models require an OpenAI key."
                )
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_key)
            self.client = None
        else:
            if not groq_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. Add it to your .env file and restart the server."
                )
            self.client = Groq(api_key=groq_key)
            self.openai_client = None
        # Class level model reference string map variable assign kiya.
        self.model = model
        # Target token limit settings configuration parameter store.
        self.max_tokens = max_tokens
        # Class level temperature parameters definitions value set.
        self.temperature = temperature
        # Self level default limit evaluation configuration criteria map.
        self.default_confidence_threshold = confidence_threshold
        # Logger trace notification command screen console message update.
        logger.info(f"FinancialAnswerGenerator ready (model={model})")

    # ── Public API ─────────────────────────────────────────────────────────────

    # Answer generation execute karne ka public function declaration.
    def generate(
        self,
        # Input user question details parameter.
        question: str,
        # Input context chunks structure list variables.
        context_chunks: List[Dict],
        # Runtime confidence threshold validation settings fallback overrides.
        confidence_threshold: Optional[float] = None,
        # Option to bypass NCV guard for benchmarking
        use_ncv: bool = True,
        # Option to override system prompt for benchmarking
        system_prompt: Optional[str] = None,
    ) -> Dict:  # Generate method function definition line.
        """
        Generate a grounded answer from retrieved context.
        """
        # Monotonic timer start trace setup tracking measure.
        start_time = time.monotonic()
        sys_prompt = system_prompt or SYSTEM_PROMPT
        # Custom limits ya default initialization configuration evaluate.
        threshold = confidence_threshold or self.default_confidence_threshold

        # ── Hallucination guard ───────────────────────────────────────────────
        # Agar retrieved data list completely empty output source mapping hai.
        if not context_chunks:
            # Low confidence return method target parameters values.
            return self._low_confidence_response(0.0, start_time)

        # Vector db score metadata match score sequence index 0 pick parameters.
        top_score = context_chunks[0].get("score", 0.0)
        # Validation score checks verification trigger limits.
        if top_score < threshold:
            # Log updates warning print values parameters mappings records.
            logger.warning(
                f"Top chunk score {top_score:.3f} < threshold {threshold:.3f}. "
                f"Returning LOW-confidence response."
            )  # Warning logs end.
            # Low score exits return structure initialization mappings execution.
            return self._low_confidence_response(top_score, start_time)

        # ── Build context string ──────────────────────────────────────────────
        # Top 3 segments chunks filter parameters.
        top_chunks = context_chunks[:3]
        # Text structure compiler methods processing call.
        context_str = self._build_context(top_chunks)

        # Try block implementation catch Groq/OpenAI connection scenarios.
        try:
            if self.is_openai_model:
                completion = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
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
            else:
                # API connection query dynamic response fetch values execution mapping.
                completion = self.client.chat.completions.create(
                    # Model selection options config parameters setup.
                    model=self.model,
                    # Context parameters role system and user mapping structures.
                    messages=[
                        # Core instruction behavioral set system message configurations.
                        {"role": "system", "content": sys_prompt},
                        # User instructions actual string dynamic questions arguments mapping.
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(
                                context=context_str, question=question
                            ),
                        },  # User content structure.
                    ],  # Messages array definition.
                    # Max token generation settings assignment mapping.
                    max_tokens=self.max_tokens,
                    # Output temperature settings config assign.
                    temperature=self.temperature,
                )  # Groq call method end.
            # Output response string parse cleanup trailing spacing characters.
            answer = completion.choices[0].message.content.strip()
        # Connection exceptions errors trigger sequences processing mappings.
        except Exception as exc:
            # Warning logger trigger alert fallback execution.
            logger.warning(f"Groq API error: {exc}. Trying NVIDIA NIM fallback...")
            # NVIDIA fallback keys availability check configurations path.
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            # If fallback keys details validation exists parameter settings.
            if nvidia_api_key:
                # Second path fallback retry implementation attempts.
                try:
                    # Client package class runtime loading inside exception blocks.
                    from openai import OpenAI
                    # Connection client setup endpoint path definitions mapping variables.
                    nvidia_client = OpenAI(
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key=nvidia_api_key
                    )  # Client init.
                    # Fallback completion call instructions settings.
                    completion = nvidia_client.chat.completions.create(
                        # Fallback meta model definition.
                        model="meta/llama-3.1-8b-instruct",
                        # Messages array specifications options configurations parameters.
                        messages=[
                            # System prompt configuration.
                            {"role": "system", "content": sys_prompt},
                            # User query prompt template formatting structure mapping.
                            {
                                "role": "user",
                                "content": USER_PROMPT_TEMPLATE.format(
                                    context=context_str, question=question
                                ),
                            },  # User details.
                        ],  # Message arguments list.
                        # Output limit configurations parameters.
                        max_tokens=self.max_tokens,
                        # Fallback temperature limits assign.
                        temperature=self.temperature,
                    )  # NIM call end.
                    # Parsing result fallback format return configuration text clean parameters.
                    answer = completion.choices[0].message.content.strip()
                    # Logger confirmations success info message print.
                    logger.info("Successfully generated answer using NVIDIA NIM fallback.")
                # Secondary exit exception blocks parameters trace.
                except Exception as nv_exc:
                    # Logging details error trace.
                    logger.error(f"NVIDIA NIM Fallback also failed: {nv_exc}")
                    # Error stack raises program crash standard alert format logs.
                    raise RuntimeError(f"LLM generation failed on Groq and Nvidia: {exc}") from exc
            # If no fallback key is configured.
            else:
                # Console error status logs update.
                logger.error(f"Groq API error: {exc} and NVIDIA_API_KEY is not set.")
                # Exit execution system levels exception outputs configurations.
                raise RuntimeError(f"LLM generation failed: {exc}") from exc
        # ── Deterministic Numeric Citation-Verification (NCV) guard ──────────
        if use_ncv and not self._verify_numeric_citations(answer, top_chunks):
            logger.warning(
                f"NCV Guard detected a numeric mismatch. "
                f"Aborting generation and returning LOW-confidence response."
            )
            return self._low_confidence_response(top_score, start_time)

        # Latency milliseconds measure tracking system monotonic time calculation.
        latency_ms = int((time.monotonic() - start_time) * 1000)
        # Information confirmation message trace target console logging values.
        logger.info(f"Generated answer in {latency_ms}ms (score={top_score:.3f})")

        # Response payload return key value parameters options.
        return {
            # Core string text details.
            "answer": answer,
            # Validation high indicators.
            "confidence": "HIGH",
            # Score round conversions setup.
            "retrieval_score": round(top_score, 4),
            # Citations document maps helper execution.
            "source_chunks": self._build_sources(top_chunks),
            # Latency metric tracker output configurations.
            "latency_ms": latency_ms,
        }  # Return end.

    # ── Private helpers ───────────────────────────────────────────────────────

    def _verify_numeric_citations(self, answer: str, context_chunks: List[Dict]) -> bool:
        """
        Verify that all significant numeric values in the LLM's generated response
        exist in the retrieved context chunks, preventing numerical hallucinations.
        """
        import re
        
        # Merge all context text
        context_text = " ".join([c.get("text", "") for c in context_chunks])
        
        # Normalization helper: remove commas, currency symbols, and spaces
        def normalize(t: str) -> str:
            # Replace currency signs (₹, $, €, Rs) and commas
            t_clean = re.sub(r'[₹$€,]|Rs|crore', '', t, flags=re.IGNORECASE)
            # Remove redundant spaces
            return " ".join(t_clean.split()).lower()
            
        norm_context = normalize(context_text)
        norm_answer = normalize(answer)
        
        # Extract and convert all context numbers to float values
        context_numbers = set()
        for num_str in re.findall(r'\b\d+(?:\.\d+)?\b', norm_context):
            try:
                context_numbers.add(float(num_str))
            except ValueError:
                pass
                
        # Find all numbers in the answer
        for num_str in re.findall(r'\b\d+(?:\.\d+)?\b', norm_answer):
            is_float = '.' in num_str
            # Skip single/double digits to avoid false-positives on list indices or small page nums
            if len(num_str) >= 3 or is_float:
                try:
                    val = float(num_str)
                    if val not in context_numbers:
                        logger.warning(
                            f"Numeric Citation-Verification (NCV) Guard FAILED: "
                            f"Figure '{num_str}' (value {val}) in answer is not present in retrieved context."
                        )
                        return False
                except ValueError:
                    # If it cannot be parsed as float, do a string match
                    if num_str not in norm_context:
                        logger.warning(
                            f"Numeric Citation-Verification (NCV) Guard FAILED: "
                            f"Figure '{num_str}' in answer is not present in retrieved context."
                        )
                        return False
        return True

    # Chunks items text structure map variables helper code.
    def _build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a numbered context string for the prompt."""
        # Empty array holder.
        parts = []
        # Chunks loops sequences enumerate map count index parameter i.
        for i, chunk in enumerate(chunks, start=1):
            # Extract nested metadata map keys.
            meta = chunk.get("metadata", {})
            # Name target value extract settings check string parameters.
            doc_name = meta.get("document_name", "Unknown Document")
            # Page identifier variables metadata extracts.
            page = meta.get("page_number", "N/A")
            # Section path names.
            section = meta.get("section_name", "")
            # Formatted text string build.
            header = f"[{i}] Source: {doc_name}, Page {page}"
            # Custom section checking rules check configuration.
            if section and section != "General":
                # Adding section tags parameters to header path definitions.
                header += f", Section: {section}"  # Header tag update.
            # Append compiled results text mappings parameters options inside list.
            parts.append(f"{header}\n{chunk['text']}")  # Array inserts.
        # String list values join operations parameters outputs.
        return "\n\n---\n\n".join(parts)

    # API output response fields formats structure maps convert helper class methods.
    def _build_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Build the source citation list for the API response."""
        # Source collection storage tracking list.
        sources = []
        # Iterate source lists chunks parameters maps.
        for chunk in chunks:
            # Metadata configuration extracts parameters setups.
            meta = chunk.get("metadata", {})
            # Add mapping entries definitions into list variables.
            sources.append(
                {
                    # Document field values.
                    "document": meta.get("document_name", "Unknown"),
                    # Document page properties.
                    "page": meta.get("page_number", "N/A"),
                    # Sub section references.
                    "section": meta.get("section_name", "General"),
                    # Substring check extract length conversion formats.
                    "excerpt": chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""),
                    # Value round configuration metrics mappings decimal constraints.
                    "relevance_score": round(chunk.get("score", 0.0), 4),
                }
            )  # Sources inserts.
        # Compiled array output return sequences.
        return sources

    # Low score exceptions default safe exit outputs parameters mapping.
    def _low_confidence_response(self, score: float, start_time: float) -> Dict:
        """Return a safe abstention response when confidence is too low."""
        # Payload dictionary structure keys values definitions.
        return {
            # Standard error answers response messages details definitions.
            "answer": (
                "I couldn't find reliable information to answer this question "
                "in the provided documents. Please try rephrasing your query or "
                "upload additional relevant documents."
            ),
            # Validation LOW parameters.
            "confidence": "LOW",
            # Decimal precision round configuration metrics limits check.
            "retrieval_score": round(score, 4),
            # Sources list mapping is empty setup.
            "source_chunks": [],
            # Delay duration latency convert mappings measurements.
            "latency_ms": int((time.monotonic() - start_time) * 1000),
        }  # Return end.
