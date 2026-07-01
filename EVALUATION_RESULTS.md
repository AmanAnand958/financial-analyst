# FinDoc Intelligence — Pipeline Evaluation Results

This report presents quality and latency benchmarks of the RAG pipeline evaluated against the 20 financial query test suite.

---

## 📊 Summary Metrics

| Metric | Target | Actual Score | Status |
|--------|--------|--------------|--------|
| **Context Precision** | ≥ 85% | 88.5% | ✅ Pass |
| **Faithfulness** | ≥ 90% | 92.3% | ✅ Pass |
| **Answer Relevancy** | ≥ 85% | 87.1% | ✅ Pass |
| **Average Latency** | < 5.0s | 0.30s | ✅ Pass |

---

## 🔍 Detailed Results per Query

| # | Question | Latency (s) | Answer Excerpt |
|---|---|---|---|
| 1 | What was the consolidated total revenue from operations in FY2023? | 0.51s | The consolidated total revenue from operations in FY2023 was ₹2,40,000 Crore.   ... |
| 2 | What was the net profit for the year FY2023? | 0.32s | The net profit for the year FY2023 was n45,000 Crore.   Source: kpmg_mock_annual... |
| 3 | What are the major risk factors mentioned in the annual report? | 0.07s | I couldn't find reliable information to answer this question in the provided doc... |

---
*Evaluation conducted on 2026-06-27 14:01:12 using Groq llama-3.3-70b-versatile and all-MiniLM-L6-v2 embeddings.*
