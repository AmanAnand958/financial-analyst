# FinDoc Intelligence (FDI)
## RAG-Based Financial Document Intelligence System

[![Deploy to Azure](https://img.shields.io/badge/Deploy-Azure%20Container%20Apps-0078d4)](https://azure.microsoft.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)](https://streamlit.io)

A production-grade RAG system that enables KPMG auditors to query financial documents in plain English and receive answers with exact source citations.

---

## 🎯 Features

- **Hybrid Search** — Semantic (sentence-transformers) + BM25 keyword search with score fusion
- **Cross-Encoder Reranking** — Boosts Context Precision from ~61% to ~84%
- **Hallucination Guard** — Abstains when retrieval confidence < 0.6
- **Source Citations** — Every answer includes document name, page number, and excerpt
- **RAGAs Evaluation** — Context Precision, Faithfulness, Answer Relevancy metrics
- **Premium Streamlit UI** — Upload, query, eval dashboard, system health tabs
- **Azure-Ready** — Docker + GitHub Actions CI/CD

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Git

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/kpmg-rag-fdi.git
cd kpmg-rag-fdi

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start the backend API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. In another terminal, start the Streamlit UI
streamlit run src/ui/app.py
```

- **API:** http://localhost:8000
- **UI:** http://localhost:8501
- **API Docs (Swagger):** http://localhost:8000/docs

### Docker (Recommended)

```bash
# Copy and configure .env
cp .env.example .env
# Add GROQ_API_KEY to .env

# Start both services
docker-compose up

# API → http://localhost:8000
# UI  → http://localhost:8501
```

---

## 📁 Project Structure

```
financial-analyst/
├── src/
│   ├── ingestion/
│   │   └── document_processor.py    # PDF extraction + chunking
│   ├── retrieval/
│   │   └── hybrid_search.py         # Semantic + BM25 + reranking
│   ├── llm/
│   │   └── generator.py             # Groq integration + hallucination guard
│   ├── evaluation/
│   │   └── ragas_evaluator.py       # RAGAs metrics
│   ├── api/
│   │   └── main.py                  # FastAPI backend
│   └── ui/
│       └── app.py                   # Streamlit frontend
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── data/
│   └── processed/
│       └── eval_dataset.json        # 20 RAGAs test QA pairs
├── .github/workflows/deploy.yml     # CI/CD pipeline
├── Dockerfile                       # API container
├── Dockerfile.streamlit             # UI container
├── docker-compose.yml               # Local development
├── requirements.txt
├── config.yaml
├── .env.example
└── README.md
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                    │
│  [Upload PDF] → [Chat Interface] → [Source Citations + Eval]    │
└─────────────────────────┬───────────────────────────────────────┘
                           │
┌─────────────────────────▼───────────────────────────────────────┐
│                    FASTAPI BACKEND                               │
│  POST /upload → POST /query → GET /eval-metrics → GET /health   │
└─────────────────────────┬───────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼─────┐    ┌──────▼─────┐    ┌──────▼──────┐
    │Document  │    │ Retrieval  │    │ LLM + Eval  │
    │Pipeline  │    │ Pipeline   │    │ Framework   │
    └────┬─────┘    └──────┬─────┘    └──────┬──────┘
         │                 │                 │
    pdfplumber        Hybrid Search      Groq API
    ↓                 (Semantic+BM25)    ↓
    Chunk             ↓                 RAGAs
    ↓                 ChromaDB          ↓
    Embed             ↓                 Metrics
    ↓                 Top-3 Chunks      Store
    ChromaDB          Reranked
```

---

## 🔧 Configuration

Edit `config.yaml` to tune the pipeline:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 512 | Tokens per chunk |
| `chunk_overlap` | 100 | Overlap between chunks |
| `semantic_weight` | 0.6 | Weight for semantic search |
| `bm25_weight` | 0.4 | Weight for BM25 search |
| `confidence_threshold` | 0.6 | Min score to generate answer |
| `top_k_final` | 3 | Chunks sent to LLM |

---

## 📈 Success Metrics

| Metric | Target | Mechanism |
|--------|--------|-----------|
| Context Precision | ≥ 85% | Cross-encoder reranking |
| Faithfulness | ≥ 90% | System prompt + confidence guard |
| Answer Relevancy | ≥ 85% | Structured prompting |
| Latency | < 5 sec | Groq inference (<500ms) |
| Hallucination Rate | ≤ 5% | Abstain below threshold=0.6 |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📤 Azure Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full Azure deployment instructions.

```bash
# Quick deploy
az login
az group create --name kpmg-rag-rg --location eastus
# Follow DEPLOYMENT.md for full steps
```

---

## 📚 Supported Documents

Tested with publicly available annual reports:
- **Reliance Industries FY2023** — investor.reliance.com
- **Tata Consultancy Services FY2023** — tcs.com/investor-relations
- **Infosys FY2023** — infosys.com/investors

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Groq API key (free) |
| `OPENAI_API_KEY` | Optional | For RAGAs full evaluation |
| `CHROMA_DB_PATH` | Optional | ChromaDB path (default: ./chroma_db) |
| `DEBUG` | Optional | Enable debug logging |

---

## 📄 License

MIT License — See LICENSE file.

---

**Built for KPMG Financial Audit Team · Powered by Groq + ChromaDB + RAGAs**
# financial-analyst
