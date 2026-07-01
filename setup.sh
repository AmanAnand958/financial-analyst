#!/usr/bin/env bash
# =============================================================================
# FinDoc Intelligence — Setup Script
# Run this once to initialize the local development environment.
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   FinDoc Intelligence — Setup Script     ║"
echo "║   RAG Financial Document Intelligence    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python version ─────────────────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# ── Create virtual environment ────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment …"
    python3 -m venv venv
else
    echo "✅ Virtual environment exists"
fi

source venv/bin/activate
echo "✅ Activated virtual environment"

# ── Install dependencies ──────────────────────────────────────────────────────
echo "📥 Installing dependencies (this may take a few minutes) …"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

# ── Create .env if not exists ─────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env from template."
    echo "   Please add your GROQ_API_KEY to .env before starting:"
    echo "   → Get a free key at https://console.groq.com"
    echo ""
else
    echo "✅ .env file exists"
fi

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p chroma_db data/raw data/processed
echo "✅ Directories created"

# ── Run tests ─────────────────────────────────────────────────────────────────
echo ""
echo "🧪 Running ingestion tests (no API key required) …"
GROQ_API_KEY="test_key" python -m pytest tests/test_ingestion.py -v --tb=short 2>&1 || true

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:            ║"
echo "║                                          ║"
echo "║   1. Add GROQ_API_KEY to .env            ║"
echo "║   2. source venv/bin/activate            ║"
echo "║   3. uvicorn src.api.main:app --reload   ║"
echo "║   4. streamlit run src/ui/app.py         ║"
echo "╚══════════════════════════════════════════╝"
echo ""
