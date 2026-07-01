FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by pdfplumber / PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir chromadb==0.4.22
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config.yaml .

# Create required directories
RUN mkdir -p chroma_db data/raw data/processed

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
