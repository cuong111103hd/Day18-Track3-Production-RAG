# Lab 18: Production RAG Pipeline
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for NLP
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p reports analysis/reflections data

# Default command
CMD ["python", "main.py"]
