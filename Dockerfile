FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1

# Create working directory
WORKDIR /app

# Install build tools for llama-cpp-python and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
# Install hf-xet for faster downloads
RUN pip install --no-cache-dir -r requirements.txt hf-xet

# Pre-download the GGUF model into the image
RUN mkdir -p /models && \
    hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir /models

# Copy source code
COPY main.py .

# Create input and output directories
RUN mkdir -p /input /output

# Entry point
ENTRYPOINT ["python", "main.py"]
