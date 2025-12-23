# Chatterbox TTS Server Dockerfile
# Optimized for NVIDIA GPU with CUDA support

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 chatterbox
RUN chown -R chatterbox:chatterbox /app

# Copy requirements first for better caching
COPY --chown=chatterbox:chatterbox server/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Chatterbox source code (from parent context)
COPY --chown=chatterbox:chatterbox src /app/chatterbox_src
COPY --chown=chatterbox:chatterbox pyproject.toml /app/chatterbox_src/

# Install Chatterbox as editable package
WORKDIR /app/chatterbox_src
RUN pip install --no-cache-dir -e .

# Copy server code
WORKDIR /app
COPY --chown=chatterbox:chatterbox server/main.py .

# Create directory for model cache
RUN mkdir -p /app/.cache/huggingface && \
    chown -R chatterbox:chatterbox /app/.cache

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Switch to non-root user
USER chatterbox

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

