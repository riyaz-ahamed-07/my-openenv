# SupportTriageEnv Dockerfile
#
# Builds a self-contained OpenEnv-compliant environment server.
# Compatible with:
#   - Local development: docker build + docker run
#   - HuggingFace Spaces: openenv push / HF Docker Space
#
# Build:
#   docker build -t support-triage-env .
#
# Run:
#   docker run -p 8000:8000 support-triage-env
#
# Test:
#   curl http://localhost:8000/health

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies into a virtual environment for clean copying
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements first for Docker layer caching
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app/env

# Install the package itself
RUN pip install --no-cache-dir /app/env

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Set environment paths
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Enable web interface for HuggingFace Space (allows interactive debugging)
ENV ENABLE_WEB_INTERFACE=true

# HuggingFace Spaces uses port 7860 by default; we support both
ENV PORT=7860
EXPOSE 7860
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-7860}/health')" || exit 1

# Start the server
# Uses PORT env var so it works on both HF Spaces (7860) and local (8000)
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
