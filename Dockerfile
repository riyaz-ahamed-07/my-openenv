# SupportTriageEnv Dockerfile
#
# Single-stage build: installs all Python deps then runs from source.
# This avoids the "pip install /app/env" failure caused by setuptools
# trying to import openenv-core before it's installed.
#
# Build:
#   docker build -t support-triage-env .
#
# Run (local):
#   docker run -p 7860:7860 support-triage-env
#
# Test:
#   curl http://localhost:7860/health

FROM python:3.11-slim

WORKDIR /app

# System deps (needed by some transitive packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies directly (no package self-install needed)
# openenv-core brings fastapi, uvicorn, pydantic, websockets
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.3" \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.0.0" \
    "websockets>=12.0" \
    "openai>=1.0.0"

# Copy application source (runs from source — no pip install of local package)
COPY . /app/env

# Add project root to PYTHONPATH so all imports work
ENV PYTHONPATH="/app/env"
ENV ENABLE_WEB_INTERFACE=true

# HuggingFace Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the server — cd into /app/env so relative imports resolve correctly
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
