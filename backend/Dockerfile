# ---- Stage 1: Base image ----
# Python 3.11 on Debian slim (~150 MB). Includes the C libraries
# that numpy, FAISS, and sentence-transformers need.
# Reference: https://hub.docker.com/_/python
FROM python:3.11-slim AS base

# ---- Stage 2: Install uv ----
# Copy uv binary from Astral's official image. This is the recommended
# approach -- faster than `pip install uv` and doesn't pollute the
# Python environment.
# Reference: https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Stage 3: System dependencies ----
# faiss-cpu needs OpenMP; sentence-transformers needs gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Stage 4: Working directory ----
WORKDIR /app

# ---- Stage 5: Install Python dependencies ----
# Copy dependency files first for Docker layer caching. If pyproject.toml
# and uv.lock haven't changed, Docker reuses this layer -- skipping the
# entire dependency install on rebuilds where only source code changed.
# --frozen ensures uv uses the lockfile exactly (no re-resolution).
# --no-dev skips development-only dependencies (pytest, etc.).
# Reference: https://docs.astral.sh/uv/guides/integration/docker/
COPY pyproject.toml uv.lock ./

# Install CPU-only torch first, then sync remaining deps (skip torch since already installed)
# --no-install-package torch prevents uv from overwriting with the GPU version from the lockfile
RUN uv sync --frozen --no-dev --no-install-project \
    --no-install-package torch \
    --no-install-package nvidia-cublas-cu12 \
    --no-install-package nvidia-cuda-cupti-cu12 \
    --no-install-package nvidia-cuda-nvrtc-cu12 \
    --no-install-package nvidia-cuda-runtime-cu12 \
    --no-install-package nvidia-cudnn-cu12 \
    --no-install-package nvidia-cufft-cu12 \
    --no-install-package nvidia-cufile-cu12 \
    --no-install-package nvidia-curand-cu12 \
    --no-install-package nvidia-cusolver-cu12 \
    --no-install-package nvidia-cusparse-cu12 \
    --no-install-package nvidia-cusparselt-cu12 \
    --no-install-package nvidia-nccl-cu12 \
    --no-install-package nvidia-nvjitlink-cu12 \
    --no-install-package nvidia-nvtx-cu12 \
    --no-install-package nvidia-nvshmem-cu12 \
    --no-install-package triton \
    --no-install-package cuda-bindings \
    --no-install-package cuda-bindings-post11 \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip /root/.cache/uv

# ---- Stage 6: Pre-download embedding model ----
# Download the sentence-transformers model at build time so the
# container starts instantly without fetching ~90 MB on first run.
# Value is passed from docker-compose via build args (sourced from .env).
ARG EMBEDDING_MODEL
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# ---- Stage 7: Copy application code ----
# Now copy the source. This layer changes frequently but doesn't
# invalidate the dependency cache above.
COPY app/ ./app/

# ---- Stage 7: Expose port ----
EXPOSE 8080

# ---- Stage 8: Health check ----
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# ---- Stage 9: Start command ----
# Use uv run to execute within the managed environment.
# PORT is injected by Cloud Run (default 8080); falls back to 8080 locally.
# Reference: https://www.uvicorn.org/
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]