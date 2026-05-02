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
RUN uv sync --frozen --no-dev --no-install-project

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
EXPOSE 8000

# ---- Stage 8: Health check ----
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ---- Stage 9: Start command ----
# Use uv run to execute within the managed environment.
# Reference: https://www.uvicorn.org/
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]