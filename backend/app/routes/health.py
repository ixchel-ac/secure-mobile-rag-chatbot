"""GET /health -- service check."""

import httpx
from fastapi import APIRouter, Request

from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, GROQ_BASE_URL, GROQ_API_KEY

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """Check if the backend is healthy, FAISS is loaded, and LLM is reachable."""
    pipeline = getattr(request.app.state, "pipeline", None)
    faiss_loaded = pipeline is not None
    faiss_vectors = 0
    if faiss_loaded:
        faiss_vectors = pipeline.retriever.store.index.ntotal

    # Check LLM provider reachability
    llm_status = "unchecked"
    try:
        if LLM_PROVIDER == "ollama":
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            llm_status = "reachable" if resp.status_code == 200 else "error"
        elif LLM_PROVIDER == "groq":
            resp = httpx.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=5.0,
            )
            llm_status = "reachable" if resp.status_code == 200 else "error"
        elif LLM_PROVIDER == "wandb":
            llm_status = "configured"
    except httpx.ConnectError:
        llm_status = "unreachable"
    except httpx.TimeoutException:
        llm_status = "timeout"

    status = "healthy" if faiss_loaded and llm_status in ("reachable", "configured") else "degraded"

    return {
        "status": status,
        "faiss_loaded": faiss_loaded,
        "faiss_vectors": faiss_vectors,
        "llm_provider": LLM_PROVIDER,
        "llm_status": llm_status,
    }