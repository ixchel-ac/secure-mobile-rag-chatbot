"""GET /health -- service check."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check if the backend is healthy, FAISS is loaded, and Ollama is reachable."""
    # TODO: Check FAISS index loaded
    # TODO: Check Ollama connectivity
    return {
        "status": "healthy",
        "faiss_loaded": False,
        "ollama": "unchecked",
    }