"""POST /query -- main RAG endpoint."""

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def handle_query(body: QueryRequest, request: Request):
    """Run the full RAG pipeline: retrieve -> generate -> FW-L2 validate."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. FAISS index may not be built yet. Call POST /ingest first.",
        )

    try:
        result = await pipeline.query_async(query=body.query, top_k=body.top_k)
    except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
        raise HTTPException(status_code=502, detail=f"LLM provider unreachable: {e}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider returned {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error ({type(e).__name__}): {e}")

    # Map RAGResponse -> QueryResponse
    redacted_entities = []
    if result.fw_l2_result and result.fw_l2_result.detections:
        redacted_entities = [d.entity_type for d in result.fw_l2_result.detections]

    sources = list({c.metadata.get("source_file", "") for c in result.chunks})

    fw_l2_passed = not result.was_redacted and not result.injection_detected

    return QueryResponse(
        response=result.answer,
        redacted_entities=redacted_entities,
        sources=sources,
        fw_l2_passed=fw_l2_passed,
    )