"""POST /ingest -- data ingestion endpoint."""

from fastapi import APIRouter

from app.models.schemas import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest):
    """Trigger the data ingestion pipeline: load -> clean -> chunk -> embed -> FAISS build."""
    # TODO: Run ingestion pipeline
    return IngestResponse(chunks_ingested=0)