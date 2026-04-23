"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., description="The natural language medical query")
    top_k: int = Field(default=5, description="Number of chunks to retrieve from FAISS")


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    response: str = Field(..., description="The anonymized LLM response")
    redacted_entities: list[str] = Field(
        default_factory=list,
        description="List of PHI entities that were redacted by FW-L2",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source files used to generate the response",
    )
    fw_l2_passed: bool = Field(..., description="Whether FW-L2 validation passed")


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    data_path: str = Field(default="./data", description="Path to the Synthea data directory")
    clear_existing: bool = Field(
        default=True, description="Whether to clear existing FAISS index before rebuilding"
    )


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    chunks_ingested: int = Field(..., description="Number of chunks ingested into FAISS")