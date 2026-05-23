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
        description="List of PII entities that were redacted by FW-L2",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source files used to generate the response",
    )
    fw_l2_passed: bool = Field(..., description="Whether FW-L2 validation passed")


class TestRequest(BaseModel):
    """Request body for POST /test — configurable evaluation endpoint."""

    query: str = Field(..., description="The natural language medical query")
    top_k: int = Field(default=5, description="Number of chunks to retrieve from FAISS")
    profile: str = Field(
        default="hardened_fw_l2_bert",
        description="Evaluation profile: naive, naive_fw_l2_base, naive_fw_l2_bert, "
                    "hardened, hardened_fw_l2_base, hardened_fw_l2_bert",
    )
    temperature: float = Field(default=0.1, description="LLM sampling temperature")
    sections: list[str] | None = Field(
        default=None,
        description="Filter retrieval to specific sections (e.g., ['DEMOGRAPHICS', 'MEDICATIONS'])",
    )


class TestResponse(BaseModel):
    """Response body for POST /test — includes raw answer and diagnostics."""

    response: str = Field(..., description="The final (possibly redacted) LLM response")
    raw_response: str = Field(..., description="The raw LLM response before FW-L2")
    profile: str = Field(..., description="The evaluation profile used")
    model: str = Field(..., description="LLM model used")
    redacted_entities: list[str] = Field(default_factory=list, description="PII entity types detected")
    was_redacted: bool = Field(..., description="Whether FW-L2 redacted any PII")
    injection_detected: bool = Field(..., description="Whether injection patterns were found")
    sources: list[str] = Field(default_factory=list, description="Source files used")
    sections_retrieved: list[str] = Field(default_factory=list, description="Chunk sections retrieved")
    fw_l2_passed: bool = Field(..., description="Whether FW-L2 validation passed (no PII, no injection)")


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    data_path: str = Field(default="./data", description="Path to the Synthea data directory")
    clear_existing: bool = Field(
        default=True, description="Whether to clear existing FAISS index before rebuilding"
    )


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    chunks_ingested: int = Field(..., description="Number of chunks ingested into FAISS")