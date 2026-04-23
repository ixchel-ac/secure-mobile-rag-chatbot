"""POST /query -- main RAG endpoint."""

from fastapi import APIRouter

from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Steps 4-7 from the architecture data flow."""
    # Step 4: Embed query + retrieve from FAISS
    # retrieved = await rag.retrieve(request.query, request.top_k)

    # Step 5: Generate response with Llama
    # raw_response = await rag.generate(request.query, retrieved)

    # Step 6: FW-L2 -- validate + anonymize PHI
    # validated = fw_l2.validate(raw_response, retrieved)

    # Step 7: Return anonymized response
    return QueryResponse(
        response="TODO: implement RAG pipeline",
        redacted_entities=[],
        sources=[],
        fw_l2_passed=True,
    )