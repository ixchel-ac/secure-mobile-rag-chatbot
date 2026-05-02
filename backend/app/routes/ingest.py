"""POST /ingest -- data ingestion endpoint."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.config import INDEX_DIR
from app.models.schemas import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(body: IngestRequest, request: Request):
    """Trigger the full ingestion pipeline: load -> clean -> chunk -> embed -> FAISS build.

    If clear_existing=False and a FAISS index already exists, the existing
    index is loaded as a seed (skips rebuild). Set clear_existing=True to
    force a full rebuild from the Synthea source data.
    """
    from app.ingestion.pipeline import run_full_ingestion
    from app.rag.pipeline import RAGPipeline

    index_dir = Path(INDEX_DIR)
    index_file = index_dir / "faiss.index"
    metadata_file = index_dir / "metadata.jsonl"
    fw_l2 = getattr(request.app.state, "fw_l2", None)

    # If not clearing, reuse existing index as seed
    if not body.clear_existing and index_file.exists() and metadata_file.exists():
        try:
            pipeline = RAGPipeline(index_dir=index_dir, fw_l2=fw_l2)
            request.app.state.pipeline = pipeline
            vectors = pipeline.retriever.store.index.ntotal
            print(f"[ingest] Loaded existing index as seed ({vectors} vectors)")
            return IngestResponse(chunks_ingested=vectors)
        except Exception as e:
            print(f"[ingest] Could not load existing index ({e}), rebuilding...")

    # Full rebuild
    data_path = Path(body.data_path)
    text_dir = data_path / "synthea" / "text"
    csv_dir = data_path / "synthea" / "csv"
    processed_dir = data_path / "processed"

    if not text_dir.exists():
        raise HTTPException(status_code=400, detail=f"Data directory not found: {text_dir}")

    try:
        store, report = await asyncio.to_thread(
            run_full_ingestion, text_dir, csv_dir, index_dir, processed_dir, verbose=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # Reinitialize the RAG pipeline with the new index
    try:
        request.app.state.pipeline = RAGPipeline(index_dir=index_dir, fw_l2=fw_l2)
        print(f"[ingest] RAG pipeline reloaded ({report.chunks_generated} chunks)")
    except Exception as e:
        print(f"[ingest] Warning: could not reload pipeline: {e}")

    return IngestResponse(chunks_ingested=report.chunks_generated)