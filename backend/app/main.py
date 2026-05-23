"""FastAPI entry point for the mobile-rag-firewall backend."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import INDEX_DIR
from app.routes import health, query, ingest, test


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index, FW-L2, and RAG pipeline on startup."""
    import threading
    from app.firewall.fw_l2 import FWL2
    from app.rag.pipeline import RAGPipeline

    fw_l2 = FWL2(ner_backend="bert")
    app.state.fw_l2 = fw_l2

    index_path = Path(INDEX_DIR)
    print(f"[startup] index path = {index_path}")
    index_file = index_path / "faiss.index"
    metadata_file = index_path / "metadata.jsonl"

    if index_file.exists() and metadata_file.exists():
        try:
            pipeline = RAGPipeline(index_dir=index_path, fw_l2=fw_l2)
            app.state.pipeline = pipeline
            vectors = pipeline.retriever.store.index.ntotal
            print(f"[startup] RAG pipeline loaded from existing index ({vectors} vectors)")
        except Exception as e:
            app.state.pipeline = None
            print(f"[startup] Failed to load existing index: {e}")
    else:
        app.state.pipeline = None
        print(f"[startup] No index found at {index_path}")
        print("[startup] Call POST /ingest to build the index")

    # Cache for /test profile pipelines — built lazily on first use
    app.state.test_pipelines: dict = {}
    app.state.test_pipelines_lock = threading.Lock()

    yield


app = FastAPI(
    title="Mobile RAG Firewall Backend",
    description="Cloud backend for the two-level firewall RAG system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(test.router)