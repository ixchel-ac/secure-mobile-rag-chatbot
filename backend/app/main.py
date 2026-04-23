"""FastAPI entry point for the mobile-rag-firewall backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, query, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index and embedding model on startup."""
    # TODO: Load FAISS index from disk
    # TODO: Load embedding model
    yield
    # Cleanup (if needed)


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