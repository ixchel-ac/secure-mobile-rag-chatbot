"""FAISS index search -- wraps embedder + faiss_store.

Phase 2, Step 2.1:
- Takes query string, returns list of RetrievedChunk(text, metadata, score)
- Loads embedding model and FAISS index once, reuses for multiple queries
- Optional section filtering (e.g., only MEDICATIONS chunks)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import TOP_K
from app.rag.embedder import Embedder
from app.vectorstore.faiss_store import FAISSStore


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the FAISS index."""

    text: str
    metadata: dict
    score: float

    def __str__(self) -> str:
        """Human-readable summary."""
        section = self.metadata.get("section", "N/A")
        patient = self.metadata.get("patient_name", "N/A")
        preview = self.text[:100].replace("\n", " ")
        return (
            f"RetrievedChunk (score={self.score:.4f})\n"
            f"  Section: {section}\n"
            f"  Patient: {patient}\n"
            f"  Preview: {preview}..."
        )


class Retriever:
    """Retrieves relevant chunks for a query using semantic search."""

    def __init__(self, index_dir: str | Path):
        """Initialize retriever by loading the embedding model and FAISS index.

        Args:
            index_dir: Path to directory containing faiss.index and metadata.jsonl
        """
        self.embedder = Embedder()
        self.store = FAISSStore()
        self.store.load(index_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        sections: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            sections: Optional list of section names to filter by
                      (e.g., ["MEDICATIONS", "CONDITIONS"]).

        Returns:
            List of RetrievedChunk objects sorted by descending score.
        """
        # If filtering by section, request more results to compensate for filtering
        search_k = top_k * 10 if sections else top_k

        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=search_k)

        chunks = [
            RetrievedChunk(
                text=result.text,
                metadata=result.metadata,
                score=result.score,
            )
            for result in results
        ]

        # Filter by section if requested
        if sections:
            sections_upper = {s.upper() for s in sections}
            chunks = [
                c for c in chunks
                if c.metadata.get("section", "").upper() in sections_upper
            ]

        return chunks[:top_k]


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent.parent.parent
    index_dir = project_root / "index"

    if not (index_dir / "faiss.index").exists():
        index_dir = project_root / "data" / "index"

    if not (index_dir / "faiss.index").exists():
        print(f"Error: No FAISS index found. Run 'uv run ingestion' first.")
        sys.exit(1)

    retriever = Retriever(index_dir)

    query = "hypertension treatment"
    print(f"\nQuery: \"{query}\"\n")

    results = retriever.retrieve(query, top_k=5)
    for i, chunk in enumerate(results):
        print(f"[{i + 1}] {chunk}\n")