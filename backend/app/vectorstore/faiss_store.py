"""FAISS index management: build, save, load, and query.

Phase 1, Step 1.6:
- Build IndexFlatIP (exact inner product search on L2-normalized vectors = cosine similarity)
- Add vectors to the index
- Save index to disk (index/faiss.index)
- Save metadata to disk (index/metadata.jsonl) -- one JSON line per vector row
- Load index + metadata from disk
- Query: return top-k results with metadata and scores
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np


@dataclass
class SearchResult:
    """A single search result from FAISS."""

    text: str
    metadata: dict
    score: float


class FAISSStore:
    """Manages a FAISS index with parallel metadata storage."""

    def __init__(self, dimension: int = 384):
        """Initialize an empty FAISS store.

        Args:
            dimension: Vector dimension (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def build(self, vectors: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        """Build a new FAISS index from vectors.

        Args:
            vectors: np.ndarray of shape (n, dimension), L2-normalized.
            texts: List of chunk texts (parallel to vectors).
            metadata: List of metadata dicts (parallel to vectors).
        """
        assert vectors.shape[1] == self.dimension, (
            f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
        )
        assert len(texts) == vectors.shape[0], "texts length must match number of vectors"
        assert len(metadata) == vectors.shape[0], "metadata length must match number of vectors"

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.texts = texts
        self.metadata = metadata

        print(f"[faiss_store] Built index with {self.index.ntotal} vectors ({self.dimension}d)")

    def save(self, index_dir: str | Path) -> None:
        """Save the FAISS index and metadata to disk.

        Creates:
            - {index_dir}/faiss.index -- the FAISS binary index
            - {index_dir}/metadata.jsonl -- one JSON line per vector row

        Args:
            index_dir: Directory to save files in.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "metadata.jsonl"

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        print(f"[faiss_store] Saved FAISS index to {index_path}")

        # Save metadata + text as JSONL (one JSON line per vector row)
        with open(metadata_path, "w", encoding="utf-8") as f:
            for text, meta in zip(self.texts, self.metadata):
                line = {"text": text, **meta}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"[faiss_store] Saved metadata to {metadata_path} ({len(self.metadata)} rows)")

    def load(self, index_dir: str | Path) -> None:
        """Load a FAISS index and metadata from disk.

        Args:
            index_dir: Directory containing faiss.index and metadata.jsonl.
        """
        index_dir = Path(index_dir)
        index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "metadata.jsonl"

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d
        print(f"[faiss_store] Loaded FAISS index from {index_path} ({self.index.ntotal} vectors)")

        # Load metadata
        self.texts = []
        self.metadata = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.texts.append(row.pop("text"))
                self.metadata.append(row)

        print(f"[faiss_store] Loaded metadata from {metadata_path} ({len(self.metadata)} rows)")

        assert self.index.ntotal == len(self.texts), (
            f"Index/metadata mismatch: {self.index.ntotal} vectors vs {len(self.texts)} metadata rows"
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Search the FAISS index for the top-k nearest vectors.

        Args:
            query_vector: np.ndarray of shape (1, dimension), L2-normalized.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call build() or load() first.")

        # Clamp top_k to the number of vectors in the index
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    text=self.texts[idx],
                    metadata=self.metadata[idx],
                    score=float(score),
                )
            )

        return results