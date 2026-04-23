"""Embedding engine using all-MiniLM-L6-v2 via sentence-transformers.

Phase 1, Step 1.5:
- Load all-MiniLM-L6-v2 model (384 dimensions, ~22M params)
- Encode text chunks into dense vectors for FAISS indexing and query search
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class Embedder:
    """Wraps sentence-transformers for text -> vector conversion."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Load the embedding model.

        Args:
            model_name: HuggingFace model name. Default: all-MiniLM-L6-v2
        """
        print(f"[embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[embedder] Model loaded. Dimension: {self.dimension}")

    def embed(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into normalized vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Show progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), dimension) with L2-normalized vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize so inner product = cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a normalized vector.

        Args:
            query: The query text.

        Returns:
            np.ndarray of shape (1, dimension).
        """
        return self.embed([query], batch_size=1, show_progress=False)