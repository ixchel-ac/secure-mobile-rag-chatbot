"""FAISS index search with patient-aware retrieval and cross-encoder reranking.

Three-stage retrieval:
1. FAISS bi-encoder search (fast, top-N candidates)
2. Patient augmentation — if query mentions a patient name, pull ALL their chunks
3. Cross-encoder reranking (accurate, re-scores all candidates)

The patient-aware step ensures that queries like "When was Letty's latest check-up?"
find ENCOUNTERS chunks even when the bi-encoder embedding doesn't match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import TOP_K, RERANKER_MODEL, RERANKER_TOP_K
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
    """Patient-aware retriever: FAISS search + patient augmentation + reranking."""

    def __init__(self, index_dir: str | Path, reranker: bool = True):
        """Initialize retriever with embedding model, FAISS index, and reranker.

        Args:
            index_dir: Path to directory containing faiss.index and metadata.jsonl
            reranker: Whether to enable cross-encoder reranking (default: True)
        """
        self.embedder = Embedder()
        self.store = FAISSStore()
        self.store.load(index_dir)

        # Build patient name index for patient-aware retrieval
        self._patient_chunks = self._build_patient_index()

        self._reranker = None
        self._reranker_lock = __import__("threading").Lock()
        if reranker:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[retriever] Loading reranker: {RERANKER_MODEL}")
                self._reranker = CrossEncoder(RERANKER_MODEL)
                print(f"[retriever] Reranker loaded")
            except Exception as e:
                print(f"[retriever] Reranker not available, using FAISS only: {e}")

    def _build_patient_index(self) -> dict[str, list[int]]:
        """Build a lookup from patient name keywords to chunk indices.

        Maps lowercased name parts (first, middle, last) to the indices
        of all chunks belonging to that patient. Used for patient-aware
        retrieval when the query mentions a patient by name.
        """
        patient_chunks: dict[str, list[int]] = {}

        for idx, meta in enumerate(self.store.metadata):
            patient_name = meta.get("patient_name", "")
            if not patient_name:
                continue

            patient_id = meta.get("patient_id", "")
            if patient_id not in patient_chunks:
                patient_chunks[patient_id] = []
            patient_chunks[patient_id].append(idx)

        # Build name-to-patient_id lookup (strip Synthea digits for matching)
        import re
        self._name_to_patient: dict[str, str] = {}
        for idx, meta in enumerate(self.store.metadata):
            patient_name = meta.get("patient_name", "")
            patient_id = meta.get("patient_id", "")
            if not patient_name or not patient_id:
                continue
            # Index by each name part (lowercase, digits stripped)
            for part in patient_name.split():
                clean = re.sub(r"\d+", "", part).lower()
                if len(clean) >= 3:  # skip very short fragments
                    self._name_to_patient[clean] = patient_id

        print(f"[retriever] Patient index: {len(patient_chunks)} patients, "
              f"{len(self._name_to_patient)} name keys")
        return patient_chunks

    def _find_patient_id(self, query: str) -> str | None:
        """Check if the query mentions a known patient name.

        Matches lowercased query words against the patient name index.
        Requires at least 2 name parts to match to avoid false positives
        on common words.
        """
        import re
        # Strip punctuation, digits, and trailing possessive 's (e.g., "Kemmer's" → "kemmer")
        cleaned = []
        for w in query.split():
            w = re.sub(r"[^a-zA-Z]", "", w).lower()
            if w.endswith("s") and len(w) > 3:
                cleaned.append(w)       # keep full word
                cleaned.append(w[:-1])  # also try without trailing s
            elif len(w) >= 3:
                cleaned.append(w)
        query_words = set(cleaned)

        # Find all patient IDs that match any query word
        matched_ids: dict[str, int] = {}
        for word in query_words:
            pid = self._name_to_patient.get(word)
            if pid:
                matched_ids[pid] = matched_ids.get(pid, 0) + 1

        # Require at least 2 name parts to match (e.g., "Letty" + "Kemmer")
        for pid, count in matched_ids.items():
            if count >= 2:
                return pid

        return None

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        sections: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Stage 1: FAISS bi-encoder search for top-N candidates.
        Stage 2: If query mentions a patient, add ALL their chunks as candidates.
        Stage 3: Cross-encoder reranking to re-score all candidates.

        Args:
            query: Natural language query string.
            top_k: Number of final results to return.
            sections: Optional list of section names to filter by
                      (e.g., ["MEDICATIONS", "CONDITIONS"]).

        Returns:
            List of RetrievedChunk objects sorted by descending score.
        """
        # Stage 1: FAISS search
        if sections:
            search_k = max(top_k * 10, RERANKER_TOP_K * 2)
        elif self._reranker:
            search_k = RERANKER_TOP_K
        else:
            search_k = top_k

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

        # Stage 2: Patient-aware augmentation
        # If the query mentions a patient by name, add ALL their chunks
        # so the reranker can find the right section
        patient_id = self._find_patient_id(query)
        if patient_id and patient_id in self._patient_chunks:
            existing_indices = {id(c) for c in chunks}
            patient_indices = self._patient_chunks[patient_id]
            added = 0
            for idx in patient_indices:
                chunk = RetrievedChunk(
                    text=self.store.texts[idx],
                    metadata=self.store.metadata[idx],
                    score=0.0,  # will be re-scored by reranker
                )
                # Avoid duplicates (chunk may already be in FAISS results)
                if not any(c.metadata.get("patient_id") == patient_id
                          and c.metadata.get("section") == chunk.metadata.get("section")
                          and c.metadata.get("chunk_index") == chunk.metadata.get("chunk_index")
                          for c in chunks):
                    chunks.append(chunk)
                    added += 1
            if added > 0:
                print(f"[retriever] Added {added} patient chunks for {patient_id}")

        # Filter by section if requested (before reranking to save compute)
        if sections:
            sections_upper = {s.upper() for s in sections}
            chunks = [
                c for c in chunks
                if c.metadata.get("section", "").upper() in sections_upper
            ]

        # Stage 3: Rerank with cross-encoder
        # Lock ensures only one thread calls predict() at a time on this instance.
        if self._reranker and len(chunks) > 1:
            pairs = [[query, c.text] for c in chunks]
            with self._reranker_lock:
                scores = self._reranker.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)

            chunks.sort(key=lambda c: c.score, reverse=True)

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
