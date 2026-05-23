"""Ingestion pipeline: load → clean → chunk → embed → index.

Phase 1 & 2:
- Orchestrates the full ingestion flow
- Loads Synthea patient records
- Cleans raw text
- Chunks into sections
- Embeds chunks using sentence-transformers
- Builds and saves FAISS index
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .loader import load_all, build_pii_groundtruth, PatientRecord
from .cleaner import clean_text
from .chunker import chunk_patient_record, Chunk
from app.rag.embedder import Embedder
from app.vectorstore.faiss_store import FAISSStore


@dataclass
class PipelineReport:
    """Statistics from a pipeline run."""

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Counts
    patients_loaded: int = 0
    patients_matched: int = 0
    chunks_generated: int = 0
    vectors_created: int = 0
    vector_dimension: int = 0

    # Section breakdown
    sections: Counter = field(default_factory=Counter)

    # File sizes
    index_path: Path | None = None
    index_size_mb: float = 0.0
    metadata_size_mb: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def chunks_per_patient(self) -> float:
        if self.patients_loaded == 0:
            return 0.0
        return self.chunks_generated / self.patients_loaded

    def print_report(self) -> None:
        """Print a formatted report to the console."""
        print("\n" + "=" * 60)
        print("               INGESTION PIPELINE REPORT")
        print("=" * 60)

        print(f"\n{'TIMING':-^60}")
        print(f"  Total duration:        {self.duration_seconds:.2f} seconds")

        print(f"\n{'PATIENTS':-^60}")
        print(f"  Loaded:                {self.patients_loaded}")
        print(f"  Matched to CSV:        {self.patients_matched}")
        match_rate = (self.patients_matched / self.patients_loaded * 100) if self.patients_loaded else 0
        print(f"  Match rate:            {match_rate:.1f}%")

        print(f"\n{'CHUNKS':-^60}")
        print(f"  Total chunks:          {self.chunks_generated}")
        print(f"  Avg per patient:       {self.chunks_per_patient:.1f}")

        print(f"\n{'SECTIONS':-^60}")
        for section, count in self.sections.most_common():
            print(f"  {section:20s}   {count}")

        print(f"\n{'VECTORS':-^60}")
        print(f"  Total vectors:         {self.vectors_created}")
        print(f"  Dimension:             {self.vector_dimension}")

        print(f"\n{'INDEX FILES':-^60}")
        if self.index_path:
            print(f"  Location:              {self.index_path}")
        print(f"  FAISS index size:      {self.index_size_mb:.2f} MB")
        print(f"  Metadata size:         {self.metadata_size_mb:.2f} MB")
        print(f"  Total size:            {self.index_size_mb + self.metadata_size_mb:.2f} MB")

        print("\n" + "=" * 60)


def run_ingestion(
    text_dir: str | Path,
    csv_dir: str | Path,
    verbose: bool = False,
) -> list[Chunk]:
    """Run the full ingestion pipeline.

    Args:
        text_dir: Path to Synthea text files (data/synthea/text/)
        csv_dir: Path to Synthea CSV files (data/synthea/csv/)
        verbose: Print progress information

    Returns:
        List of Chunk objects ready for embedding.
    """
    text_dir = Path(text_dir)
    csv_dir = Path(csv_dir)

    # Step 1: Load patient records
    if verbose:
        print("[pipeline] Loading patient records...")
    records = load_all(text_dir, csv_dir)

    # Step 2: Clean and chunk each record
    if verbose:
        print(f"[pipeline] Processing {len(records)} records...")

    all_chunks: list[Chunk] = []

    for record in records:
        # Clean the raw text
        cleaned = clean_text(record.raw_text)

        # Chunk into sections
        chunks = chunk_patient_record(
            cleaned_text=cleaned,
            patient_id=record.patient_id,
            patient_name=record.patient_name,
            source_file=record.source_file,
            pii_entities=record.pii_entities,
        )

        all_chunks.extend(chunks)

    if verbose:
        print(f"[pipeline] Generated {len(all_chunks)} chunks from {len(records)} patients")

    return all_chunks


def run_ingestion_from_records(
    records: list[PatientRecord],
    verbose: bool = False,
) -> list[Chunk]:
    """Run ingestion on pre-loaded records.

    Useful for testing or when records are already in memory.

    Args:
        records: List of PatientRecord objects
        verbose: Print progress information

    Returns:
        List of Chunk objects ready for embedding.
    """
    if verbose:
        print(f"[pipeline] Processing {len(records)} records...")

    all_chunks: list[Chunk] = []

    for record in records:
        cleaned = clean_text(record.raw_text)
        chunks = chunk_patient_record(
            cleaned_text=cleaned,
            patient_id=record.patient_id,
            patient_name=record.patient_name,
            source_file=record.source_file,
            pii_entities=record.pii_entities,
        )
        all_chunks.extend(chunks)

    if verbose:
        print(f"[pipeline] Generated {len(all_chunks)} chunks")

    return all_chunks


def run_full_ingestion(
    text_dir: str | Path,
    csv_dir: str | Path,
    index_dir: str | Path,
    processed_dir: str | Path | None = None,
    verbose: bool = False,
) -> tuple[FAISSStore, PipelineReport]:
    """Run the full ingestion pipeline including embedding and FAISS index creation.

    Args:
        text_dir: Path to Synthea text files (data/synthea/text/)
        csv_dir: Path to Synthea CSV files (data/synthea/csv/)
        index_dir: Path to save FAISS index (e.g., data/index/)
        processed_dir: Path to save processed data (e.g., data/processed/)
        verbose: Print progress information

    Returns:
        Tuple of (FAISSStore with the built index, PipelineReport with stats).
    """
    report = PipelineReport()
    report.start_time = time.time()

    text_dir = Path(text_dir)
    csv_dir = Path(csv_dir)
    index_dir = Path(index_dir)

    # Step 1: Load patient records
    if verbose:
        print("[pipeline] Loading patient records...")
    records = load_all(text_dir, csv_dir)
    report.patients_loaded = len(records)
    report.patients_matched = sum(1 for r in records if r.pii_entities)

    # Step 1.8: Build PII ground-truth index
    if processed_dir:
        processed_dir = Path(processed_dir)
        phi_output = processed_dir / "phi_groundtruth.json"
        if verbose:
            print(f"[pipeline] Building PII ground truth -> {phi_output}")
        build_pii_groundtruth(csv_dir / "patients.csv", phi_output)

    # Step 2: Clean and chunk each record
    if verbose:
        print(f"[pipeline] Processing {len(records)} records...")

    all_chunks: list[Chunk] = []
    for record in records:
        cleaned = clean_text(record.raw_text)
        chunks = chunk_patient_record(
            cleaned_text=cleaned,
            patient_id=record.patient_id,
            patient_name=record.patient_name,
            source_file=record.source_file,
            pii_entities=record.pii_entities,
        )
        all_chunks.extend(chunks)

        # Track sections
        for chunk in chunks:
            report.sections[chunk.metadata.get("section", "UNKNOWN")] += 1

    report.chunks_generated = len(all_chunks)

    if verbose:
        print(f"[pipeline] Generated {len(all_chunks)} chunks from {len(records)} patients")

    # Step 3: Embed chunks
    if verbose:
        print(f"[pipeline] Embedding {len(all_chunks)} chunks...")

    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    vectors = embedder.embed(texts, show_progress=verbose)

    report.vectors_created = vectors.shape[0]
    report.vector_dimension = vectors.shape[1]

    if verbose:
        print(f"[pipeline] Generated {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")

    # Step 4: Build FAISS index
    if verbose:
        print("[pipeline] Building FAISS index...")

    store = FAISSStore(dimension=embedder.dimension)
    metadata = [chunk.metadata for chunk in all_chunks]
    store.build(vectors, texts, metadata)

    # Step 5: Save to disk
    if verbose:
        print(f"[pipeline] Saving index to {index_dir}...")

    store.save(index_dir)

    # Collect file sizes
    report.index_path = index_dir
    faiss_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.jsonl"
    if faiss_path.exists():
        report.index_size_mb = faiss_path.stat().st_size / (1024 * 1024)
    if metadata_path.exists():
        report.metadata_size_mb = metadata_path.stat().st_size / (1024 * 1024)

    report.end_time = time.time()

    if verbose:
        print("[pipeline] Done!")

    return store, report


if __name__ == "__main__":
    # Full ingestion pipeline: load → clean → chunk → embed → index
    import sys

    project_root = Path(__file__).parent.parent.parent.parent
    text_dir = project_root / "data" / "synthea" / "text"
    csv_dir = project_root / "data" / "synthea" / "csv"
    index_dir = project_root / "data" / "index"

    if not text_dir.exists():
        print(f"Error: {text_dir} not found")
        sys.exit(1)

    store, report = run_full_ingestion(text_dir, csv_dir, index_dir, verbose=True)

    # Print report
    report.print_report()

    # Test search
    print("\n--- Testing search ---")
    embedder = Embedder()
    query = "What medications is the patient taking?"
    query_vector = embedder.embed_query(query)

    results = store.search(query_vector, top_k=3)
    for i, result in enumerate(results):
        print(f"\n[{i+1}] Score: {result.score:.4f}")
        print(f"    Section: {result.metadata.get('section')}")
        print(f"    Patient: {result.metadata.get('patient_name')}")
        print(f"    Text: {result.text[:150]}...")