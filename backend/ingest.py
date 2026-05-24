"""Ingestion script: chains loader -> cleaner -> chunker -> embedder -> FAISS build.

Phase 1, Step 1.7:
Run with: cd backend && uv run python ingest.py

This script:
1. Loads all Synthea .txt files and cross-references patients.csv
2. Cleans each file (removes separators, normalizes whitespace)
3. Chunks by clinical section with metadata and PHI entities
4. Embeds all chunks with all-MiniLM-L6-v2 (384d)
5. Builds a FAISS IndexFlatIP and saves to index/
6. Builds a PHI ground-truth index for evaluation
"""

import json
import sys
import time
from pathlib import Path

# Add the backend directory to Python path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.loader import load_all
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_patient_record, Chunk
from app.rag.embedder import Embedder
from app.vectorstore.faiss_store import FAISSStore


# --- Configuration ---
DATA_DIR = Path(__file__).parent.parent / "data"
TEXT_DIR = DATA_DIR / "synthea" / "text"
CSV_DIR = DATA_DIR / "synthea" / "csv"
INDEX_DIR = Path(__file__).parent.parent / "index"
PROCESSED_DIR = DATA_DIR / "processed"


def build_pii_groundtruth(records) -> dict:
    """Build PII ground-truth index: {patient_id -> {ssn, name, dob, address}}.

    Phase 1, Step 1.8.
    """
    groundtruth = {}
    for record in records:
        if record.phi_entities:
            groundtruth[record.patient_id] = record.phi_entities
    return groundtruth


def main():
    start_time = time.time()

    # ---- Step 1: Load ----
    print("=" * 60)
    print("STEP 1: Loading Synthea text files + CSV ground truth")
    print("=" * 60)

    records = load_all(text_dir=TEXT_DIR, csv_dir=CSV_DIR)
    print(f"  -> {len(records)} patient records loaded\n")

    # ---- Step 2 & 3: Clean + Chunk ----
    print("=" * 60)
    print("STEP 2-3: Cleaning and chunking by section")
    print("=" * 60)

    all_chunks: list[Chunk] = []

    for record in records:
        cleaned = clean_text(record.raw_text)
        chunks = chunk_patient_record(
            cleaned_text=cleaned,
            patient_id=record.patient_id,
            patient_name=record.patient_name,
            source_file=record.source_file,
            phi_entities=record.phi_entities,
        )
        all_chunks.extend(chunks)

    print(f"  -> {len(all_chunks)} total chunks from {len(records)} patients")

    # Show section distribution
    section_counts: dict[str, int] = {}
    for chunk in all_chunks:
        section = chunk.metadata.get("section", "UNKNOWN")
        section_counts[section] = section_counts.get(section, 0) + 1
    print("  -> Section distribution:")
    for section, count in sorted(section_counts.items()):
        print(f"       {section}: {count}")
    print()

    # ---- Step 3.5: Save chunks as JSONL ----
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            line = {"text": chunk.text, **chunk.metadata}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"  -> Saved chunks to {chunks_path}\n")

    # ---- Step 4: Embed ----
    print("=" * 60)
    print("STEP 4: Embedding chunks with all-MiniLM-L6-v2")
    print("=" * 60)

    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]
    vectors = embedder.embed(texts)
    print(f"  -> Embedded {vectors.shape[0]} chunks into {vectors.shape[1]}d vectors\n")

    # ---- Step 5: Build FAISS index ----
    print("=" * 60)
    print("STEP 5: Building FAISS index and saving to disk")
    print("=" * 60)

    metadata_list = [chunk.metadata for chunk in all_chunks]
    store = FAISSStore(dimension=embedder.dimension)
    store.build(vectors, texts, metadata_list)
    store.save(INDEX_DIR)
    print()

    # ---- Step 1.8: Build PII ground-truth ----
    print("=" * 60)
    print("STEP 6: Building PII ground-truth index")
    print("=" * 60)

    groundtruth = build_pii_groundtruth(records)
    groundtruth_path = PROCESSED_DIR / "pii_groundtruth.json"
    with open(groundtruth_path, "w", encoding="utf-8") as f:
        json.dump(groundtruth, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved PHI ground-truth for {len(groundtruth)} patients to {groundtruth_path}\n")

    # ---- Summary ----
    elapsed = time.time() - start_time
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Patients loaded:     {len(records)}")
    print(f"  Total chunks:        {len(all_chunks)}")
    print(f"  Vector dimension:    {embedder.dimension}")
    print(f"  FAISS index:         {INDEX_DIR / 'faiss.index'}")
    print(f"  Metadata:            {INDEX_DIR / 'metadata.jsonl'}")
    print(f"  PHI ground-truth:    {groundtruth_path}")
    print(f"  Time elapsed:        {elapsed:.1f}s")
    print()

    # ---- Quick verification search ----
    print("=" * 60)
    print("VERIFICATION: Test search")
    print("=" * 60)

    test_queries = [
        "What medications is this patient taking?",
        "Does the patient have diabetes?",
        "What are the patient's allergies?",
    ]

    for query in test_queries:
        query_vec = embedder.embed_query(query)
        results = store.search(query_vec, top_k=3)
        print(f"\n  Query: \"{query}\"")
        for i, r in enumerate(results):
            section = r.metadata.get("section", "?")
            patient = r.metadata.get("patient_name", "?")
            print(f"    [{i+1}] score={r.score:.4f}  section={section}  patient={patient}")
            print(f"        {r.text[:100]}...")


if __name__ == "__main__":
    main()
