"""Section-based text splitting for Synthea patient records.

Phase 1, Step 1.4:
- Split at section boundaries (MEDICATIONS, CONDITIONS, OBSERVATIONS, etc.)
- Each section becomes one chunk
- If a section exceeds 512 tokens (~2048 chars), apply recursive splitting with overlap
- Attach metadata: patient_id, patient_name, section, source_file, phi_entities
"""

from __future__ import annotations

import re  #regular expresions 
from dataclasses import dataclass, field #help to use chunk


# Known Synthea section headers
SECTION_HEADERS = [
    "ALLERGIES",
    "MEDICATIONS",
    "CONDITIONS",
    "CARE PLANS",
    "REPORTS",
    "OBSERVATIONS",
    "PROCEDURES",
    "IMMUNIZATIONS",
    "ENCOUNTERS",
    "IMAGING STUDIES",
]

# Regex to match section headers (e.g., "MEDICATIONS:" at the start of a line)
SECTION_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r"):?\s*$",
    re.MULTILINE,
)

# Approximate chars-per-token ratio for English medical text
CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 512
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN  # ~2048
OVERLAP_CHARS = 200  # ~50 tokens of overlap


@dataclass
class Chunk:
    """A single chunk of text with metadata."""

    text: str
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary."""
        m = self.metadata
        preview = self.text[:100].replace("\n", " ")
        lines = [
            f"Chunk: {m.get('section', 'N/A')}",
            f"  Patient: {m.get('patient_name', 'N/A')}",
            f"  Index:   {m.get('chunk_index', 0)}/{m.get('total_section_chunks', 1)}",
            f"  Length:  {len(self.text)} chars",
            f"  Preview: {preview}...",
        ]
        return "\n".join(lines)


def extract_header_block(cleaned_text: str) -> str:
    """Extract the patient header block (name, demographics) before the first section."""
    match = SECTION_RE.search(cleaned_text)
    if match:
        return cleaned_text[: match.start()].strip()
    return cleaned_text.strip()


def split_into_sections(cleaned_text: str) -> list[tuple[str, str]]:
    """Split cleaned text into (section_name, section_content) pairs.

    Returns a list of tuples. The first element may be ("DEMOGRAPHICS", header_text)
    for the patient header block before any section.
    """
    sections: list[tuple[str, str]] = []

    # Extract header block
    header = extract_header_block(cleaned_text)
    if header:
        sections.append(("DEMOGRAPHICS", header))

    # Find all section boundaries
    matches = list(SECTION_RE.finditer(cleaned_text))

    for i, match in enumerate(matches):
        section_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        content = cleaned_text[start:end].strip()

        if content:
            sections.append((section_name, content))

    return sections


def recursive_split(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into smaller pieces if it exceeds max_chars.

    Splits at newline boundaries to preserve line-level semantics
    (each Synthea entry is one line). Adds overlap between chunks
    while keeping the total within max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Split by single newline (Synthea data has no paragraph breaks)
    lines = text.split("\n")

    # Build chunks by accumulating lines up to (max_chars - overlap - 1)
    # Reserves space for overlap prefix + joining newline on subsequent chunks
    effective_limit = max_chars - overlap - 1

    chunks: list[str] = []
    current = ""

    for line in lines:
        if len(current) + len(line) + 1 <= effective_limit:
            current = current + "\n" + line if current else line
        else:
            if current:
                chunks.append(current.strip())
            current = line

    if current.strip():
        chunks.append(current.strip())

    # Add overlap: prepend the tail of the previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        chunks = overlapped

    return chunks


def chunk_patient_record(
    cleaned_text: str,
    patient_id: str,
    patient_name: str,
    source_file: str,
    phi_entities: dict,
) -> list[Chunk]:
    """Chunk a single cleaned patient record into section-based chunks.

    Each section becomes one chunk. If a section exceeds 512 tokens,
    it is recursively split with overlap.

    Args:
        cleaned_text: Cleaned text from cleaner.py
        patient_id: UUID from patients.csv
        patient_name: Patient name from file header
        source_file: Original filename
        phi_entities: Dict with ssn, dob, name, address from CSV

    Returns:
        List of Chunk objects ready for embedding.
    """
    sections = split_into_sections(cleaned_text)
    chunks: list[Chunk] = []

    for section_name, content in sections:
        # Inject PHI into DEMOGRAPHICS chunk (simulates real EHR records)
        if section_name == "DEMOGRAPHICS" and phi_entities:
            phi_block = ""
            if phi_entities.get("ssn"):
                phi_block += f"SSN:                 {phi_entities['ssn']}\n"
            if phi_entities.get("address"):
                phi_block += f"Address:             {phi_entities['address']}\n"
            if phi_block:
                content = phi_block + content

        # Prepend section context to the text for better embeddings
        section_text = f"{patient_name} -- {section_name}: {content}"

        # Split if too long
        text_pieces = recursive_split(section_text)

        for i, piece in enumerate(text_pieces):
            chunk_id = f"{patient_id}_{section_name}_{i}" if len(text_pieces) > 1 else f"{patient_id}_{section_name}"

            chunk = Chunk(
                text=piece,
                metadata={
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "section": section_name,
                    "source_file": source_file,
                    "phi_entities": phi_entities,
                    "chunk_index": i,
                    "total_section_chunks": len(text_pieces),
                },
            )
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    # Run directly: python -m app.ingestion.chunker
    from pathlib import Path
    from .loader import load_all
    from .cleaner import clean_text

    project_root = Path(__file__).parent.parent.parent.parent
    text_dir = project_root / "data" / "synthea" / "text"
    csv_dir = project_root / "data" / "synthea" / "csv"

    records = load_all(text_dir, csv_dir)
    record = records[0]

    cleaned = clean_text(record.raw_text)
    chunks = chunk_patient_record(
        cleaned_text=cleaned,
        patient_id=record.patient_id,
        patient_name=record.patient_name,
        source_file=record.source_file,
        phi_entities=record.phi_entities,
    )

    print(f"=== Generated {len(chunks)} chunks ===\n")

    print("--- Chunk 1 (DEMOGRAPHICS) ---")
    print(chunks[0])
    print()

    print("--- Chunk 2 (first clinical section) ---")
    print(chunks[1])

    for chunk in chunks:
        print(chunk)