"""Normalize and strip formatting artifacts from Synthea text files.

Phase 1, Step 1.3:
- Remove separator lines (---...---)
- Remove ===== header decorations
- Normalize whitespace
- Preserve section labels (MEDICATIONS, CONDITIONS, etc.)
"""

import re


# Separator line pattern: lines that are only dashes (at least 3)
SEPARATOR_RE = re.compile(r"^-{3,}$", re.MULTILINE)

# Header decoration: lines that are only equals signs (at least 3)
DECORATION_RE = re.compile(r"^={3,}$", re.MULTILINE)

# Multiple blank lines -> single blank line
MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Leading/trailing whitespace on each line
LINE_WHITESPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(raw_text: str) -> str:
    """Clean a raw Synthea patient text file.

    Removes formatting artifacts while preserving section labels
    and clinical content.

    Args:
        raw_text: The raw text from a Synthea .txt file.

    Returns:
        Cleaned text ready for chunking.
    """
    text = raw_text

    # Remove separator lines (----------------...----)
    text = SEPARATOR_RE.sub("", text)

    # Remove header decoration (=====...====)
    text = DECORATION_RE.sub("", text)

    # Remove trailing whitespace on each line
    text = LINE_WHITESPACE_RE.sub("", text)

    # Collapse multiple blank lines into a single one
    text = MULTI_BLANK_RE.sub("\n\n", text)

    # Strip leading/trailing whitespace from the whole document
    text = text.strip()

    return text


if __name__ == "__main__":
    # Run directly: python -m app.ingestion.cleaner
    from pathlib import Path
    from .loader import load_all

    project_root = Path(__file__).parent.parent.parent.parent
    text_dir = project_root / "data" / "synthea" / "text"
    csv_dir = project_root / "data" / "synthea" / "csv"

    records = load_all(text_dir, csv_dir)
    record = records[0]

    print("=== RAW TEXT (first 500 chars) ===")
    print(record.raw_text[:500])
    print("\n")

    cleaned = clean_text(record.raw_text)

    print("=== CLEANED TEXT (first 500 chars) ===")
    print(cleaned[:500])