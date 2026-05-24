"""Load Synthea text files and CSV ground truth.

Phase 1, Step 1.2:
- Read .txt files from data/synthea/text/
- Extract patient UUID from filename (e.g., Name_Name_UUID.txt)
- Cross-reference data/synthea/csv/patients.csv to attach SSN, DOB, address
"""
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PatientRecord:
    """A raw patient record loaded from Synthea output."""
    patient_id: str
    patient_name: str
    source_file: str
    raw_text: str
    pii_entities: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable summary."""
        pii = self.pii_entities
        lines = [
            f"PatientRecord: {self.patient_name}",
            f"  ID:     {self.patient_id}",
            f"  File:   {self.source_file}",
            f"  DOB:    {pii.get('dob', 'N/A')}",
            f"  SSN:    {pii.get('ssn', 'N/A')}",
            f"  Text:   {len(self.raw_text)} chars", #{self.raw_text}", 
        ]
        return "\n".join(lines)


def _generate_synthetic_contact(first: str, last: str, patient_id: str) -> dict:
    """Generate deterministic synthetic email and phone from patient fields.

    Uses MD5 hash of the patient UUID to produce reproducible values
    across runs. Email strips Synthea digits from names.

    Args:
        first: Patient first name (e.g., "Adah626")
        last: Patient last name (e.g., "Klein929")
        patient_id: Patient UUID

    Returns:
        Dict with "email" and "phone" keys.
    """
    h = hashlib.md5(patient_id.encode()).hexdigest()

    # Email: strip Synthea digits, lowercase
    clean_first = re.sub(r"\d+", "", first).lower()
    clean_last = re.sub(r"\d+", "", last).lower()
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com"]
    domain = domains[int(h[:2], 16) % len(domains)]
    email = f"{clean_first}.{clean_last}@{domain}"

    # Phone: deterministic from hash digits
    digits = re.sub(r"[^0-9]", "", h)[:10]
    phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}"

    return {"email": email, "phone": phone}


def load_patients_csv(csv_path: Path) -> dict[str, dict]:
    """Load patients.csv and return a lookup keyed by patient UUID -> row dict.

    The CSV columns used:
        Id, BIRTHDATE, SSN, FIRST, MIDDLE, LAST, ADDRESS, CITY, STATE, ZIP

    Synthetic email and phone are generated per patient (Synthea does not
    produce these fields).
    """
    lookup_by_id: dict[str, dict] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["Id"]
            first = row["FIRST"]
            middle = row.get("MIDDLE", "")
            last = row["LAST"]

            contact = _generate_synthetic_contact(first, last, patient_id)

            patient_info = {
                "patient_id": patient_id,
                "ssn": row.get("SSN", ""),
                "dob": row.get("BIRTHDATE", ""),
                "name": f"{first} {last}",
                "full_name": f"{first} {middle} {last}".strip(),
                "address": f"{row.get('ADDRESS', '')}, {row.get('CITY', '')}, {row.get('STATE', '')} {row.get('ZIP', '')}",
                "email": contact["email"],
                "phone": contact["phone"],
                "first": first,
                "middle": middle,
                "last": last,
            }

            lookup_by_id[patient_id] = patient_info

    return lookup_by_id


def extract_uuid_from_filename(filename: str) -> str:
    """Extract the UUID from a Synthea text filename.

    Synthea filenames follow the pattern:
        FirstName_MiddleName_LastName_UUID.txt
    e.g., Adah626_Flo729_Klein929_f7adc5f3-fc3c-f928-7cd1-d8d3a49f3c38.txt

    Returns the UUID portion, or empty string if not found.
    """
    # UUID pattern: 8-4-4-4-12 hex characters
    uuid_pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
    match = re.search(uuid_pattern, filename, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_patient_name_from_header(text: str) -> str:
    """Extract the patient name from the first line of a Synthea text file.

    Synthea text files start with the patient name on line 1, followed by
    a line of '=' characters. Example:
        Adah626 Flo729 Klein929
        =======================
    """
    lines = text.strip().split("\n")
    if lines:
        return lines[0].strip()
    return ""


def load_text_files(text_dir: Path) -> list[tuple[str, str]]:
    """Load all .txt files from the Synthea text output directory.

    Returns a list of (filename, raw_text) tuples.
    """
    results = []
    text_path = Path(text_dir)

    for txt_file in sorted(text_path.glob("*.txt")):
        raw_text = txt_file.read_text(encoding="utf-8")
        results.append((txt_file.name, raw_text))

    return results


def load_all(
    text_dir: str | Path,
    csv_dir: str | Path,
) -> list[PatientRecord]:
    """Load all Synthea text files and cross-reference with patients.csv.

    This is the main entry point for Step 1 of the ingestion pipeline.

    Args:
        text_dir: Path to data/synthea/text/
        csv_dir: Path to data/synthea/csv/

    Returns:
        List of PatientRecord objects with raw text and PII metadata.
    """
    text_dir = Path(text_dir)
    csv_dir = Path(csv_dir)

    # Load CSV lookup by patient UUID
    patients_csv = csv_dir / "patients.csv"
    id_lookup = load_patients_csv(patients_csv)

    # Load text files and cross-reference
    records: list[PatientRecord] = []
    text_files = load_text_files(text_dir)

    matched = 0
    unmatched = 0

    for filename, raw_text in text_files:
        patient_name = parse_patient_name_from_header(raw_text)
        patient_uuid = extract_uuid_from_filename(filename)

        # Look up patient by UUID extracted from filename
        patient_info = id_lookup.get(patient_uuid)

        if patient_info:
            matched += 1
            record = PatientRecord(
                patient_id=patient_uuid,
                patient_name=patient_name,
                source_file=filename,
                raw_text=raw_text,
                pii_entities={
                    "ssn": patient_info["ssn"],
                    "dob": patient_info["dob"],
                    "name": patient_info["name"],
                    "full_name": patient_info["full_name"],
                    "address": patient_info["address"],
                    "email": patient_info["email"],
                    "phone": patient_info["phone"],
                },
            )
        else:
            unmatched += 1
            # Still create a record even without CSV match
            record = PatientRecord(
                patient_id=patient_uuid or filename.replace(".txt", ""),
                patient_name=patient_name,
                source_file=filename,
                raw_text=raw_text,
                pii_entities={},
            )

        records.append(record)

    print(f"[loader] Loaded {len(records)} patient files ({matched} matched CSV, {unmatched} unmatched)")
    return records


def build_pii_groundtruth(csv_path: str | Path, output_path: str | Path) -> dict[str, dict]:
    """Build PII ground-truth index from patients.csv.

    Phase 1, Step 1.8:
    Creates a JSON file mapping patient_id to PII entities:
        {patient_id: {ssn, name, full_name, dob, address, email, phone}}

    Args:
        csv_path: Path to patients.csv
        output_path: Path to write pii_groundtruth.json

    Returns:
        The ground-truth dictionary.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    id_lookup = load_patients_csv(csv_path)

    # Build ground truth with only PII fields
    groundtruth: dict[str, dict] = {}
    for patient_id, info in id_lookup.items():
        groundtruth[patient_id] = {
            "ssn": info["ssn"],
            "dob": info["dob"],
            "name": info["name"],
            "full_name": info["full_name"],
            "address": info["address"],
            "email": info["email"],
            "phone": info["phone"],
        }

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(groundtruth, f, indent=2, ensure_ascii=False)

    print(f"[loader] Saved PII ground truth: {len(groundtruth)} patients -> {output_path}")
    return groundtruth

if __name__ == "__main__":
    # Run directly: python -m app.ingestion.loader
    project_root = Path(__file__).parent.parent.parent.parent
    text_dir = project_root / "data" / "synthea" / "text"
    csv_dir = project_root / "data" / "synthea" / "csv"

    records = load_all(text_dir, csv_dir)

    print("\n--- Example PatientRecord ---")
    print(records[0])
