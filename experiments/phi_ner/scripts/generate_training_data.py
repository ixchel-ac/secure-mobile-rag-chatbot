"""Generate NER training data from Synthea chunks + PHI ground truth.

Creates BIO-tagged training data for fine-tuning BERT NER models.
Each token is labeled as B-NAME, I-NAME, B-ADDRESS, I-ADDRESS, or O (outside).

Usage:
    python experiments/phi_ner/scripts/generate_training_data.py

Output:
    experiments/phi_ner/data/train.json
    experiments/phi_ner/data/val.json
"""

import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env", override=True)

SEED = 42
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GROUNDTRUTH_PATH = PROJECT_ROOT / "data" / "processed" / "phi_groundtruth.json"
INDEX_METADATA_PATH = PROJECT_ROOT / "index" / "metadata.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Entity types the NER model should detect (regex handles SSN, DOB, etc.)
ENTITY_TYPES = ["NAME", "ADDRESS"]


def load_groundtruth() -> dict:
    with open(GROUNDTRUTH_PATH, "r") as f:
        return json.load(f)


def load_chunks() -> list[dict]:
    """Load chunk metadata from the FAISS index."""
    chunks = []
    with open(INDEX_METADATA_PATH, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def find_entity_spans(text: str, entity_value: str, entity_type: str) -> list[dict]:
    """Find all occurrences of an entity in text, return char spans."""
    spans = []
    pattern = re.compile(re.escape(entity_value), re.IGNORECASE)
    for match in pattern.finditer(text):
        spans.append({
            "start": match.start(),
            "end": match.end(),
            "entity_type": entity_type,
            "value": match.group(),
        })
    return spans


def tokenize_simple(text: str) -> list[tuple[str, int, int]]:
    """Simple whitespace + punctuation tokenizer that tracks char offsets."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def assign_bio_tags(
    tokens: list[tuple[str, int, int]],
    spans: list[dict],
) -> list[str]:
    """Assign BIO tags to tokens based on entity spans."""
    tags = ["O"] * len(tokens)

    for span in spans:
        in_entity = False
        for i, (token_text, token_start, token_end) in enumerate(tokens):
            # Token overlaps with entity span
            if token_start < span["end"] and token_end > span["start"]:
                if not in_entity:
                    tags[i] = f"B-{span['entity_type']}"
                    in_entity = True
                else:
                    tags[i] = f"I-{span['entity_type']}"
            else:
                in_entity = False

    return tags


def create_training_example(text: str, phi: dict) -> dict | None:
    """Create a single NER training example from text + PHI."""
    spans = []

    # Find name occurrences
    for name_key in ["full_name", "name"]:
        name = phi.get(name_key, "")
        if name and len(name) > 3:
            spans.extend(find_entity_spans(text, name, "NAME"))

    # Find address occurrences
    address = phi.get("address", "")
    if address and len(address) > 10:
        spans.extend(find_entity_spans(text, address, "ADDRESS"))

    # Also find address components (city, state)
    if address:
        parts = address.split(", ")
        for part in parts:
            part = part.strip()
            if len(part) > 3 and part not in ["Washington"]:  # Skip very common words
                spans.extend(find_entity_spans(text, part, "ADDRESS"))

    # Tokenize and assign BIO tags
    tokens = tokenize_simple(text)
    if not tokens:
        return None

    tags = assign_bio_tags(tokens, spans)

    # Skip examples with no entities (keep some for negative examples)
    has_entities = any(t != "O" for t in tags)

    return {
        "tokens": [t[0] for t in tokens],
        "ner_tags": tags,
        "has_entities": has_entities,
    }


def generate_from_chunks(chunks: list[dict], groundtruth: dict) -> list[dict]:
    """Generate training examples from FAISS chunks."""
    examples = []

    for chunk in chunks:
        patient_id = chunk.get("patient_id", "")
        text = chunk.get("text", "")
        phi = groundtruth.get(patient_id)

        if not phi or not text:
            continue

        example = create_training_example(text, phi)
        if example:
            examples.append(example)

    return examples


def generate_synthetic_responses(groundtruth: dict, count: int = 500) -> list[dict]:
    """Generate synthetic LLM responses that contain PHI (for training).

    These simulate what the naive prompt would produce — responses that
    freely include names, addresses, SSNs, DOBs.
    """
    templates = [
        "The patient {name} was born on {dob}. Their address is {address}.",
        "Patient demographics: {name}, DOB: {dob}, residing at {address}.",
        "Based on the records, {name} lives at {address}.",
        "{name} -- DEMOGRAPHICS: Address: {address}, Birth Date: {dob}.",
        "The patient's name is {name}. They can be reached at {address}.",
        "Medical record for {name}: Date of birth {dob}, home address {address}.",
        "Demographics show {name}, born {dob}, address on file: {address}.",
        "Patient {name} is currently residing at {address}.",
    ]

    patients = list(groundtruth.values())
    examples = []

    for _ in range(count):
        patient = random.choice(patients)
        template = random.choice(templates)

        text = template.format(
            name=random.choice([patient["name"], patient["full_name"]]),
            dob=patient["dob"],
            address=patient["address"],
        )

        example = create_training_example(text, patient)
        if example:
            examples.append(example)

    return examples


def generate_negative_examples(count: int = 200) -> list[dict]:
    """Generate clean medical text with no PHI (negative examples). V1 original."""
    templates = [
        "The patient is taking Aspirin 81mg daily for cardiovascular protection.",
        "Diagnosed with essential hypertension on 2020-01-15.",
        "Current medications include Metformin 500mg twice daily.",
        "The patient was referred for a follow-up appointment.",
        "Lab results show glucose levels within normal range.",
        "Immunization: Influenza vaccine administered on 2025-10-01.",
        "The patient reports no known allergies.",
        "Procedure: Blood pressure check performed during encounter.",
        "Care plan includes lifestyle education regarding hypertension.",
        "Observation: Body mass index 24.5, within normal limits.",
        "The patient completed physical therapy sessions.",
        "Medication was discontinued due to side effects.",
        "I can only answer clinical questions about patient health records.",
        "The patient has been compliant with the treatment plan.",
        "No significant changes noted since the last visit.",
    ]

    examples = []
    for _ in range(count):
        text = random.choice(templates)
        tokens = tokenize_simple(text)
        examples.append({
            "tokens": [t[0] for t in tokens],
            "ner_tags": ["O"] * len(tokens),
            "has_entities": False,
        })

    return examples


def generate_negative_examples_v2(count: int = 500) -> list[dict]:
    """Generate enhanced negative examples with numbers that are NOT addresses (V2).

    Teaches the model that standalone numbers, list numbers, vitals,
    dosages, and clinical measurements are NOT PHI entities.
    """
    templates = [
        # Original templates
        "The patient is taking Aspirin 81mg daily for cardiovascular protection.",
        "Diagnosed with essential hypertension on 2020-01-15.",
        "Current medications include Metformin 500mg twice daily.",
        "The patient was referred for a follow-up appointment.",
        "Lab results show glucose levels within normal range.",
        "Immunization: Influenza vaccine administered on 2025-10-01.",
        "The patient reports no known allergies.",
        "Procedure: Blood pressure check performed during encounter.",
        "Care plan includes lifestyle education regarding hypertension.",
        "Observation: Body mass index 24.5, within normal limits.",
        "The patient completed physical therapy sessions.",
        "Medication was discontinued due to side effects.",
        "I can only answer clinical questions about patient health records.",
        "The patient has been compliant with the treatment plan.",
        "No significant changes noted since the last visit.",
        # Numbered lists (model must learn list numbers are NOT addresses)
        "1. The patient was prescribed Metformin 500mg.",
        "2. Blood pressure was measured at 120/80 mmHg.",
        "3. A follow-up appointment was scheduled in 2 weeks.",
        "4. The patient was advised to reduce sodium intake.",
        "5. Immunizations are up to date.",
        "1. Aspirin 81mg daily\n2. Lisinopril 10mg daily\n3. Metformin 500mg twice daily",
        "Based on the records:\n1. Hypertension diagnosed in 2020\n2. Diabetes diagnosed in 2022\n3. No known allergies",
        "The following 3 conditions were noted:\n1. Essential hypertension\n2. Type 2 diabetes\n3. Obesity",
        "Patient has 5 active medications and 3 chronic conditions.",
        "Review of 4 encounters from the past 12 months.",
        # Vitals and measurements (numbers that are NOT addresses)
        "Heart rate: 72 bpm, blood pressure: 130/85 mmHg.",
        "BMI: 28.5, weight: 185 lbs, height: 5'10\".",
        "Temperature: 98.6°F, respiratory rate: 16 breaths/min.",
        "A1C level: 7.2%, fasting glucose: 126 mg/dL.",
        "Cholesterol: total 210, LDL 130, HDL 55.",
        "Oxygen saturation: 98% on room air.",
        "Pain level reported as 4 out of 10.",
        # Dosages with numbers
        "Prescribed 500mg twice daily for 14 days.",
        "Dosage increased from 10mg to 20mg daily.",
        "Take 2 tablets by mouth every 8 hours.",
        "Inject 40 units subcutaneously before meals.",
        "Apply 1 patch every 72 hours.",
        # Dates in clinical context (NOT DOB)
        "Next appointment scheduled for 2026-05-15.",
        "Medication started on 2025-01-20, discontinued 2025-03-15.",
        "Lab work ordered, results expected in 3-5 business days.",
        "The condition has persisted for approximately 6 months.",
        "Patient was seen 4 times in the last year.",
        # Clinical codes and identifiers (NOT PHI)
        "ICD-10 code: I10 (Essential hypertension).",
        "CPT code 99213 for office visit.",
        "SNOMED CT: 38341003 (Hypertension).",
        "RxNorm: 1191 (Aspirin).",
        "LOINC: 2345-7 (Glucose).",
    ]

    # Also generate numbered list variants with varied counts
    list_templates = [
        "Based on the provided records, here are the findings:\n{items}",
        "The patient's medical history includes:\n{items}",
        "Summary of {n} key observations:\n{items}",
        "Clinical notes ({n} entries):\n{items}",
    ]

    list_items = [
        "Essential hypertension since 2020",
        "Type 2 diabetes managed with Metformin",
        "Seasonal allergies treated with antihistamines",
        "Routine physical exam completed",
        "Flu vaccine administered",
        "Blood work within normal range",
        "No acute findings on examination",
        "Patient reports improved symptoms",
        "Medication refill approved",
        "Follow-up in 3 months recommended",
    ]

    examples = []

    # Standard templates
    for _ in range(count - 100):
        text = random.choice(templates)
        tokens = tokenize_simple(text)
        examples.append({
            "tokens": [t[0] for t in tokens],
            "ner_tags": ["O"] * len(tokens),
            "has_entities": False,
        })

    # Numbered list variants
    for _ in range(100):
        n = random.randint(2, 6)
        items = "\n".join(f"{i+1}. {random.choice(list_items)}" for i in range(n))
        template = random.choice(list_templates)
        text = template.format(items=items, n=n)
        tokens = tokenize_simple(text)
        examples.append({
            "tokens": [t[0] for t in tokens],
            "ner_tags": ["O"] * len(tokens),
            "has_entities": False,
        })

    return examples


def generate_synthetic_responses_v2(groundtruth: dict, count: int = 800) -> list[dict]:
    """Generate enhanced synthetic LLM responses with varied address formats (V2).

    Adds more diverse address contexts so the model learns address structure,
    not just 'numbers near text'.
    """
    templates = [
        # Original templates
        "The patient {name} was born on {dob}. Their address is {address}.",
        "Patient demographics: {name}, DOB: {dob}, residing at {address}.",
        "Based on the records, {name} lives at {address}.",
        "{name} -- DEMOGRAPHICS: Address: {address}, Birth Date: {dob}.",
        "The patient's name is {name}. They can be reached at {address}.",
        "Medical record for {name}: Date of birth {dob}, home address {address}.",
        "Demographics show {name}, born {dob}, address on file: {address}.",
        "Patient {name} is currently residing at {address}.",
        # V2: Numbered list responses (model must learn to separate list numbers from addresses)
        "1. {name}\n   - Birth Date: {dob}\n   - Address: {address}",
        "Patient 1: {name}, DOB: {dob}\nPatient address: {address}",
        "Record #1:\n  Name: {name}\n  DOB: {dob}\n  Address: {address}",
        "Here are the demographics for 1 patient:\n\n{name}\nDOB: {dob}\nAddress: {address}",
        # V2: Inline address mentions
        "The patient {name} from {address} was prescribed medication.",
        "{name} reported a change of address to {address} on {dob}.",
        "According to records, {name} resides at {address} since {dob}.",
    ]

    patients = list(groundtruth.values())
    examples = []

    for _ in range(count):
        patient = random.choice(patients)
        template = random.choice(templates)

        text = template.format(
            name=random.choice([patient["name"], patient["full_name"]]),
            dob=patient["dob"],
            address=patient["address"],
        )

        example = create_training_example(text, patient)
        if example:
            examples.append(example)

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate NER training data")
    parser.add_argument(
        "--version", default="v1", choices=["v1", "v2"],
        help="v1: original data, v2: enhanced with better negatives and address diversity",
    )
    args = parser.parse_args()

    random.seed(SEED)

    print(f"[generate] Version: {args.version}")
    print("[generate] Loading ground truth...")
    groundtruth = load_groundtruth()
    print(f"  {len(groundtruth)} patients")

    print("[generate] Loading chunks...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks")

    # Generate examples from different sources
    print("[generate] Generating from chunks...")
    chunk_examples = generate_from_chunks(chunks, groundtruth)
    print(f"  {len(chunk_examples)} examples ({sum(1 for e in chunk_examples if e['has_entities'])} with entities)")

    if args.version == "v2":
        print("[generate] Generating synthetic responses (v2 — enhanced)...")
        synthetic_examples = generate_synthetic_responses_v2(groundtruth, count=800)
        print(f"  {len(synthetic_examples)} examples")

        print("[generate] Generating negative examples (v2 — enhanced)...")
        negative_examples = generate_negative_examples_v2(count=500)
        print(f"  {len(negative_examples)} examples")
    else:
        print("[generate] Generating synthetic responses (v1 — original)...")
        synthetic_examples = generate_synthetic_responses(groundtruth, count=500)
        print(f"  {len(synthetic_examples)} examples")

        print("[generate] Generating negative examples (v1 — original)...")
        negative_examples = generate_negative_examples(count=300)
        print(f"  {len(negative_examples)} examples")

    # Combine and shuffle
    all_examples = chunk_examples + synthetic_examples + negative_examples
    random.shuffle(all_examples)

    # Split train/val/test (70/15/15)
    n = len(all_examples)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = all_examples[:train_end]
    val = all_examples[train_end:val_end]
    test = all_examples[val_end:]

    # Count entity distribution
    label_counts = {}
    for ex in all_examples:
        for tag in ex["ner_tags"]:
            label_counts[tag] = label_counts.get(tag, 0) + 1

    # Count entities per split
    def count_entities(examples):
        return sum(1 for e in examples if e["has_entities"])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.json"
    val_path = OUTPUT_DIR / "val.json"
    test_path = OUTPUT_DIR / "test.json"

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)

    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    print(f"\n[generate] Saved:")
    print(f"  Train: {len(train):>6} examples ({count_entities(train)} with entities) -> {train_path}")
    print(f"  Val:   {len(val):>6} examples ({count_entities(val)} with entities) -> {val_path}")
    print(f"  Test:  {len(test):>6} examples ({count_entities(test)} with entities) -> {test_path}")
    print(f"\n  Label distribution (all):")
    for label in sorted(label_counts):
        print(f"    {label:12s} {label_counts[label]:>6}")

    # Publish datasets to Weave
    wandb_project = os.getenv("WANDB_PROJECT", "mobile-rag-firewall")
    try:
        import weave
        weave.init(wandb_project)

        for name, data in [("phi-ner-train", train), ("phi-ner-val", val), ("phi-ner-test", test)]:
            dataset = weave.Dataset(name=name, rows=data)
            weave.publish(dataset)

        print(f"\n[generate] Published to Weave: phi-ner-train, phi-ner-val, phi-ner-test (version: {args.version})")

    except ImportError:
        print("\n[generate] Weave not installed, skipping publish.")
    except Exception as e:
        print(f"\n[generate] Weave publish failed: {e}")


if __name__ == "__main__":
    main()