# PII NER Model — Optimization Plan

**Goal:** Fix NAME/ADDRESS confusion and missed-name false negatives observed in production LLM responses.

**Observed failure (production):**

```
Input:  "Gregorio Orozco: amLODIPine 2.5 MG..."
Output: "Gregorio Oro[ADDRESS]: amLODIPine 2.5 MG..."
                    ^^^^^^^^^ NAME misclassified as ADDRESS

Missed: "Barbara Leontine Brakus" → not redacted at all
Missed: "Nathan Ernser"           → not redacted at all
```

---

## Root Cause Analysis

| Issue | Root Cause | Evidence |
|-------|-----------|----------|
| NAME → ADDRESS confusion | Model never saw clean names (without Synthea numeric suffixes) during training | Training data uses `Adah626 Klein929` format; LLM strips suffixes to `Gregorio Orozco` |
| Missed names entirely | Multi-word real names without digits fall outside learned patterns | `Barbara Leontine Brakus` has 3 tokens, no digits — unlike any training example |
| High ADDRESS FP rate (reported 169 FP) | Tokenizer subword artifacts on hyphenated/compound words | `HOSPITAL-SEATTLE` → `##SEATTLE` classified as ADDRESS |

**Core problem:** The training data distribution does not match the inference distribution. The model learned to detect Synthea-formatted names (alphanumeric tokens like `Klein929`) but the LLM returns cleaned, real-looking names that the model has never seen.

Ground truth names in `data/processed/phi_groundtruth.json` look like:
```json
{
  "name": "Wilfredo622 Fritsch593",
  "full_name": "Wilfredo622 Isaiah615 Fritsch593"
}
```
But the LLM outputs: `"Wilfredo Fritsch"` or `"Wilfredo Isaiah Fritsch"` — no numeric suffixes.

---

## Phase 1 — Training Pipeline Quick Wins (no new data needed)

These come directly from the ablation study and can be applied immediately.

- [ ] **1.1 Increase training to 10 epochs** — Ablation showed +1.3 F1 points, val F1 still climbing at epoch 10 with no overfitting.
- [ ] **1.2 Apply 3× oversampling of entity-rich examples** — Ablation showed +1.2 F1 points.
- [ ] **1.3 Increase batch size to 32** — Ablation showed +0.78 F1 points and faster training.
- [ ] **1.4 Confirm learning rate at 5e-5** — Already optimal per ablation.

**Expected gain:** Baseline 0.9734 → ~0.986–0.988 F1 (on current test set).

**Important:** These changes improve performance on the existing Synthea distribution but will NOT fix the clean-name problem. Phase 2 is required for that.

### Code Changes

#### 1.1 + 1.3 — `fw_l2_ner/scripts/train.py`

`MODEL_CONFIGS` already has `epochs: 10`. Batch size needs updating from 16 → 32:

```python
# train.py — MODEL_CONFIGS (line ~46)
# BEFORE:
MODEL_CONFIGS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "lr": 5e-5,
        "epochs": 10,
        "batch_size": 16,            # ← change this
        ...
    },
    "bert": {
        ...
        "batch_size": 16,            # ← change this
        ...
    },
    "roberta": {
        ...
        "batch_size": 16,            # ← change this
        ...
    },
}

# AFTER:
MODEL_CONFIGS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "lr": 5e-5,
        "epochs": 10,
        "batch_size": 32,            # ← ablation best
        ...
    },
    # same for bert and roberta
}
```

#### 1.2 — `fw_l2_ner/scripts/train.py`

Add oversampling in `load_data()` or before creating the dataset in `train_model()`:

```python
# train.py — add after load_data() call in train_model() (~line 275)

def oversample_entity_examples(data: list[dict], factor: int = 3) -> list[dict]:
    """Duplicate examples containing entities to address class imbalance.

    The training data is 99.2% O tokens. Oversampling entity-rich
    examples 3x improved F1 by +1.2 points in ablation study.
    """
    entity_examples = [ex for ex in data if ex.get("has_entities", False)]
    non_entity_examples = [ex for ex in data if not ex.get("has_entities", False)]
    # Duplicate entity examples (factor - 1) times (they already appear once)
    oversampled = non_entity_examples + entity_examples * factor
    random.shuffle(oversampled)
    return oversampled
```

Then call it in `train_model()`:

```python
# train.py — inside train_model(), before creating PIINERDataset (~line 314)

# Apply 3x oversampling of entity-rich examples
train_data = oversample_entity_examples(train_data, factor=3)
print(f"  After 3x oversampling: {len(train_data)} examples")
```

Add `import random` at top of file.

#### 1.1 — Early stopping patience

Currently no early stopping in `train.py` (it was in the notebooks). To keep it but avoid premature stopping:

```python
# train.py — add to imports
from transformers import EarlyStoppingCallback

# train.py — add to Trainer() call (~line 341)
trainer = Trainer(
    ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],  # was 2
)
```

---

## Phase 1B — Notebook Changes (`01_training.ipynb`)

The Colab notebook has its own copies of `MODEL_CONFIGS`, training logic, and data loading that are **independent from `train.py`**. Both must be updated in sync.

### Current notebook vs script differences

| Setting | Notebook (cell 10) | Script (`train.py`) | Target |
|---------|-------------------|---------------------|--------|
| Epochs | 5 | 10 | **10** |
| Batch size (distilbert) | 32 | 16 | **32** |
| Batch size (bert/roberta) | 24 | 16 | **32** |
| Early stopping patience | 2 (cell 14) | none | **4** |
| Data source | Weave refs | local JSON | unchanged |
| Oversampling | none | none | **3×** |

### Cell-by-cell changes

#### Cell 10 — `MODEL_CONFIGS`

```python
# BEFORE (cell 10):
MODEL_CONFIGS = {
    "distilbert": {"name": "distilbert-base-uncased", "lr": 5e-5, "epochs": 5, "batch_size": 32},
    "bert": {"name": "bert-base-uncased", "lr": 3e-5, "epochs": 5, "batch_size": 24},
    "roberta": {"name": "roberta-base", "lr": 2e-5, "epochs": 5, "batch_size": 24},
}

# AFTER:
MODEL_CONFIGS = {
    "distilbert": {"name": "distilbert-base-uncased", "lr": 5e-5, "epochs": 10, "batch_size": 32},
    "bert": {"name": "bert-base-uncased", "lr": 3e-5, "epochs": 10, "batch_size": 32},
    "roberta": {"name": "roberta-base", "lr": 2e-5, "epochs": 10, "batch_size": 32},
}
```

#### Cell 8 — Data loading

The notebook pulls data from Weave (`phi-ner-train:latest`). After running `ner-generate --version v3` locally and publishing to Weave, the notebook will automatically pick up v3 data. **No code change needed** — just re-run `ner-generate --version v3` before retraining.

However, **add oversampling** after loading the data. Add a new cell between cell 8 and cell 9 (or append to cell 8):

```python
# NEW — Add after data loading (cell 8)
# 3× oversampling of entity-rich examples (ablation: +1.2 F1 points)
import random
random.seed(42)

entity_examples = [ex for ex in train_data if ex.get("has_entities", False)]
non_entity = [ex for ex in train_data if not ex.get("has_entities", False)]
train_data = non_entity + entity_examples * 3
random.shuffle(train_data)

print(f"After 3× oversampling: {len(train_data)} training examples "
      f"({len(entity_examples)} entity examples × 3)")
```

#### Cell 14 — `train_model()` early stopping

```python
# BEFORE (cell 14, inside train_model):
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],

# AFTER:
callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
```

#### Cell 9 — Markdown description update

Update the markdown cell to reflect the new config:

```markdown
| Model | Params | LR | Batch | Why |
|-------|--------|-----|-------|-----|
| DistilBERT | 66M | 5e-5 | 32 | Smallest, fastest — best if task is easy |
| BERT | 110M | 3e-5 | 32 | Standard baseline |
| RoBERTa | 125M | 2e-5 | 32 | Often best for NER — but needs lower LR |
```

#### Cell 13 — Markdown description update

Update early stopping description:

```markdown
- `EarlyStoppingCallback(patience=4)` — stops if F1 doesn't improve for 4 consecutive epochs (increased from 2 to avoid premature stopping per ablation study)
```

#### Cell 15 — Markdown description update

```markdown
After all 3 complete, a comparison table shows F1 macro, F1 per entity type.
Early stopping (patience=4) will halt training if F1 plateaus before the max 10 epochs.
```

### Summary of notebook changes

| Cell | Type | Change |
|------|------|--------|
| 8 (or new cell after 8) | Code | Add 3× oversampling of entity-rich examples |
| 9 | Markdown | Update batch size table (24 → 32) |
| 10 | Code | `epochs`: 5 → 10, `batch_size`: 24 → 32 for bert/roberta |
| 13 | Markdown | Update patience description (2 → 4) |
| 14 | Code | `early_stopping_patience`: 2 → 4 |
| 15 | Markdown | Update epoch count and patience references |

**Note:** The notebook does NOT need changes for Phase 2 (clean-name augmentation) — that happens in `generate_training_data.py` and is published to Weave. The notebook just pulls `phi-ner-train:latest` which will contain the v3 data.

---

## Phase 2 — Training Data: Clean Name Augmentation (critical fix)

This phase directly addresses the production failure. The LLM strips Synthea numeric suffixes before responding, so the model must learn to detect names without them.

- [ ] **2.1 Add clean-name variants to training data**
- [ ] **2.2 Add multi-token real-name patterns in clinical contexts**
- [ ] **2.3 Add names in list/medication contexts**
- [ ] **2.4 Diversify name positions**
- [ ] **2.5 Update `generate_training_data.py` with `--version v3`**

**Validation:** After regenerating data, confirm that the training set contains:
- At least 30% of name examples using clean (no-suffix) format
- Examples in list/bullet contexts matching LLM output patterns

### Code Changes

#### 2.1 — `fw_l2_ner/scripts/generate_training_data.py`

Add a helper function to strip Synthea numeric suffixes from names:

```python
# generate_training_data.py — add after ENTITY_TYPES (~line 34)

def strip_synthea_suffix(name: str) -> str:
    """Strip numeric suffixes from Synthea names.

    'Wilfredo622 Isaiah615 Fritsch593' → 'Wilfredo Isaiah Fritsch'
    'Lindsay928 Zieme486'              → 'Lindsay Zieme'
    """
    return re.sub(r"(\d{2,4})", "", name).strip()
```

#### 2.1 + 2.2 — New synthetic response generators

Add clean-name variants to the existing synthetic response generator:

```python
# generate_training_data.py — add new function after generate_synthetic_responses_v2()

def generate_clean_name_responses(groundtruth: dict, count: int = 800) -> list[dict]:
    """Generate synthetic responses using clean names (no Synthea suffixes).

    This directly addresses the production failure where the LLM outputs
    names like 'Gregorio Orozco' but the model was only trained on
    'Gregorio366 Orozco750'.
    """
    templates = [
        # Standard clinical contexts
        "The patient {name} was born on {dob}. Their address is {address}.",
        "Patient demographics: {name}, DOB: {dob}, residing at {address}.",
        "Based on the records, {name} lives at {address}.",
        "The patient's name is {name}. They can be reached at {address}.",
        "Medical record for {name}: Date of birth {dob}, home address {address}.",
        "Patient {name} is currently residing at {address}.",
        "{name} -- DEMOGRAPHICS: Address: {address}, Birth Date: {dob}.",
        "Demographics show {name}, born {dob}, address on file: {address}.",
        # Clinical encounter contexts
        "{name} was seen today for a follow-up visit.",
        "{name} presented with complaints of chest pain.",
        "The patient {name} was prescribed amLODIPine 2.5 MG Oral Tablet.",
        "{name} reports improved symptoms since the last visit.",
        "Encounter for {name} at SWEDISH EDMONDS on {dob}.",
        "{name} was referred to cardiology for further evaluation.",
    ]

    patients = list(groundtruth.values())
    examples = []

    for _ in range(count):
        patient = random.choice(patients)
        template = random.choice(templates)

        # Use clean name (strip Synthea suffixes)
        clean_name = strip_synthea_suffix(
            random.choice([patient["name"], patient["full_name"]])
        )

        text = template.format(
            name=clean_name,
            dob=patient["dob"],
            address=patient["address"],
        )

        # Create PII dict with clean name for span matching
        clean_pii = {
            "name": strip_synthea_suffix(patient["name"]),
            "full_name": strip_synthea_suffix(patient["full_name"]),
            "address": patient["address"],
            "dob": patient["dob"],
        }

        example = create_training_example(text, clean_pii)
        if example:
            examples.append(example)

    return examples
```

#### 2.3 — Medication list context templates

```python
# generate_training_data.py — add new function

def generate_medication_list_responses(groundtruth: dict, count: int = 400) -> list[dict]:
    """Generate synthetic medication list responses matching production format.

    The production failure occurred in this exact pattern:
      '- Gregorio Orozco: amLODIPine 2.5 MG Oral Tablet'
    """
    medications = [
        "amLODIPine 2.5 MG Oral Tablet",
        "lisinopril 10 MG Oral Tablet",
        "Hydrochlorothiazide 25 MG Oral Tablet",
        "Metformin 500 MG Oral Tablet",
        "Aspirin 81 MG Oral Tablet",
        "Atorvastatin 20 MG Oral Tablet",
        "Omeprazole 20 MG Delayed Release Oral Capsule",
        "Acetaminophen 325 MG Oral Tablet",
        "insulin isophane, human 70 UNT/ML Injectable Suspension",
    ]

    conditions = [
        "hypertension", "diabetes mellitus type 2",
        "hyperlipidemia", "GERD", "chronic pain",
    ]

    list_templates = [
        "- {name}: {med}",
        "* {name}: {med}",
        "{name}: {med}, {med2}",
        "- {name} ({condition}): {med}",
        "Patient {name} is taking {med} and {med2}.",
    ]

    intro_templates = [
        "The patients with {condition} are taking the following medications:\n\n{items}",
        "Medication list:\n\n{items}",
        "Based on the records, the following patients are on treatment:\n\n{items}",
    ]

    patients = list(groundtruth.values())
    examples = []

    for _ in range(count):
        # Pick 2-4 patients for a list response
        n_patients = random.randint(2, 4)
        selected = random.sample(patients, min(n_patients, len(patients)))
        condition = random.choice(conditions)

        items = []
        for patient in selected:
            clean_name = strip_synthea_suffix(
                random.choice([patient["name"], patient["full_name"]])
            )
            med = random.choice(medications)
            med2 = random.choice(medications)
            template = random.choice(list_templates)
            item = template.format(
                name=clean_name, med=med, med2=med2, condition=condition
            )
            items.append(item)

        intro = random.choice(intro_templates)
        text = intro.format(condition=condition, items="\n".join(items))

        # Create training examples for each patient in the list
        # We need to tag all clean names in the full text
        all_spans = []
        tokens = tokenize_simple(text)
        if not tokens:
            continue

        for patient in selected:
            clean_name = strip_synthea_suffix(patient["name"])
            clean_full = strip_synthea_suffix(patient["full_name"])
            for name in [clean_name, clean_full]:
                if name and len(name) > 3:
                    all_spans.extend(find_entity_spans(text, name, "NAME"))

        tags = assign_bio_tags(tokens, all_spans)
        has_entities = any(t != "O" for t in tags)

        examples.append({
            "tokens": [t[0] for t in tokens],
            "ner_tags": tags,
            "has_entities": has_entities,
        })

    return examples
```

#### 2.5 — Wire v3 into `main()`

```python
# generate_training_data.py — update argparse (~line 386)
parser.add_argument(
    "--version", default="v1", choices=["v1", "v2", "v3"],
    help="v1: original, v2: enhanced negatives/addresses, v3: + clean names",
)

# generate_training_data.py — add v3 branch in main() (~line 408)
if args.version == "v3":
    print("[generate] Generating synthetic responses (v2 — enhanced)...")
    synthetic_examples = generate_synthetic_responses_v2(groundtruth, count=800)
    print(f"  {len(synthetic_examples)} examples")

    print("[generate] Generating clean-name responses (v3 — new)...")
    clean_name_examples = generate_clean_name_responses(groundtruth, count=800)
    print(f"  {len(clean_name_examples)} examples")

    print("[generate] Generating medication list responses (v3 — new)...")
    med_list_examples = generate_medication_list_responses(groundtruth, count=400)
    print(f"  {len(med_list_examples)} examples")

    print("[generate] Generating negative examples (v2 — enhanced)...")
    negative_examples = generate_negative_examples_v2(count=500)
    print(f"  {len(negative_examples)} examples")

    # Combine (v3 adds clean_name + med_list on top of v2)
    all_examples = (
        chunk_examples + synthetic_examples
        + clean_name_examples + med_list_examples
        + negative_examples
    )
elif args.version == "v2":
    # ... existing v2 code ...
```

Also update `generate_from_chunks()` to produce clean-name variants alongside original chunks:

```python
# generate_training_data.py — add inside generate_from_chunks(), after creating
# the original example (~line 148)

# V3: also create a clean-name variant of the same chunk
if args_version == "v3":
    clean_text = text
    for name_key in ["full_name", "name"]:
        original = pii.get(name_key, "")
        if original:
            clean = strip_synthea_suffix(original)
            clean_text = clean_text.replace(original, clean)

    if clean_text != text:
        clean_pii = {
            "name": strip_synthea_suffix(pii["name"]),
            "full_name": strip_synthea_suffix(pii["full_name"]),
            "address": pii["address"],
            "dob": pii["dob"],
        }
        clean_example = create_training_example(clean_text, clean_pii)
        if clean_example:
            examples.append(clean_example)
```

---

## Phase 3 — Address False Positive Reduction

- [ ] **3.1 Hyphen preprocessing in inference**
- [ ] **3.2 Add medical-code negatives to training data**
- [ ] **3.3 Post-prediction confidence filter**

### Code Changes

#### 3.1 — `backend/app/firewall/fw_l2.py`

Add hyphen normalization before running the BERT pipeline:

```python
# fw_l2.py — BERTNERClassifier.classify() (line ~454)
# BEFORE:
def classify(self, text: str) -> list[Detection]:
    results = self._pipeline(text)
    ...

# AFTER:
# Add a helper (at class level, ~line 435):
_HYPHEN_COMPOUND = re.compile(r"([A-Z][A-Za-z]+)-([A-Z][A-Za-z]+)")

def classify(self, text: str) -> list[Detection]:
    # Normalize hyphens in compound names to prevent subword artifacts
    # e.g. "HOSPITAL-SEATTLE" → "HOSPITAL SEATTLE"
    # Track offset shifts so Detection positions map back to original text
    normalized = self._HYPHEN_COMPOUND.sub(r"\1 \2", text)

    results = self._pipeline(normalized)
    detections: list[Detection] = []

    for r in results:
        entity_group = r["entity_group"]
        if entity_group in ("NAME", "ADDRESS"):
            value, start, end = self._trim_entity(normalized, r["start"], r["end"])
            if not value.strip():
                continue

            # Map positions back to original text if lengths differ
            # (hyphen→space is same length, so positions are preserved)
            detections.append(Detection(
                entity_type=entity_group,
                value=value,
                start=start,
                end=end,
                source="ner_bert",
            ))

    return detections
```

#### 3.2 — `fw_l2_ner/scripts/generate_training_data.py`

Already partially covered by `generate_negative_examples_v2()` which includes SNOMED-CT and ICD codes. Add more specific examples:

```python
# generate_training_data.py — add to generate_negative_examples_v2() templates list

# Additional medical code negatives (Phase 3.2)
"SNOMED CT: 38341003 (Hypertension) confirmed on 2024-01-15.",
"ICD-10: E11.9 Type 2 diabetes mellitus without complications.",
"NDC: 0378-4150-01 Amlodipine Besylate 5mg Tablet.",
"CPT: 99214 Office visit, established patient, moderate complexity.",
"LOINC: 4548-4 Hemoglobin A1c in Blood.",
"RxNorm: 314076 lisinopril 10 MG Oral Tablet.",
"HCPCS: J0178 injection, aflibercept, 1 mg.",
"Encounter ID: E-2024-001542, Status: completed.",
"NPI: 1234567890, Provider specialty: Internal Medicine.",
```

#### 3.3 — `backend/app/firewall/fw_l2.py`

Add confidence threshold to filter low-confidence predictions:

```python
# fw_l2.py — BERTNERClassifier.classify() (~line 463)
# Add after entity_group check:

MIN_CONFIDENCE = 0.85

for r in results:
    entity_group = r["entity_group"]
    if entity_group in ("NAME", "ADDRESS"):
        # Skip low-confidence predictions
        if r["score"] < MIN_CONFIDENCE:
            print(f"[fw_l2] Filtered low-confidence {entity_group}: "
                  f"'{r['word']}' (score={r['score']:.3f})")
            continue

        value, start, end = self._trim_entity(text, r["start"], r["end"])
        ...
```

---

## Phase 4 — Evaluation on Production-Like Data

The current test set (1,861 examples) uses the same Synthea distribution as training. It does not catch the clean-name failure.

- [ ] **4.1 Create a production-realistic test set**
- [ ] **4.2 Add the exact failing example as a regression test**
- [ ] **4.3 Dual test reporting**

### Code Changes

#### 4.2 — `fw_l2_ner/scripts/evaluate_model.py`

Add clean-name and medication-list test cases:

```python
# evaluate_model.py — add to TEST_CASES list (~line 22)

TEST_CASES = [
    # ... existing test cases ...

    # === Production regression tests (clean names) ===

    # Exact production failure: NAME/ADDRESS confusion
    "Gregorio Orozco: amLODIPine 2.5 MG Oral Tablet, lisinopril 10 MG Oral Tablet.",

    # Missed multi-word names
    "Barbara Leontine Brakus: amLODIPine 2.5 MG Oral Tablet.",
    "Nathan Ernser: Hydrochlorothiazide 25 MG Oral Tablet.",

    # Clean names in clinical contexts (no Synthea suffixes)
    "The patient Wilfredo Fritsch was prescribed Metformin 500mg.",
    "Patient Lindsay Zieme reports no known allergies.",
    "Encounter for Corie Sofia Jast at SWEDISH EDMONDS.",

    # Medication list format (production pattern)
    "The patients with hypertension are taking:\n- Gregorio Orozco: amLODIPine 2.5 MG\n- Barbara Leontine Brakus: lisinopril 10 MG",

    # Mixed clean + Synthea names (model should handle both)
    "Adah626 Klein929 and Wilfredo Fritsch were both seen today.",
]
```

#### 4.1 — Create new production test script

New file: `fw_l2_ner/scripts/evaluate_production.py`

```python
"""Evaluate NER model on production-like LLM responses (clean names).

Generates test examples that match actual LLM output format:
- Names without Synthea numeric suffixes
- Medication lists, encounter summaries
- Multi-patient responses

Usage:
    python fw_l2_ner/scripts/evaluate_production.py --model distilbert
"""

import json
import re
from pathlib import Path
from transformers import pipeline as hf_pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GROUNDTRUTH_PATH = PROJECT_ROOT / "data" / "processed" / "phi_groundtruth.json"
MODEL_BASE = Path(__file__).parent.parent / "models"


def strip_synthea_suffix(name: str) -> str:
    return re.sub(r"\d{2,4}", "", name).strip()


def build_production_test_cases(groundtruth: dict) -> list[dict]:
    """Build test cases from ground truth using clean names."""
    cases = []
    patients = list(groundtruth.values())[:20]  # Use first 20 patients

    for patient in patients:
        clean_name = strip_synthea_suffix(patient["name"])
        clean_full = strip_synthea_suffix(patient["full_name"])
        address = patient["address"]

        # Clinical context
        cases.append({
            "text": f"The patient {clean_full} was prescribed Metformin 500mg.",
            "expected_names": [clean_full],
            "expected_addresses": [],
        })

        # Medication list
        cases.append({
            "text": f"- {clean_name}: amLODIPine 2.5 MG Oral Tablet",
            "expected_names": [clean_name],
            "expected_addresses": [],
        })

        # Full demographics
        cases.append({
            "text": f"Patient {clean_full} resides at {address}.",
            "expected_names": [clean_full],
            "expected_addresses": [address],
        })

    return cases


def evaluate(model_key: str):
    model_path = MODEL_BASE / model_key / "best"
    ner = hf_pipeline("ner", model=str(model_path),
                      tokenizer=str(model_path),
                      aggregation_strategy="simple")

    with open(GROUNDTRUTH_PATH) as f:
        groundtruth = json.load(f)

    cases = build_production_test_cases(groundtruth)

    name_tp, name_fn, name_fp = 0, 0, 0
    addr_tp, addr_fn, addr_fp = 0, 0, 0

    for case in cases:
        results = ner(case["text"])
        pred_names = [r["word"] for r in results if r["entity_group"] == "NAME"]
        pred_addrs = [r["word"] for r in results if r["entity_group"] == "ADDRESS"]

        # Score names (at least partial overlap counts as TP)
        for expected in case["expected_names"]:
            if any(expected.split()[0] in p for p in pred_names):
                name_tp += 1
            else:
                name_fn += 1
                print(f"  MISS NAME: '{expected}' in: {case['text'][:80]}")

        # Check for name FPs when none expected
        if not case["expected_names"] and pred_names:
            name_fp += len(pred_names)

    total = name_tp + name_fn
    recall = name_tp / total if total > 0 else 0
    print(f"\nProduction NAME recall: {name_tp}/{total} = {recall:.2%}")
    print(f"Production NAME FP: {name_fp}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert")
    args = parser.parse_args()
    evaluate(args.model)
```

#### 4.3 — Add CLI entry point

```toml
# fw_l2_ner/pyproject.toml — add to [project.scripts]
ner-eval-prod = "experiments.phi_ner.scripts.evaluate_production:evaluate"
```

---

## Phase 5 — Retrain and Deploy

- [ ] **5.1 Regenerate training data** with v3
- [ ] **5.2 Train DistilBERT** with Phase 1 settings
- [ ] **5.3 Evaluate** on both test sets
- [ ] **5.4 Publish** new model to W&B
- [ ] **5.5 Verify deployment**

### Commands

```bash
# 5.1 — Generate v3 training data
cd fw_l2_ner
uv run ner-generate --version v3

# 5.2 — Train with optimized settings (batch_size=32 already set in code)
uv run ner-train --model distilbert

# 5.3 — Evaluate on both test sets
uv run ner-evaluate --model distilbert
python scripts/evaluate_production.py --model distilbert

# 5.4 — Export and publish
uv run ner-export --model distilbert

# 5.5 — Verify the exact production failure is fixed
cd ../../backend
python -c "
from app.firewall.fw_l2 import FWL2
fw = FWL2(ner_backend='bert')
result = fw.validate(
    'The patients with hypertension are taking the following medications:\n'
    '- Gregorio Orozco: amLODIPine 2.5 MG Oral Tablet\n'
    '- Barbara Leontine Brakus: lisinopril 10 MG Oral Tablet\n'
    '- Nathan Ernser: Hydrochlorothiazide 25 MG Oral Tablet'
)
print(result.sanitized_text)
print()
for d in result.detections:
    print(f'  {d.entity_type:10s} | {d.source:12s} | {d.value}')
"
```

Expected output after fix:
```
The patients with hypertension are taking the following medications:
- [NAME]: amLODIPine 2.5 MG Oral Tablet
- [NAME]: lisinopril 10 MG Oral Tablet
- [NAME]: Hydrochlorothiazide 25 MG Oral Tablet

  NAME       | ner_bert     | Gregorio Orozco
  NAME       | ner_bert     | Barbara Leontine Brakus
  NAME       | ner_bert     | Nathan Ernser
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Synthea test F1 (seqeval strict) | 0.9734 | ≥ 0.985 |
| Production test F1 (clean names) | ~0.60 (estimated) | ≥ 0.95 |
| NAME entities missed (production set) | 2 of 3 | 0 |
| NAME → ADDRESS confusion | Present | Eliminated |
| ADDRESS false positive rate | 0.75% | ≤ 0.75% (no regression) |
| Inference latency P50 | 27.6 ms | ≤ 30 ms |

---

## Files Changed (Summary)

| File | Phase | Changes |
|------|-------|---------|
| `fw_l2_ner/scripts/generate_training_data.py` | 2, 3 | Add `strip_synthea_suffix()`, `generate_clean_name_responses()`, `generate_medication_list_responses()`, v3 branch in `main()`, additional medical-code negatives |
| `fw_l2_ner/scripts/train.py` | 1 | Batch size 16→32, add `oversample_entity_examples()`, add `EarlyStoppingCallback(patience=4)` |
| `fw_l2_ner/scripts/evaluate_model.py` | 4 | Add clean-name and medication-list regression test cases |
| `fw_l2_ner/scripts/evaluate_production.py` | 4 | **New file** — production-like evaluation with clean names |
| `backend/app/firewall/fw_l2.py` | 3 | Add `_HYPHEN_COMPOUND` normalization, add `MIN_CONFIDENCE` filter in `classify()` |

---

## Priority Order

1. **Phase 2** (clean-name augmentation) — fixes the production failure, highest impact
2. **Phase 1** (training pipeline) — easy wins, apply alongside Phase 2
3. **Phase 4** (production test set) — needed to validate the fix
4. **Phase 3** (address FP reduction) — quality improvement, lower urgency
5. **Phase 5** (retrain + deploy) — final step after 1–4 are complete
