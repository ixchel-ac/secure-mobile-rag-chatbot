"""Combine adversarial + benign queries into train/val/test splits.

Input:
    data/golden_sets/adversarial_queries.json  (1,000 queries, C1-C5)
    data/golden_sets/benign_queries.json        (1,000 queries, safe)

Output:
    fw_l1/data/train.json  (70%, ~1,400 examples)
    fw_l1/data/val.json    (15%, ~300 examples)
    fw_l1/data/test.json   (15%, ~300 examples)
    Published to Weave: fw-l1-train, fw-l1-val, fw-l1-test

Usage:
    cd fw_l1
    uv run l1-generate
"""

import json
import os
import random
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env", override=True)

SEED = 42
ADVERSARIAL_PATH = _project_root / "data" / "golden_sets" / "adversarial_queries.json"
BENIGN_PATH = _project_root / "data" / "golden_sets" / "benign_queries.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

LABEL_MAP = {"safe": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def main():
    random.seed(SEED)

    # Load both query sets
    if not ADVERSARIAL_PATH.exists():
        print(f"Error: {ADVERSARIAL_PATH} not found")
        print("Run: cd backend && uv run generate-adversarial-queries")
        raise SystemExit(1)

    if not BENIGN_PATH.exists():
        print(f"Error: {BENIGN_PATH} not found")
        print("Run: cd backend && uv run generate-benign-queries")
        raise SystemExit(1)

    adversarial = load_queries(ADVERSARIAL_PATH)
    benign = load_queries(BENIGN_PATH)

    print(f"[generate] Loaded {len(adversarial)} adversarial queries")
    print(f"[generate] Loaded {len(benign)} benign queries")

    # Create unified training format
    examples = []
    for q in adversarial + benign:
        category = q["category"]
        examples.append({
            "id": q["id"],
            "text": q["query"],
            "label": category,
            "label_id": LABEL_MAP[category],
            "expected_action": q["expected_action"],
            "subcategory": q.get("subcategory", ""),
            "difficulty": q.get("difficulty", ""),
        })

    # Stratified split: 70% train, 15% val, 15% test
    labels = [ex["label"] for ex in examples]
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        examples, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    # Print distribution
    print(f"\n[generate] Split distribution:")
    for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        counts = Counter(ex["label"] for ex in data)
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"  {name}: {len(data)} examples ({dist})")

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")

    # Publish to Weave
    wandb_project = os.getenv("WANDB_PROJECT", "mobile-rag-firewall")
    try:
        import weave
        weave.init(wandb_project)
        for name, data in [("fw-l1-train", train_data), ("fw-l1-val", val_data), ("fw-l1-test", test_data)]:
            dataset = weave.Dataset(name=name, rows=data)
            weave.publish(dataset)
        print(f"\n[generate] Published to Weave: fw-l1-train, fw-l1-val, fw-l1-test")
    except Exception as e:
        print(f"\n[generate] Weave publish failed: {e}")


if __name__ == "__main__":
    main()