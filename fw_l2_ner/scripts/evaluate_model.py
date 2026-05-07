"""Evaluate a trained PHI NER model on test cases and compare with spaCy.

Runs both the fine-tuned model and spaCy on the same inputs to show
the improvement.

Usage:
    python fw_l2_ner/scripts/evaluate_model.py --model distilbert
"""

import argparse
import json
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_BASE = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

LABEL_LIST = ["O", "B-NAME", "I-NAME", "B-ADDRESS", "I-ADDRESS"]

TEST_CASES = [
    # Synthea names (spaCy misses these)
    "The patient Adah626 Flo729 Klein929 was seen today.",
    "Patient: Ahmad985 Brent147 Stracke611, DOB: 1980-01-05.",

    # Full address strings
    "Address: 308 Deckow Union, Pasco, Washington 99301.",
    "Lives at 885 Crona Underpass Apt 22, Pasco, Washington 99301.",

    # Mixed PHI in clinical context
    "Adah626 Klein929 (SSN: 999-83-1042) was prescribed Metformin.",
    "Patient Kelly223 Predovic534 reports pain at 308 Deckow Union, Pasco.",

    # Clean text (should have no detections)
    "The patient is taking Aspirin 81mg daily for cardiovascular protection.",
    "Diagnosed with essential hypertension on 2020-01-15.",

    # Realistic LLM response
    "Based on the records, Marisol435 Páez758 lives at 554 Yost Underpass Unit 38, Yakima, Washington 98942.",
]


def evaluate_finetuned(model_key: str):
    """Evaluate the fine-tuned model."""
    model_path = MODEL_BASE / model_key / "best"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Run: python fw_l2_ner/scripts/train.py --model {model_key}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Fine-tuned: {model_key}")
    print(f"{'=' * 60}")

    ner = pipeline(
        "ner",
        model=str(model_path),
        tokenizer=str(model_path),
        aggregation_strategy="simple",
    )

    for text in TEST_CASES:
        results = ner(text)
        entities = [
            f"{r['entity_group']}:{r['word']}" for r in results
            if r["entity_group"] != "O"
        ]
        status = ", ".join(entities) if entities else "(clean)"
        print(f"  {status:50s} | {text[:60]}...")


def evaluate_spacy():
    """Evaluate spaCy for comparison."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    print(f"\n{'=' * 60}")
    print(f"  spaCy en_core_web_sm (current)")
    print(f"{'=' * 60}")

    for text in TEST_CASES:
        doc = nlp(text)
        entities = [
            f"{ent.label_}:{ent.text}" for ent in doc.ents
            if ent.label_ in ("PERSON", "GPE", "LOC", "FAC")
        ]
        status = ", ".join(entities) if entities else "(clean)"
        print(f"  {status:50s} | {text[:60]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert",
                        choices=["distilbert", "bert", "roberta", "all"])
    args = parser.parse_args()

    evaluate_spacy()

    models = ["distilbert", "bert", "roberta"] if args.model == "all" else [args.model]
    for model_key in models:
        evaluate_finetuned(model_key)


if __name__ == "__main__":
    main()