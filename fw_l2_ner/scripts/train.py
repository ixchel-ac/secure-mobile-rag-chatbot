"""Fine-tune BERT-like models for PII NER.

Supports 3 model architectures for comparison:
  1. distilbert-base-uncased   (~66M params, ~260MB, fastest)
  2. bert-base-uncased         (~110M params, ~440MB, baseline)
  3. roberta-base              (~125M params, ~500MB, best for NER)

Usage:
    python fw_l2_ner/scripts/train.py --model distilbert
    python fw_l2_ner/scripts/train.py --model bert
    python fw_l2_ner/scripts/train.py --model roberta
    python fw_l2_ner/scripts/train.py --model all  # trains all 3

Results are logged to W&B for comparison.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import wandb
import weave
from dotenv import load_dotenv
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env", override=True)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_BASE = Path(__file__).parent.parent / "models"

# Model configurations
MODEL_CONFIGS = {
    "distilbert": {
        "name": "distilbert-base-uncased",
        "lr": 5e-5,
        "epochs": 10,
        "batch_size": 16,
        "description": "DistilBERT (~66M params, fastest inference)",
    },
    "bert": {
        "name": "bert-base-uncased",
        "lr": 3e-5,
        "epochs": 10,
        "batch_size": 16,
        "description": "BERT base (~110M params, standard baseline)",
    },
    "roberta": {
        "name": "roberta-base",
        "lr": 2e-5,
        "epochs": 10,
        "batch_size": 16,
        "description": "RoBERTa (~125M params, best NER performance)",
    },
}

# NER label scheme
LABEL_LIST = ["O", "B-NAME", "I-NAME", "B-ADDRESS", "I-ADDRESS"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}


class PIINERDataset(Dataset):
    """Token classification dataset for PII NER."""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # Special tokens
            elif word_id != previous_word_id:
                # First subword of a word
                label_str = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                labels.append(LABEL_TO_ID.get(label_str, 0))
            else:
                # Continuation subword — use I- tag if entity, else -100
                label_str = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                if label_str.startswith("B-"):
                    labels.append(LABEL_TO_ID.get("I-" + label_str[2:], 0))
                elif label_str.startswith("I-"):
                    labels.append(LABEL_TO_ID.get(label_str, 0))
                else:
                    labels.append(-100)
            previous_word_id = word_id

        encoding["labels"] = labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


def compute_metrics(eval_pred):
    """Compute precision, recall, F1 per entity type."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Flatten and filter out -100 (special tokens)
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    metrics = {}

    # Overall accuracy (excluding O)
    entity_mask = true_labels != LABEL_TO_ID["O"]
    if entity_mask.sum() > 0:
        entity_correct = (pred_labels[entity_mask] == true_labels[entity_mask]).sum()
        metrics["entity_accuracy"] = float(entity_correct / entity_mask.sum())

    # Per entity type
    for entity in ["NAME", "ADDRESS"]:
        b_id = LABEL_TO_ID.get(f"B-{entity}", -1)
        i_id = LABEL_TO_ID.get(f"I-{entity}", -1)

        true_entity = np.isin(true_labels, [b_id, i_id])
        pred_entity = np.isin(pred_labels, [b_id, i_id])

        tp = (true_entity & pred_entity).sum()
        fp = (~true_entity & pred_entity).sum()
        fn = (true_entity & ~pred_entity).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"precision_{entity}"] = precision
        metrics[f"recall_{entity}"] = recall
        metrics[f"f1_{entity}"] = f1

    # Overall F1 (macro average of entity types)
    entity_f1s = [metrics.get(f"f1_{e}", 0) for e in ["NAME", "ADDRESS"]]
    metrics["f1_macro"] = np.mean(entity_f1s)

    return metrics


def load_data():
    """Load train, validation, and test data."""
    with open(DATA_DIR / "train.json", "r") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "val.json", "r") as f:
        val_data = json.load(f)

    test_path = DATA_DIR / "test.json"
    test_data = []
    if test_path.exists():
        with open(test_path, "r") as f:
            test_data = json.load(f)

    return train_data, val_data, test_data


# ── Weave Model + Scorers for leaderboard ────────────────────────────


class NERModel(weave.Model):
    """Wraps a trained NER model for Weave evaluation."""

    model_key: str = ""
    model_path: str = ""
    _pipeline: object = None

    def _ensure_loaded(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "ner",
                model=self.model_path,
                tokenizer=self.model_path,
                aggregation_strategy="simple",
            )

    @weave.op
    def predict(self, tokens: list[str], ner_tags: list[str]) -> dict:
        """Run NER prediction on tokenized text."""
        self._ensure_loaded()

        text = " ".join(tokens)
        results = self._pipeline(text)

        pred_entities = [
            {"entity": r["entity_group"], "word": r["word"], "score": float(r["score"])}
            for r in results
            if r["entity_group"] != "O"
        ]

        # Count predicted entity types
        pred_names = sum(1 for e in pred_entities if e["entity"] == "NAME")
        pred_addrs = sum(1 for e in pred_entities if e["entity"] == "ADDRESS")

        # Count true entity types
        true_names = sum(1 for t in ner_tags if t in ("B-NAME", "I-NAME"))
        true_addrs = sum(1 for t in ner_tags if t in ("B-ADDRESS", "I-ADDRESS"))

        return {
            "predicted_entities": pred_entities,
            "pred_name_count": pred_names,
            "pred_address_count": pred_addrs,
            "true_name_count": true_names,
            "true_address_count": true_addrs,
            "has_entities": any(t != "O" for t in ner_tags),
        }


@weave.op
def ner_scorer(output: dict) -> dict:
    """Score NER predictions: did the model find the entities?"""
    pred_names = output.get("pred_name_count", 0)
    pred_addrs = output.get("pred_address_count", 0)
    true_names = output.get("true_name_count", 0)
    true_addrs = output.get("true_address_count", 0)
    has_entities = output.get("has_entities", False)

    # Name detection
    name_detected = 1.0 if (true_names > 0 and pred_names > 0) else (1.0 if true_names == 0 else 0.0)

    # Address detection
    addr_detected = 1.0 if (true_addrs > 0 and pred_addrs > 0) else (1.0 if true_addrs == 0 else 0.0)

    # False positives
    name_fp = 1.0 if (true_names == 0 and pred_names > 0) else 0.0
    addr_fp = 1.0 if (true_addrs == 0 and pred_addrs > 0) else 0.0

    return {
        "name_detected": name_detected,
        "address_detected": addr_detected,
        "name_false_positive": name_fp,
        "address_false_positive": addr_fp,
    }


def train_model(model_key: str, train_data: list, val_data: list):
    """Train a single model configuration."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    output_dir = OUTPUT_BASE / model_key

    print(f"\n{'=' * 60}")
    print(f"  Training: {model_key} ({model_name})")
    print(f"  {config['description']}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"{'=' * 60}")

    # Initialize W&B run
    wandb.init(
        project="mobile-rag-firewall",
        name=f"pii-ner-{model_key}",
        config={
            "model": model_name,
            "model_key": model_key,
            "learning_rate": config["lr"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "train_size": len(train_data),
            "val_size": len(val_data),
            "labels": LABEL_LIST,
        },
        tags=["pii-ner", model_key],
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # Create datasets
    train_dataset = PIINERDataset(train_data, tokenizer)
    val_dataset = PIINERDataset(val_data, tokenizer)

    # Data collator handles padding
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"pii-ner-{model_key}",
        report_to="wandb",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Final evaluation
    eval_results = trainer.evaluate()
    print(f"\n  Final results for {model_key}:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # Log final metrics to W&B
    wandb.log({f"final/{k}": v for k, v in eval_results.items()})
    wandb.finish()

    return eval_results


async def run_weave_evaluation(model_key: str, test_data: list):
    """Run Weave evaluation on the test set for leaderboard tracking."""
    model_path = str(OUTPUT_BASE / model_key / "best")

    if not Path(model_path).exists():
        print(f"  [weave] Skipping {model_key} — model not found")
        return None

    print(f"\n  [weave] Running Weave evaluation for {model_key}...")

    # Publish test dataset
    test_dataset = weave.Dataset(name="phi-ner-test", rows=test_data)
    weave.publish(test_dataset)

    # Create Weave model
    model = NERModel(
        name=f"phi-ner-{model_key}",
        model_key=model_key,
        model_path=model_path,
    )

    # Run evaluation
    evaluation = weave.Evaluation(
        name=f"ner-eval-{model_key}",
        dataset=test_dataset,
        scorers=[ner_scorer],
        metadata={"model_key": model_key},
    )

    results = await evaluation.evaluate(model)

    # Print summary
    scorer_results = results.get("ner_scorer", {})
    name_det = scorer_results.get("name_detected", {}).get("mean", 0)
    addr_det = scorer_results.get("address_detected", {}).get("mean", 0)
    name_fp = scorer_results.get("name_false_positive", {}).get("mean", 0)
    addr_fp = scorer_results.get("address_false_positive", {}).get("mean", 0)

    print(f"  [weave] {model_key}: name_detected={name_det:.2%}, address_detected={addr_det:.2%}")
    print(f"  [weave] {model_key}: name_fp={name_fp:.2%}, address_fp={addr_fp:.2%}")

    return results


def main():
    import asyncio

    parser = argparse.ArgumentParser(description="Train PII NER model")
    parser.add_argument(
        "--model", default="distilbert",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model to train (default: distilbert)",
    )
    parser.add_argument(
        "--skip-weave", action="store_true",
        help="Skip Weave evaluation after training",
    )
    args = parser.parse_args()

    # Initialize Weave
    wandb_project = os.getenv("WANDB_PROJECT", "mobile-rag-firewall")
    if not args.skip_weave:
        weave.init(wandb_project)

    # Load data
    print("[train] Loading data...")
    train_data, val_data, test_data = load_data()
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    # Train
    models_to_train = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for model_key in models_to_train:
        results = train_model(model_key, train_data, val_data)
        all_results[model_key] = results

    # Run Weave evaluation on test set
    if not args.skip_weave and test_data:
        print(f"\n{'=' * 60}")
        print(f"  WEAVE EVALUATION ON TEST SET")
        print(f"{'=' * 60}")

        for model_key in models_to_train:
            asyncio.run(run_weave_evaluation(model_key, test_data))

    # Print comparison
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"  MODEL COMPARISON (validation set)")
        print(f"{'=' * 60}")
        print(f"  {'Model':<15} {'F1 Macro':>10} {'F1 NAME':>10} {'F1 ADDR':>10}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
        for model_key, results in all_results.items():
            f1_macro = results.get("eval_f1_macro", 0)
            f1_name = results.get("eval_f1_NAME", 0)
            f1_addr = results.get("eval_f1_ADDRESS", 0)
            print(f"  {model_key:<15} {f1_macro:>10.4f} {f1_name:>10.4f} {f1_addr:>10.4f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()