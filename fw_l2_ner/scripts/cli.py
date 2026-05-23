"""CLI entry points for PII NER experiments.

Usage (from fw_l2_ner/):
    uv run ner-help                          # Show all commands
    uv run ner-generate                      # Generate training data
    uv run ner-train --model distilbert      # Train single model
    uv run ner-train --model all             # Train all 3 models
    uv run ner-evaluate --model distilbert   # Evaluate vs spaCy
    uv run ner-compare                       # Compare all trained models
    uv run ner-export --model distilbert     # Export best model to FW-L2
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


COMMANDS = {
    "Data": {
        "ner-generate":  "Generate NER training data (--version v1|v2)",
    },
    "Training": {
        "ner-train":     "Fine-tune a BERT model (--model distilbert|bert|roberta|all)",
        "ner-evaluate":  "Evaluate a trained model on test cases vs spaCy",
        "ner-compare":   "Compare all trained models side-by-side",
    },
    "Deployment": {
        "ner-export":    "Export best model to backend FW-L2",
    },
    "General": {
        "ner-help":      "Show this help message",
    },
}


def help():
    """Show available CLI commands."""
    print("\n" + "=" * 60)
    print("           PHI NER EXPERIMENT - CLI COMMANDS")
    print("=" * 60)
    print("\nUsage:  cd fw_l2_ner && uv run <command> [options]\n")

    for group, commands in COMMANDS.items():
        print(f"  {group}:")
        for name, desc in commands.items():
            print(f"    {name:20s}  {desc}")
        print()

    print("  Data versions:")
    print("    v1                   Original training data")
    print("    v2                   Enhanced: better negatives, address diversity")
    print()
    print("  Models available:")
    print("    distilbert           DistilBERT (~66M params, fastest)")
    print("    bert                 BERT base (~110M params, standard)")
    print("    roberta              RoBERTa (~125M params, best accuracy)")
    print("    all                  Train/evaluate all 3")
    print()
    print("  Examples:")
    print("    uv run ner-generate --version v2")
    print("    uv run ner-train --model distilbert")
    print("    uv run ner-evaluate --model all")
    print("    uv run ner-export --model distilbert")

    print("\n" + "=" * 60)


def generate():
    """Generate NER training data from Synthea chunks + ground truth."""
    from scripts.generate_training_data import main
    main()


def train():
    """Fine-tune a BERT model for PII NER."""
    from fw_l2_ner.scripts.train import main
    main()


def evaluate():
    """Evaluate a trained model on test cases."""
    from scripts.evaluate_model import main
    main()


def compare():
    """Compare all trained models side-by-side."""
    import argparse
    import json

    models_dir = Path(__file__).parent.parent / "models"

    print("\n" + "=" * 60)
    print("           MODEL COMPARISON")
    print("=" * 60)

    results = {}

    for model_key in ["distilbert", "bert", "roberta"]:
        trainer_state = models_dir / model_key / "best" / "trainer_state.json"
        eval_results = models_dir / model_key / "best" / "eval_results.json"

        if not (models_dir / model_key / "best").exists():
            print(f"\n  {model_key}: NOT TRAINED")
            continue

        # Try to load eval results from training output
        checkpoint_dirs = sorted(
            [d for d in (models_dir / model_key).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint")],
            key=lambda x: x.name,
        )

        # Load trainer state for best metrics
        best_state = models_dir / model_key / "best" / "trainer_state.json"
        if best_state.exists():
            with open(best_state, "r") as f:
                state = json.load(f)
            best_metrics = state.get("best_metric", "N/A")
            results[model_key] = {"f1_macro": best_metrics}
            print(f"\n  {model_key}: F1 macro = {best_metrics}")
        else:
            print(f"\n  {model_key}: trained but no metrics found")
            # Check model size
            model_size = sum(
                f.stat().st_size for f in (models_dir / model_key / "best").rglob("*") if f.is_file()
            )
            print(f"    Model size: {model_size / 1024 / 1024:.1f} MB")

    if not results:
        print("\n  No trained models found. Run: uv run ner-train --model all")

    # Run evaluation on test cases
    print(f"\n{'=' * 60}")
    print("  Running test case evaluation...")
    print(f"{'=' * 60}")

    from scripts.evaluate_model import evaluate_spacy, evaluate_finetuned

    evaluate_spacy()
    for model_key in ["distilbert", "bert", "roberta"]:
        if (models_dir / model_key / "best").exists():
            evaluate_finetuned(model_key)

    print(f"\n{'=' * 60}")


def export():
    """Export the best trained model to backend FW-L2."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Export model to FW-L2")
    parser.add_argument("--model", required=True,
                        choices=["distilbert", "bert", "roberta"],
                        help="Which model to export")
    args = parser.parse_args()

    source = Path(__file__).parent.parent / "models" / args.model / "best"
    target = PROJECT_ROOT / "backend" / "app" / "firewall" / "ner_model"

    if not source.exists():
        print(f"Error: Model not found at {source}")
        print(f"Run: uv run ner-train --model {args.model}")
        sys.exit(1)

    print(f"[export] Source: {source}")
    print(f"[export] Target: {target}")

    if target.exists():
        print(f"[export] Removing existing model at {target}")
        shutil.rmtree(target)

    shutil.copytree(source, target)

    # Calculate size
    model_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"[export] Exported {args.model} model ({model_size / 1024 / 1024:.1f} MB)")
    print(f"[export] Update FW-L2 to use: NERClassifier(model_path='{target}')")