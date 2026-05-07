"""CLI entry points for FW-L1 experiment scripts.

Golden set generation (adversarial + benign) is handled by the backend CLI:
    cd backend && uv run generate-adversarial-queries
    cd backend && uv run generate-benign-queries

Usage (from fw_l1/):
    uv run l1-help                           # Show all commands
    uv run l1-generate                       # Combine → train/val/test + Weave publish
    uv run l1-train --model mobilebert       # Train single model
    uv run l1-train --model all              # Train all 3 models
    uv run l1-evaluate --model mobilebert    # Evaluate on test set
    uv run l1-export --model mobilebert      # Export to ONNX + INT8 + deploy
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


COMMANDS = {
    "Data": {
        "l1-generate":        "Combine adversarial + benign → train/val/test + Weave",
    },
    "Training": {
        "l1-train":           "Fine-tune a model (--model mobilebert|distilbert|tinybert|all)",
        "l1-evaluate":        "Evaluate a trained model on test set",
    },
    "Deployment": {
        "l1-export":          "Export to ONNX + INT8 quantization → fw_l1/models/",
    },
    "General": {
        "l1-help":            "Show this help message",
    },
}


def help():
    """Show available CLI commands."""
    print("\n" + "=" * 60)
    print("           FW-L1 EXPERIMENT - CLI COMMANDS")
    print("=" * 60)
    print("\nUsage:  cd fw_l1 && uv run <command> [options]\n")

    for group, commands in COMMANDS.items():
        print(f"  {group}:")
        for name, desc in commands.items():
            print(f"    {name:25s}  {desc}")
        print()

    print("  Golden set generation (run from backend/):")
    print("    uv run generate-adversarial-queries")
    print("    uv run generate-benign-queries")
    print()
    print("  Models available:")
    print("    mobilebert           MobileBERT (~25M params, deployment target)")
    print("    distilbert           DistilBERT (~66M params, accuracy baseline)")
    print("    tinybert             TinyBERT (~14.5M params, smallest)")
    print("    all                  Train/evaluate all 3")
    print()
    print("  Examples:")
    print("    uv run l1-generate")
    print("    uv run l1-train --model mobilebert")
    print("    uv run l1-evaluate --model all")
    print("    uv run l1-export --model mobilebert")

    print("\n" + "=" * 60)


def generate():
    """Combine adversarial + benign → train/val/test splits + publish to Weave."""
    from scripts.generate_training_data import main
    main()


def train():
    """Fine-tune a BERT model for query classification."""
    from scripts.train import main
    main()


def evaluate():
    """Evaluate a trained model on the test set."""
    from scripts.evaluate_model import main
    main()


def export():
    """Export best model to ONNX + INT8 and copy to fw_l1/models/."""
    from scripts.export_onnx import main
    main()