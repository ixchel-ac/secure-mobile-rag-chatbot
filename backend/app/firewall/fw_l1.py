"""FW-L1: Query-side firewall for adversarial prompt detection.
Classifies incoming queries as safe or adversarial (C1-C5) using
an ONNX-quantized DistilBERT model. Blocks adversarial queries before
they reach the RAG pipeline.
Model loading priority:
    1. Local cached ONNX model (fw_l1/models/fw_l1.onnx)
    2. Pull from W&B artifact (fw-l1-model:latest)
    3. Fail with error
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# All possible classification labels the model can predict.
# index position matches the model output neuron order (0=safe, 1=C1, ..., 5=C5).
LABEL_LIST = ["safe", "C1", "C2", "C3", "C4", "C5"]

# Reverse lookup: converts a predicted class index back to its human-readable label.
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# Default path to the ONNX model directory, resolved relative to this file location.
# Walks up 4 levels from this file to reach the project root, then into fw_l1/models/.
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "fw_l1" / "models"


@dataclass
class FWL1Result:
    """Result of FW-L1 query classification."""
    query: str
    classification: str              # "safe", "C1", ..., "C5"
    confidence: float                # softmax probability of predicted class
    is_blocked: bool                 # True if classification != "safe"
    probabilities: dict[str, float]  # all class probabilities

    @property
    def action(self) -> str:
        # Convenience label for logging and downstream routing decisions.
        return "block" if self.is_blocked else "allow"

    def __str__(self) -> str:
        return (f"FWL1Result: {self.classification} "
                f"(confidence={self.confidence:.3f}, action={self.action})")


class FWL1:
    """On-device query classifier using ONNX Runtime."""

    # W&B artifact and project names used when the local model is missing.
    WANDB_ARTIFACT = "fw-l1-model"
    WANDB_PROJECT = "mobile-rag-firewall"

    def __init__(self, model_dir: str | Path | None = None, threshold: float = 0.5):
        from transformers import AutoTokenizer
        import onnxruntime as ort

        # Resolve the model directory: use the provided path or fall back to the default.
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        onnx_path = model_dir / "fw_l1.onnx"
        tokenizer_path = model_dir / "tokenizer"

        # If the ONNX model isn't cached locally, pull it from W&B before continuing.
        if not onnx_path.exists():
            model_dir = self._pull_from_wandb(model_dir)
            onnx_path = model_dir / "fw_l1.onnx"
            tokenizer_path = model_dir / "tokenizer"

        print(f"[fw_l1] Loading ONNX model from {onnx_path}")

        # Start an ONNX Runtime inference sessio