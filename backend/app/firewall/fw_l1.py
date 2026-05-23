"""FW-L1: Query-side firewall for adversarial prompt detection.

Classifies incoming queries as safe or adversarial (C1-C5) using
an ONNX-quantized MobileBERT model.

This module is used in the backend ONLY for the /test endpoint
(evaluation profiles). The production /query endpoint does NOT
use FW-L1 — it runs on-device in the Android app.

Model loading priority:
    1. Local cached ONNX model (fw_l1/models/fw_l1.onnx)
    2. Pull from W&B artifact (fw-l1-model:latest)
    3. Fail with error
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


LABEL_LIST = ["safe", "C1", "C2", "C3", "C4", "C5"]
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# Resolve model directory — works both locally (backend/app/firewall/ → project root)
# and in Docker (/app/app/firewall/ → /app/)
_project_root = Path(__file__).parent.parent.parent.parent
_docker_root = Path(__file__).parent.parent.parent

if (_project_root / "fw_l1" / "models" / "fw_l1.onnx").exists():
    DEFAULT_MODEL_DIR = _project_root / "fw_l1" / "models"
elif (_docker_root / "fw_l1" / "models" / "fw_l1.onnx").exists():
    DEFAULT_MODEL_DIR = _docker_root / "fw_l1" / "models"
else:
    DEFAULT_MODEL_DIR = _project_root / "fw_l1" / "models"  # fallback, will trigger W&B download


@dataclass
class FWL1Result:
    """Result of FW-L1 query classification."""

    query: str
    classification: str         # "safe", "C1", ..., "C5"
    confidence: float           # softmax probability of predicted class
    is_blocked: bool            # True if classification != "safe"
    probabilities: dict[str, float]  # all class probabilities
    # Strip fields — populated when a compound query has adversarial parts removed
    stripped_query: str | None = None
    stripped_parts: list[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        if self.stripped_query is not None:
            return "strip"
        return "block" if self.is_blocked else "allow"

    @property
    def is_stripped(self) -> bool:
        return self.stripped_query is not None

    def __str__(self) -> str:
        extra = f", stripped_query='{self.stripped_query}'" if self.is_stripped else ""
        return (f"FWL1Result: {self.classification} "
                f"(confidence={self.confidence:.3f}, action={self.action}{extra})")


class FWL1:
    """Query classifier using ONNX Runtime.

    Used in the backend /test endpoint for evaluation only.
    Production deployment runs on-device (Android).
    """

    WANDB_ARTIFACT = "fw-l1-model"
    WANDB_PROJECT = "mobile-rag-firewall"

    def __init__(self, model_dir: str | Path | None = None, threshold: float = 0.65):
        """Load FW-L1 ONNX model.

        Args:
            model_dir: Directory containing fw_l1.onnx + tokenizer/.
                       If None, uses default (fw_l1/models/).
            threshold: Minimum confidence to classify as adversarial.
                       Below this, defaults to "safe" (reduces false blocks).
                       Set to 0.65 to avoid blocking borderline-uncertain predictions
                       (e.g., benign queries with clinical vocabulary that superficially
                       resembles enumeration-style PII extraction).
        """
        from transformers import AutoTokenizer
        import onnxruntime as ort

        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        onnx_path = model_dir / "fw_l1.onnx"
        tokenizer_path = model_dir / "tokenizer"

        # Try local, then W&B
        if not onnx_path.exists():
            model_dir = self._pull_from_wandb(model_dir)
            onnx_path = model_dir / "fw_l1.onnx"
            tokenizer_path = model_dir / "tokenizer"

        print(f"[fw_l1] Loading ONNX model from {onnx_path}")
        self._session = ort.InferenceSession(str(onnx_path))
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self._threshold = threshold
        print(f"[fw_l1] FW-L1 loaded (threshold={threshold})")

    def _pull_from_wandb(self, target_dir: Path) -> Path:
        """Download model from W&B artifact."""
        import wandb

        print(f"[fw_l1] Downloading {self.WANDB_ARTIFACT}:latest from W&B")
        run = wandb.init(project=self.WANDB_PROJECT, job_type="pull-model")
        artifact = run.use_artifact(f"{self.WANDB_ARTIFACT}:latest")
        target_dir.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(target_dir))
        run.finish()
        return target_dir

    # Sentence boundary pattern: split after . ? ! ; followed by whitespace.
    # Keeps the delimiter attached to the preceding sentence.
    _SENT_SPLIT = re.compile(r'(?<=[.?!;])\s+')

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences on . ? ! ; boundaries.

        Returns a list of non-empty sentence strings. If no split points
        are found, returns the original text as a single-element list.
        """
        parts = [s.strip() for s in self._SENT_SPLIT.split(text) if s.strip()]
        return parts if parts else [text]

    def classify_and_strip(self, query: str) -> FWL1Result:
        """Classify a query; if adversarial with multiple sentences, strip the bad parts.

        Flow:
            1. Classify the whole query.
            2. If safe → return as-is (allow).
            3. If adversarial + single sentence → return as-is (block).
            4. If adversarial + multiple sentences → classify each sentence:
               - Keep safe sentences, strip adversarial ones.
               - If at least one safe sentence remains → action="strip".
               - If no safe sentences remain → action="block".
        """
        result = self.classify(query)

        # Safe query — nothing to strip
        if not result.is_blocked:
            return result

        # Try sentence-level stripping
        sentences = self._split_sentences(query)
        if len(sentences) <= 1:
            return result  # single sentence, can't strip

        safe_parts: list[str] = []
        stripped_parts: list[str] = []

        for sent in sentences:
            sent_result = self.classify(sent)
            if sent_result.is_blocked:
                stripped_parts.append(sent)
            else:
                safe_parts.append(sent)

        if not safe_parts:
            return result  # all sentences adversarial — block

        # At least one safe sentence — strip and allow
        stripped_query = " ".join(safe_parts)
        return FWL1Result(
            query=query,
            classification=result.classification,
            confidence=result.confidence,
            is_blocked=False,
            probabilities=result.probabilities,
            stripped_query=stripped_query,
            stripped_parts=stripped_parts,
        )

    def classify(self, query: str) -> FWL1Result:
        """Classify a query as safe or adversarial.

        Args:
            query: The user's input query.

        Returns:
            FWL1Result with classification, confidence, and block decision.
        """
        # Tokenize
        inputs = self._tokenizer(
            query, return_tensors="np",
            max_length=128, truncation=True, padding=True,
        )

        # ONNX inference
        logits = self._session.run(None, {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        })[0]

        # Softmax
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = (exp / exp.sum(axis=-1, keepdims=True))[0]

        pred_id = int(probs.argmax())
        pred_label = ID_TO_LABEL[pred_id]
        confidence = float(probs[pred_id])

        # Apply threshold: if adversarial but low confidence, default to safe
        is_blocked = pred_label != "safe" and confidence >= self._threshold

        probabilities = {ID_TO_LABEL[i]: float(p) for i, p in enumerate(probs)}

        return FWL1Result(
            query=query,
            classification=pred_label,
            confidence=confidence,
            is_blocked=is_blocked,
            probabilities=probabilities,
        )
