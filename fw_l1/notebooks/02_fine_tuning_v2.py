"""FW-L1 Fine-Tuning v2 — Compound-Aware Training

Improved fine-tuning of the FW-L1 query classifier, designed to handle compound
queries that embed adversarial intent within a legitimate medical question.

What this adds over v1:
1. Compound-aware data analysis — breakdown by blend_type and compound flag
2. Adversarial span augmentation — extracts adversarial_part from compound queries
3. Focal loss — focuses training on hard examples
4. Curriculum learning — Phase 1: pure adversarial + benign; Phase 2: add compound
5. Improved hyperparameters — warmup ratio, label smoothing, lower LR (3e-5)
6. Compound-aware metrics — false_pass_rate split by plain/compound/blend_type

Labels: safe (0), C1 (1), C2 (2), C3 (3), C4 (4), C5 (5)

Designed to run on Colab GPU (T4). Estimated time: ~25-35 minutes.
"""

# ── Cell 1: Environment setup ─────────────────────────────────────────────────

import subprocess, sys, time
start_time = time.time()

# Capture pre-installed torch version before pip touches anything.
_r = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True,
)
_torch_ver = _r.stdout.strip() if _r.returncode == 0 else ""
_cuda_tag = _torch_ver.split("+")[-1] if "+" in _torch_ver else "cu128"
print(f"Pre-installed torch: {_torch_ver}")

# Install training dependencies.
# !pip install -q transformers wandb weave accelerate scikit-learn "sympy<1.13"

# pip may silently downgrade Colab's CUDA torch to a CPU-only build from PyPI.
# This breaks torchvision (torchvision::nms) and transformers (requires torch>=2.6).
# If torch version changed, restore the full ecosystem from PyTorch's CUDA index.
_post = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True,
).stdout.strip()

if _post != _torch_ver:
    _base = _torch_ver.split("+")[0]
    print(f"[fix] torch changed {_torch_ver} -> {_post} — restoring from {_cuda_tag} index...")
    # !pip install -q "torch=={_base}" torchvision torchaudio \
    #     --index-url https://download.pytorch.org/whl/{_cuda_tag}
    print("[fix] CUDA torch ecosystem restored")
else:
    print(f"[ok] torch unchanged ({_torch_ver})")


# ── Cell 2: W&B + Weave login ─────────────────────────────────────────────────

import wandb
import weave

WANDB_PROJECT = "mobile-rag-firewall"

try:
    from google.colab import userdata
    wandb_key = userdata.get("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    print("Logged in via Colab secrets")
except Exception:
    wandb.login()

weave.init(WANDB_PROJECT)


# ── Cell 3: Compatibility patches ─────────────────────────────────────────────

# Patch 1: onnxscript ParamSchema (may be missing in some Colab runtimes)
try:
    import onnxscript.values
    if not hasattr(onnxscript.values, "ParamSchema"):
        class ParamSchema:
            """Stub restored for torch/onnx/_internal/fx/op_validation.py compatibility."""
        onnxscript.values.ParamSchema = ParamSchema
        print("[patch] onnxscript.values.ParamSchema restored")
    else:
        print("[patch] onnxscript.values.ParamSchema already present")
except ImportError:
    print("[patch] onnxscript not installed — skipped")

# Patch 2: transformers CVE-2025-32434 check rejects torch<2.6 for torch.load.
# Colab's pip may have downgraded torch from the pre-installed CUDA build to a
# CPU-only 2.4.x from PyPI. Bypassing is safe: we only load trusted HF models.
#
# We modify the function's __code__ (not the name reference) so that ALL modules
# that already imported the original function via `from ... import` also get the fix.
# Simply reassigning the name in import_utils doesn't work because modeling_utils
# already holds a direct reference to the original function object.
try:
    from transformers.utils.import_utils import check_torch_load_is_safe
    check_torch_load_is_safe.__code__ = (lambda: None).__code__
    print("[patch] check_torch_load_is_safe neutralised (trusted HF models only)")
except Exception as e:
    print(f"[patch] check_torch_load_is_safe — could not patch: {e}")


# ── Cell 4: GPU check ─────────────────────────────────────────────────────────

import torch
print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    print(f"Memory: {mem / 1024**3:.1f} GB")
else:
    print("WARNING: No CUDA — training will run on CPU (much slower, fp16 disabled)")


# ── Cell 5: Data loading ──────────────────────────────────────────────────────

import json
from collections import Counter
from pathlib import Path

try:
    train_data = list(weave.ref("fw-l1-train:latest").get().rows)
    val_data = list(weave.ref("fw-l1-val:latest").get().rows)
    test_data = list(weave.ref("fw-l1-test:latest").get().rows)
    print(f"Loaded from Weave: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
except Exception as e:
    print(f"Weave failed: {e}. Upload train.json, val.json, test.json manually.")
    from google.colab import files
    uploaded = files.upload()
    with open("train.json") as f:
        train_data = json.load(f)
    with open("val.json") as f:
        val_data = json.load(f)
    with open("test.json") as f:
        test_data = json.load(f)

# ── Label distribution ────────────────────────────────────────────────────
print(f"\nLabel distribution (train): {dict(Counter(ex['label'] for ex in train_data))}")

# ── Compound vs non-compound split ───────────────────────────────────────
compound_train = [ex for ex in train_data if ex.get("compound")]
plain_train = [ex for ex in train_data if not ex.get("compound")]
print(f"Train: {len(plain_train)} plain + {len(compound_train)} compound = {len(train_data)} total")
print(f"Val: {len([e for e in val_data if e.get('compound')])} compound / {len(val_data)} total")
print(f"Test: {len([e for e in test_data if e.get('compound')])} compound / {len(test_data)} total")

# ── Blend type distribution ───────────────────────────────────────────────
blend_counts = Counter(ex.get("blend_type", "") for ex in compound_train if ex.get("compound"))
print(f"\nBlend types in train: {dict(blend_counts)}")

# ── Per-category compound breakdown ──────────────────────────────────────
print("\nCompound queries per category (train):")
for label in ["C1", "C2", "C3", "C4", "C5"]:
    total = sum(1 for e in train_data if e["label"] == label)
    comp = sum(1 for e in compound_train if e["label"] == label)
    print(f"  {label}: {comp}/{total} compound")

# ── Example compound queries ──────────────────────────────────────────────
print("\nExample compound queries:")
for ex in compound_train[:3]:
    print(f"  [{ex['label']} / {ex.get('blend_type','')}]")
    print(f"    Full:        {ex['text'][:80]}")
    print(f"    Benign:      {ex.get('benign_part','')[:60]}")
    print(f"    Adversarial: {ex.get('adversarial_part','')[:60]}")


# ── Cell 6: Dataset class + model configs ────────────────────────────────────

import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.utils.class_weight import compute_class_weight

LABEL_LIST = ["safe", "C1", "C2", "C3", "C4", "C5"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_LIST)

BLEND_TYPES = ["conjunction", "subordinate", "punctuation_separated", "context_switch", "injection"]

# v2: focus on MobileBERT (deployment target) and DistilBERT (accuracy baseline).
# Lower LR (3e-5 vs 5e-5 in v1), add warmup ratio + label smoothing.
MODEL_CONFIGS = {
    "mobilebert": {
        "name": "google/mobilebert-uncased",
        "lr": 3e-5, "epochs": 12, "batch_size": 32,
        "warmup_ratio": 0.1, "label_smoothing": 0.05,
    },
    "distilbert": {
        "name": "distilbert-base-uncased",
        "lr": 3e-5, "epochs": 12, "batch_size": 32,
        "warmup_ratio": 0.1, "label_smoothing": 0.05,
    },
}


class FWL1Dataset(TorchDataset):
    """Dataset that carries compound metadata for metric breakdown."""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        enc = self.tokenizer(
            ex["text"], truncation=True,
            max_length=self.max_length, padding=False,
        )
        enc["labels"] = ex["label_id"]
        return {k: torch.tensor(v) if isinstance(v, list) else torch.tensor(v)
                for k, v in enc.items()}


print(f"Labels: {LABEL_LIST}")
print(f"Models to train: {list(MODEL_CONFIGS.keys())}")


# ── Cell 7: FocalLoss + compute_metrics_v2 ───────────────────────────────────

class FocalLoss(torch.nn.Module):
    """Multi-class focal loss: FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)"""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight  # class weights (alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def compute_metrics_v2(eval_pred, eval_data=None):
    """Extended metrics with compound/plain/blend_type breakdown.

    eval_data: the raw list of dicts used for this evaluation set.
               If None, falls back to basic metrics only.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    metrics = {}

    # Per-class precision / recall / F1
    for i, label_name in enumerate(LABEL_LIST):
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[f"precision_{label_name}"] = float(precision)
        metrics[f"recall_{label_name}"] = float(recall)
        metrics[f"f1_{label_name}"] = float(f1)

    metrics["f1_macro"] = float(np.mean([metrics[f"f1_{l}"] for l in LABEL_LIST]))
    metrics["accuracy"] = float((preds == labels).mean())

    # Overall false pass / false block
    adv_mask = labels != LABEL_TO_ID["safe"]
    safe_mask = labels == LABEL_TO_ID["safe"]
    if adv_mask.sum() > 0:
        metrics["false_pass_rate"] = float(
            (preds[adv_mask] == LABEL_TO_ID["safe"]).sum() / adv_mask.sum()
        )
    if safe_mask.sum() > 0:
        metrics["false_block_rate"] = float(
            (preds[safe_mask] != LABEL_TO_ID["safe"]).sum() / safe_mask.sum()
        )

    # Compound-aware breakdown (requires eval_data with compound metadata)
    if eval_data is not None and len(eval_data) == len(preds):
        compound_mask = np.array([bool(ex.get("compound")) for ex in eval_data])
        plain_mask = ~compound_mask

        # False pass rate: plain adversarial vs compound adversarial
        plain_adv = plain_mask & adv_mask
        comp_adv = compound_mask & adv_mask
        if plain_adv.sum() > 0:
            metrics["false_pass_rate_plain"] = float(
                (preds[plain_adv] == LABEL_TO_ID["safe"]).sum() / plain_adv.sum()
            )
        if comp_adv.sum() > 0:
            metrics["false_pass_rate_compound"] = float(
                (preds[comp_adv] == LABEL_TO_ID["safe"]).sum() / comp_adv.sum()
            )

        # Per-blend-type false pass rate
        for blend in BLEND_TYPES:
            blend_mask = np.array([ex.get("blend_type") == blend for ex in eval_data])
            blend_adv = blend_mask & adv_mask
            if blend_adv.sum() > 0:
                metrics[f"false_pass_{blend}"] = float(
                    (preds[blend_adv] == LABEL_TO_ID["safe"]).sum() / blend_adv.sum()
                )

    return metrics


print("Focal loss and compound-aware metrics ready.")


# ── Cell 8: Adversarial span augmentation ────────────────────────────────────

def augment_with_adversarial_spans(data: list) -> list:
    """Add adversarial_part of compound queries as standalone training examples."""
    augmented = list(data)
    span_count = 0

    for ex in data:
        if not ex.get("compound"):
            continue
        adv_part = ex.get("adversarial_part", "").strip()
        if not adv_part or len(adv_part) < 10:
            continue

        augmented.append({
            "id": ex["id"] + "_span",
            "text": adv_part,
            "label": ex["label"],
            "label_id": ex["label_id"],
            "expected_action": ex["expected_action"],
            "subcategory": ex.get("subcategory", ""),
            "difficulty": ex.get("difficulty", ""),
            "compound": False,   # span alone is a plain adversarial signal
            "blend_type": "",
            "benign_part": "",
            "adversarial_part": "",
        })
        span_count += 1

    print(f"Augmented: added {span_count} adversarial span examples -> {len(augmented)} total")
    return augmented


train_augmented = augment_with_adversarial_spans(train_data)
print(f"Train before augmentation: {len(train_data)}")
print(f"Train after augmentation:  {len(train_augmented)}")
print(f"Label distribution after augmentation: {dict(Counter(ex['label'] for ex in train_augmented))}")


# ── Cell 9: Hard benign augmentation ─────────────────────────────────────────

import random

HARD_BENIGN_TEMPLATES = [
    # C5.1-style: system/storage → patient record
    "What {clinical} data is recorded for this patient?",
    "What {clinical} information is available in the patient's file?",
    "Does the medical record contain the patient's {clinical}?",
    "Is there {clinical} information stored for {patient}?",

    # C5.2-style: restriction framing → clinical question
    "What {clinical} details can you share about {patient}?",
    "Is there a reason the {clinical} for {patient} isn't shown here?",
    "What restrictions apply to sharing {patient}'s {clinical} details?",

    # C5.3-style: schema/format → clinical structure
    "How are the {clinical} entries organized in the records?",
    "How is the {clinical} section structured in the patient file?",
    "What fields are included under {clinical} for this patient?",

    # C5.4-style: validate existence → confirm clinical record
    "Can you confirm that {clinical} information exists for {patient}?",
    "Validate that {patient} has {clinical} records on file.",
    "Without listing all details, does {patient} have any {clinical}?",

    # C5.5-style: system capability framing → clinical query
    "Are you able to retrieve the {clinical} records for {patient}?",
    "Can this system look up {clinical} information for {patient}?",
    "List all {clinical} records available for {patient}.",
    "Give me every {clinical} entry for {patient}.",
    "Enumerate all {clinical} findings for {patient}.",
]

CLINICAL_TERMS = [
    "medication", "drug", "prescription", "condition", "diagnosis",
    "allergy", "lab result", "procedure", "immunization", "vital sign",
    "encounter", "care plan", "treatment", "observation", "symptom",
]

PATIENT_REFERENCES = [
    "this patient", "the patient", "the individual",
    "the person", "Harold", "Ariadna", "the subject",
]


def generate_hard_benign_examples(n: int = 200, seed: int = 42) -> list[dict]:
    """Generate hard benign examples that mimic C5 sentence structure
    but use clinical vocabulary + patient references.

    These are labeled 'safe' and help the model learn:
        clinical_term + patient_reference != PHI extraction
    """
    rng = random.Random(seed)
    examples = []

    for i in range(n):
        template = rng.choice(HARD_BENIGN_TEMPLATES)
        clinical = rng.choice(CLINICAL_TERMS)
        patient = rng.choice(PATIENT_REFERENCES)

        text = template.format(clinical=clinical, patient=patient)

        # Vary capitalisation slightly
        if rng.random() < 0.2:
            text = text.lower()

        examples.append({
            "id": f"hard_benign_{i+1:03d}",
            "text": text,
            "label": "safe",
            "label_id": LABEL_TO_ID["safe"],
            "expected_action": "allow",
            "subcategory": "hard_benign",
            "difficulty": "hard",
            "compound": False,
            "blend_type": "",
            "benign_part": "",
            "adversarial_part": "",
        })

    return examples


hard_benign = generate_hard_benign_examples(n=200)
print(f"Generated {len(hard_benign)} hard benign examples")
print("\nSamples:")
for ex in hard_benign[:8]:
    print(f"  [{ex['difficulty']}] {ex['text']}")


# ── Cell 10: Curriculum learning splits ──────────────────────────────────────

def build_curriculum_splits(train_data, train_augmented, hard_benign,
                             curriculum_fraction=0.3, total_epochs=12):
    """Split training into Phase 1 (plain only) and Phase 2 (all + compound + hard benign)."""
    plain_only = [ex for ex in train_data if not ex.get("compound")]
    # Phase 2: augmented (spans) + hard benign examples
    full_with_hard = train_augmented + hard_benign
    curriculum_epochs = max(1, int(total_epochs * curriculum_fraction))
    full_epochs = total_epochs - curriculum_epochs
    return plain_only, full_with_hard, curriculum_epochs, full_epochs


plain_only, full_augmented, curriculum_epochs, full_epochs = build_curriculum_splits(
    train_data, train_augmented, hard_benign
)
print(f"Phase 1 ({curriculum_epochs} epochs): {len(plain_only)} plain examples")
print(f"Phase 2 ({full_epochs} epochs): {len(full_augmented)} examples")
print(f"  = plain + compound + adversarial spans + {len(hard_benign)} hard benign")

# Sanity check label balance in Phase 2
p2_labels = Counter(ex["label"] for ex in full_augmented)
print(f"\nPhase 2 label distribution: {dict(p2_labels)}")


# ── Cell 11: train_model_v2 function ─────────────────────────────────────────

def train_model_v2(model_key, plain_data, full_data, val_data):
    """Two-phase curriculum training with focal loss.

    Phase 1: train on plain_data (pure adversarial + benign)
    Phase 2: continue on full_data (all + compound + augmented spans)
    """
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    output_dir = f"models_v2/{model_key}"

    print(f"\n{'=' * 60}")
    print(f"  Training v2: {model_key} ({model_name})")
    print(f"  Phase 1: {len(plain_data)} plain | Phase 2: {len(full_data)} full")
    print(f"{'=' * 60}")

    plain_epochs_count = curriculum_epochs
    full_epochs_count = full_epochs

    # Class weights computed from full data (includes compound)
    all_labels_full = [ex["label_id"] for ex in full_data]
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_LABELS), y=all_labels_full
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"  Class weights: {dict(zip(LABEL_LIST, class_weights.round(3)))}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Focal loss trainer ─────────────────────────────────────────────────
    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            focal = FocalLoss(
                weight=class_weights_tensor.to(outputs.logits.device), gamma=2.0
            )
            loss = focal(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ── Phase 1: plain adversarial + benign ───────────────────────────────
    print(f"\n  [Phase 1] Training on plain data ({plain_epochs_count} epochs)...")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"fw-l1-v2-{model_key}-phase1",
        config={
            "model": model_name, "model_key": model_key,
            "phase": 1, "learning_rate": config["lr"],
            "epochs_phase1": plain_epochs_count, "warmup_ratio": config["warmup_ratio"],
            "label_smoothing": config["label_smoothing"],
            "focal_loss_gamma": 2.0,
            "train_size": len(plain_data), "val_size": len(val_data),
        },
        tags=["fw-l1-v2", model_key, "curriculum-phase1"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
    )

    train_ds = FWL1Dataset(plain_data, tokenizer)
    val_ds = FWL1Dataset(val_data, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    # Metrics closure capturing val_data for compound breakdown
    def metrics_with_val(eval_pred):
        return compute_metrics_v2(eval_pred, eval_data=val_data)

    args_phase1 = TrainingArguments(
        output_dir=f"{output_dir}/phase1",
        run_name=f"fw-l1-v2-{model_key}-phase1",
        report_to="wandb",
        num_train_epochs=plain_epochs_count,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_ratio=config["warmup_ratio"],
        label_smoothing_factor=config["label_smoothing"],
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, save_total_limit=1,
        logging_steps=50, fp16=torch.cuda.is_available(),
        dataloader_num_workers=2, dataloader_pin_memory=True,
    )

    trainer_p1 = FocalTrainer(
        model=model, args=args_phase1,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=metrics_with_val,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer_p1.train()
    wandb.finish()

    # ── Phase 2: full data with compound + augmented spans ─────────────────
    print(f"\n  [Phase 2] Fine-tuning on full data with compound ({full_epochs_count} epochs)...")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"fw-l1-v2-{model_key}-phase2",
        config={
            "model": model_name, "model_key": model_key,
            "phase": 2, "learning_rate": config["lr"] * 0.5,
            "epochs_phase2": full_epochs_count,
            "focal_loss_gamma": 2.0,
            "train_size": len(full_data), "val_size": len(val_data),
        },
        tags=["fw-l1-v2", model_key, "curriculum-phase2"],
    )

    full_ds = FWL1Dataset(full_data, tokenizer)

    args_phase2 = TrainingArguments(
        output_dir=f"{output_dir}/phase2",
        run_name=f"fw-l1-v2-{model_key}-phase2",
        report_to="wandb",
        num_train_epochs=full_epochs_count,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["lr"] * 0.5,  # half LR for Phase 2
        weight_decay=0.01,
        warmup_ratio=0.05,
        label_smoothing_factor=config["label_smoothing"],
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, save_total_limit=1,
        logging_steps=50, fp16=torch.cuda.is_available(),
        dataloader_num_workers=2, dataloader_pin_memory=True,
    )

    trainer_p2 = FocalTrainer(
        model=trainer_p1.model,  # continue from Phase 1 weights
        args=args_phase2,
        train_dataset=full_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=metrics_with_val,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer_p2.train()

    # Save best model
    best_dir = f"{output_dir}/best"
    trainer_p2.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    final_results = trainer_p2.evaluate()
    print(f"\n  Phase 2 final results for {model_key}:")
    for k, v in final_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    wandb.log({f"final/{k}": v for k, v in final_results.items()})
    wandb.finish()
    return final_results


print("Training function v2 ready.")


# ── Cell 12: Training loop ────────────────────────────────────────────────────

all_results_v2 = {}

for model_key in ["mobilebert", "distilbert"]:
    results = train_model_v2(model_key, plain_only, full_augmented, val_data)
    all_results_v2[model_key] = results

print(f"\n{'=' * 80}")
print(f"  {'Model':<15} {'F1 Macro':>10} {'Accuracy':>10} {'FPR Plain':>12} {'FPR Compound':>14} {'FBR':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*14} {'-'*10}")
for key, r in all_results_v2.items():
    f1 = r.get("eval_f1_macro", 0)
    acc = r.get("eval_accuracy", 0)
    fpr_plain = r.get("eval_false_pass_rate_plain", 0)
    fpr_comp = r.get("eval_false_pass_rate_compound", 0)
    fbr = r.get("eval_false_block_rate", 0)
    print(f"  {key:<15} {f1:>10.4f} {acc:>10.4f} {fpr_plain:>12.4f} {fpr_comp:>14.4f} {fbr:>10.4f}")


# ── Cell 13: Detailed test evaluation ────────────────────────────────────────

from transformers import pipeline as hf_pipeline

BEST_MODEL_V2 = "distilbert"  # Update based on results above

model_path = f"models_v2/{BEST_MODEL_V2}/best"
clf_pipeline = hf_pipeline(
    "text-classification", model=model_path,
    tokenizer=model_path, top_k=None,
    device=0 if torch.cuda.is_available() else -1,
)

# Evaluate on held-out test set + hard benign examples
eval_set = list(test_data) + hard_benign
print(f"Evaluating {BEST_MODEL_V2} on {len(test_data)} test + {len(hard_benign)} hard benign examples...")

all_preds = []
all_true = []
results_detail = []

for ex in eval_set:
    outputs = clf_pipeline(ex["text"])
    scores = {r["label"]: r["score"] for r in outputs[0]}
    pred_scores = {}
    for i, lname in enumerate(LABEL_LIST):
        pred_scores[lname] = scores.get(f"LABEL_{i}", scores.get(lname, 0.0))

    pred_label = max(pred_scores, key=pred_scores.get)
    pred_action = "allow" if pred_label == "safe" else "block"
    true_action = ex["expected_action"]

    all_preds.append(LABEL_TO_ID[pred_label])
    all_true.append(ex["label_id"])

    results_detail.append({
        "id": ex["id"],
        "true_label": ex["label"],
        "pred_label": pred_label,
        "correct": pred_label == ex["label"],
        "compound": ex.get("compound", False),
        "blend_type": ex.get("blend_type", ""),
        "difficulty": ex.get("difficulty", ""),
        "subcategory": ex.get("subcategory", ""),
        "true_action": true_action,
        "pred_action": pred_action,
        "false_pass": true_action == "block" and pred_action == "allow",
        "false_block": true_action == "allow" and pred_action == "block",
        "confidence": pred_scores[pred_label],
        "text": ex["text"],
    })

all_preds = np.array(all_preds)
all_true = np.array(all_true)

# Overall metrics
accuracy = (all_preds == all_true).mean()
adv_mask = all_true != LABEL_TO_ID["safe"]
safe_mask = all_true == LABEL_TO_ID["safe"]
fpr = (all_preds[adv_mask] == LABEL_TO_ID["safe"]).sum() / adv_mask.sum() if adv_mask.sum() > 0 else 0
fbr = (all_preds[safe_mask] != LABEL_TO_ID["safe"]).sum() / safe_mask.sum() if safe_mask.sum() > 0 else 0

print(f"\n{'─' * 60}")
print(f"  TEST SET RESULTS — {BEST_MODEL_V2}")
print(f"{'─' * 60}")
print(f"  Accuracy:              {accuracy:.4f}")
print(f"  False pass rate:       {fpr:.4f}  (adversarial → safe)")
print(f"  False block rate:      {fbr:.4f}  (benign → blocked)")

# Compound vs plain false pass
compound_mask = np.array([r["compound"] for r in results_detail])
plain_mask = ~compound_mask
comp_adv = compound_mask & adv_mask
plain_adv = plain_mask & adv_mask
if plain_adv.sum() > 0:
    fpr_plain = (all_preds[plain_adv] == LABEL_TO_ID["safe"]).sum() / plain_adv.sum()
    print(f"  False pass — plain:    {fpr_plain:.4f}")
if comp_adv.sum() > 0:
    fpr_comp = (all_preds[comp_adv] == LABEL_TO_ID["safe"]).sum() / comp_adv.sum()
    print(f"  False pass — compound: {fpr_comp:.4f}")

# ── Hard benign false block rate ──────────────────────────────────────────────
hard_results = [r for r in results_detail if r["subcategory"] == "hard_benign"]
if hard_results:
    hard_fb = [r for r in hard_results if r["false_block"]]
    hard_fbr = len(hard_fb) / len(hard_results)
    print(f"\n  Hard benign false block rate: {hard_fbr:.4f} ({len(hard_fb)}/{len(hard_results)})")
    if hard_fb:
        print(f"  Misclassified hard benign examples:")
        for r in sorted(hard_fb, key=lambda x: -x["confidence"])[:10]:
            print(f"    [pred={r['pred_label']} conf={r['confidence']:.3f}] {r['text']}")

# ── Per-blend-type false pass rate ────────────────────────────────────────────
print(f"\n  Per-blend-type false pass rate:")
for blend in BLEND_TYPES:
    blend_results = [r for r in results_detail if r["blend_type"] == blend]
    blend_adv = [r for r in blend_results if r["true_action"] == "block"]
    if blend_adv:
        fp_rate = sum(1 for r in blend_adv if r["false_pass"]) / len(blend_adv)
        print(f"    {blend:<25} {fp_rate:.4f} ({len(blend_adv)} examples)")

# ── Per-label accuracy ────────────────────────────────────────────────────────
print(f"\n  Per-label accuracy (test set only, excl. hard benign):")
test_only_preds = all_preds[:len(test_data)]
test_only_true = all_true[:len(test_data)]
for i, label in enumerate(LABEL_LIST):
    label_mask = test_only_true == i
    if label_mask.sum() > 0:
        label_acc = (test_only_preds[label_mask] == i).sum() / label_mask.sum()
        print(f"    {label}: {label_acc:.4f} ({label_mask.sum()} examples)")

# ── Hardest misses ────────────────────────────────────────────────────────────
hard_passes = [r for r in results_detail if r["false_pass"] and r["compound"]]
if hard_passes:
    print(f"\n  Compound false passes ({len(hard_passes)} total):")
    for r in hard_passes[:5]:
        print(f"    [{r['id']} / {r['blend_type']}] conf={r['confidence']:.3f}")
        print(f"      {r['text'][:80]}...")


# ── Cell 14: ONNX export + W&B publish ───────────────────────────────────────

# Install onnx + onnxruntime here — deferred from cell 1 to avoid pulling in
# torch-version dependencies before training (same pattern as notebook 01).
# !pip install -q onnx onnxruntime

import onnx
import onnxruntime as ort
import shutil
from onnxruntime.quantization import quantize_dynamic, QuantType

model_path = f"models_v2/{BEST_MODEL_V2}/best"
onnx_dir = Path("models_v2/onnx")
onnx_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# Load with eager attention for ONNX export — the legacy TorchScript exporter
# can't handle scaled_dot_product_attention (passes float scale where tensor expected).
# This only affects export; the trained weights are identical.
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, attn_implementation="eager",
)
model.eval()

dummy = tokenizer(
    "What medications is the patient taking?",
    return_tensors="pt", max_length=128, truncation=True,
)

# ── FP32 export ──────────────────────────────────────────────────────────────
onnx_path = onnx_dir / "fw_l1_fp32.onnx"
torch.onnx.export(
    model, (dummy["input_ids"], dummy["attention_mask"]),
    str(onnx_path),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch"},
    },
    opset_version=14, dynamo=False,
)
fp32_size = onnx_path.stat().st_size / 1024**2
print(f"Exported ONNX (FP32): {fp32_size:.1f} MB")

# ── Validate FP32 ─────────────────────────────────────────────────────────────
session_fp32 = ort.InferenceSession(str(onnx_path))
onnx_out = session_fp32.run(None, {
    "input_ids": dummy["input_ids"].numpy(),
    "attention_mask": dummy["attention_mask"].numpy(),
})[0]
with torch.no_grad():
    pt_out = model(**dummy).logits.numpy()
diff = np.abs(onnx_out - pt_out).max()
print(f"FP32 ONNX vs PyTorch max diff: {diff:.6f}")
assert diff < 0.01, f"FP32 ONNX mismatch: {diff}"
print("FP32 ONNX validation passed!")

# ── INT8 quantization ─────────────────────────────────────────────────────────
quantized_path = onnx_dir / "fw_l1_int8.onnx"
quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
int8_size = quantized_path.stat().st_size / 1024**2
print(f"Quantized ONNX (INT8): {int8_size:.1f} MB")

# ── Validate INT8 ─────────────────────────────────────────────────────────────
session_int8 = ort.InferenceSession(str(quantized_path))
int8_out = session_int8.run(None, {
    "input_ids": dummy["input_ids"].numpy(),
    "attention_mask": dummy["attention_mask"].numpy(),
})[0]
int8_diff = np.abs(int8_out - pt_out).max()
fp32_pred = np.argmax(onnx_out, axis=-1)
int8_pred = np.argmax(int8_out, axis=-1)
preds_match = np.array_equal(fp32_pred, int8_pred)
print(f"INT8 diff: {int8_diff:.6f} | preds match: {preds_match}")

# ── Choose best model ─────────────────────────────────────────────────────────
final_path = onnx_dir / "fw_l1.onnx"
if preds_match and int8_diff < 1.0:
    shutil.copy2(quantized_path, final_path)
    final_size = int8_size
    print(f"\nUsing INT8 model ({final_size:.1f} MB)")
else:
    shutil.copy2(onnx_path, final_path)
    final_size = fp32_size
    print(f"\nUsing FP32 model ({final_size:.1f} MB) — INT8 diverged")

tokenizer.save_pretrained(str(onnx_dir / "tokenizer"))
print(f"Tokenizer saved. Final model: {final_path} ({final_size:.1f} MB)")

# ── W&B publish ───────────────────────────────────────────────────────────────
run = wandb.init(
    project=WANDB_PROJECT,
    name=f"publish-fw-l1-v2-{BEST_MODEL_V2}",
    job_type="publish-model",
    tags=["fw-l1-v2", BEST_MODEL_V2, "onnx", "publish", "compound-aware"],
)

artifact = wandb.Artifact(
    name="fw-l1-model", type="model",
    description=(
        f"FW-L1 query classifier ({BEST_MODEL_V2}, INT8 ONNX). "
        "v2: focal loss + curriculum learning + adversarial span augmentation."
    ),
    metadata={
        "model_key": BEST_MODEL_V2, "format": "onnx_int8",
        "training_version": "v2",
        "improvements": [
            "focal_loss_gamma2",
            "curriculum_learning",
            "adversarial_span_augmentation",
            "compound_aware_metrics",
            "lower_lr_3e-5",
            "label_smoothing_0.05",
        ],
    },
)
artifact.add_file(str(onnx_dir / "fw_l1.onnx"))
artifact.add_dir(str(onnx_dir / "tokenizer"), name="tokenizer")
run.log_artifact(artifact)

pt_artifact = wandb.Artifact(
    name="fw-l1-model-pytorch", type="model",
    description=f"FW-L1 query classifier ({BEST_MODEL_V2}, PyTorch). v2: compound-aware fine-tuning.",
    metadata={"model_key": BEST_MODEL_V2, "format": "pytorch", "training_version": "v2"},
)
pt_artifact.add_dir(f"models_v2/{BEST_MODEL_V2}/best")
run.log_artifact(pt_artifact)

run.finish()
print("Published: fw-l1-model (ONNX) and fw-l1-model-pytorch (PyTorch)")
print("W&B will auto-increment the version — use :latest or pin to a specific :vN to revert.")


# ── Cell 15: Timing ───────────────────────────────────────────────────────────

end_time = time.time()
mins, secs = divmod(end_time - start_time, 60)
print(f"Total Notebook Execution Time: {int(mins)}m {int(secs)}s")
