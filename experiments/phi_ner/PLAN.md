# PHI NER Experiment — Academic Evaluation Plan

## Overview

This plan covers a rigorous academic evaluation of fine-tuned BERT models for PHI NER in the FW-L2 response firewall. It builds on the existing training pipeline and extends it with proper evaluation methodology.

## Notebooks

| Notebook | Focus | Compute |
|----------|-------|---------|
| `01_training.ipynb` | Model training (existing, updated) | GPU (Colab) |
| `02_evaluation.ipynb` | Entity-level evaluation + cross-validation | GPU (Colab) |
| `03_error_analysis.ipynb` | Error categorization + failure modes | CPU |
| `04_ablation.ipynb` | Data ablation + hyperparameter sensitivity | GPU (Colab) |
| `05_baselines.ipynb` | Comparison with established NER systems | CPU/GPU |

## Scripts (local)

| Script | Purpose |
|--------|---------|
| `generate_training_data.py` | Updated: v1/v2 data generation (existing) |
| `train.py` | Updated: seqeval + entity-level metrics (existing) |
| `error_analysis.py` | New: categorize and analyze errors |
| `cross_validate.py` | New: k-fold cross-validation |

---

## Phase 1: Evaluation Infrastructure

### 1.1 Entity-Level Metrics (seqeval)

**Where**: Update `train.py` + `02_evaluation.ipynb`

Current metrics are token-level (each token scored independently). Academic NER requires entity-level evaluation where the entire span must match.

```
Token-level:  B-NAME correct, I-NAME correct → 2/2 tokens correct
Entity-level: "Adah626 Klein929" fully matched → 1 entity correct
              "Adah626" only (missed Klein929) → 0 entities correct (strict)
```

**Metrics to report**:
- Entity-level precision, recall, F1 per type (NAME, ADDRESS)
- Strict match (exact boundaries) vs relaxed match (overlap)
- Micro F1 (weighted by entity count) and macro F1 (equal weight per type)

**Dependencies**: `pip install seqeval`

### 1.2 Update compute_metrics

Replace token-level F1 with seqeval-based entity-level F1:

```python
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

def compute_metrics(eval_pred):
    # Convert token predictions back to BIO tag sequences
    # Use seqeval for entity-level scoring
    ...
```

---

## Phase 2: Cross-Validation

### 2.1 K-Fold Cross-Validation

**Where**: `cross_validate.py` + `02_evaluation.ipynb`

**Method**: 5-fold stratified cross-validation on the combined train+val set (holding out test for final evaluation).

```
Fold 1: [████████████████░░░░] Train on 80%, validate on 20%
Fold 2: [████████████░░░░████] Rotate validation slice
...
Fold 5: [░░░░████████████████] Last rotation
```

**Report**: Mean ± std for each metric across 5 folds.

### 2.2 Multiple Random Seeds

For each model, train with 3 different seeds and report:
- Mean ± standard deviation of F1 macro
- Statistical significance test (McNemar's) between DistilBERT vs BERT

---

## Phase 3: Ablation Study

### 3.1 Data Ablations

**Where**: `04_ablation.ipynb`

| Experiment | Training Data | What it shows |
|------------|--------------|---------------|
| `full_v1` | V1 complete | Baseline |
| `full_v2` | V2 complete | Does enhanced data help? |
| `no_synthetic` | Chunks only (no synthetic LLM responses) | Value of synthetic examples |
| `no_negatives` | No negative examples | Impact on false positive rate |
| `chunks_only` | Only Synthea chunks, no augmentation | Minimum viable data |
| `names_only` | Only NAME labels (ADDRESS masked to O) | Per-entity learnability |
| `addresses_only` | Only ADDRESS labels (NAME masked to O) | Per-entity learnability |

### 3.2 Training Ablations

| Experiment | Change | What it shows |
|------------|--------|---------------|
| `epochs_3` | 3 epochs | Is early stopping sufficient? |
| `epochs_5` | 5 epochs | Sweet spot? |
| `epochs_10` | 10 epochs (current) | Baseline |
| `lr_1e5` | Learning rate 1e-5 | Sensitivity to LR |
| `lr_5e5` | Learning rate 5e-5 | Sensitivity to LR |
| `batch_8` | Batch size 8 | Sensitivity to batch size |
| `batch_32` | Batch size 32 | Sensitivity to batch size |
| `no_weight_decay` | weight_decay=0 | Regularization impact |

### 3.3 Class Imbalance Ablations

| Experiment | Method | What it shows |
|------------|--------|---------------|
| `baseline` | Standard cross-entropy | Current approach |
| `weighted_loss` | Class-weighted cross-entropy | Does weighting help? |
| `focal_loss` | Focal loss (gamma=2) | Better for rare entities? |
| `oversampled` | Oversample entity-rich examples | Balance via data |

---

## Phase 4: Error Analysis

### 4.1 Error Categorization

**Where**: `error_analysis.py` + `03_error_analysis.ipynb`

For each error in the test set, categorize:

**False Negatives (missed entities)**:
| Category | Example | Root cause |
|----------|---------|------------|
| Short names | "Jo Klein" | Too few tokens |
| Non-Synthea names | "John Smith" | No digit pattern |
| P.O. box addresses | "P.O. Box 123" | Unusual format |
| Partial addresses | "Apt 22" | Fragment only |
| Names in context | "prescribed to Klein" | Unusual position |

**False Positives (incorrect detections)**:
| Category | Example | Root cause |
|----------|---------|------------|
| List numbers | "1." as ADDRESS | Number near text |
| Drug names | "Aspirin" as NAME | Capitalized word |
| Clinical codes | "I10" as NAME | Short code |
| Dates | "2025-01-15" as ADDRESS | Number pattern |
| Vitals | "120/80" as ADDRESS | Number pattern |

### 4.2 Confusion Matrix

Build a confusion matrix at entity level:
- NAME predicted as ADDRESS (and vice versa)
- O predicted as NAME/ADDRESS (false positives)
- NAME/ADDRESS predicted as O (false negatives)

### 4.3 Error vs Text Properties

Analyze errors correlated with:
- Text length (short vs long responses)
- Entity density (many entities vs sparse)
- Entity position (beginning vs middle vs end)
- Section type (DEMOGRAPHICS vs MEDICATIONS vs clinical narrative)

---

## Phase 5: Baseline Comparison

### 5.1 Systems to Compare

**Where**: `05_baselines.ipynb`

| System | Type | Why |
|--------|------|-----|
| **Regex only** | Rule-based | Lower bound (current FW-L2 without NER) |
| **spaCy `en_core_web_sm`** | Pre-trained NER | Current FW-L2 base (12MB) |
| **spaCy `en_core_web_lg`** | Pre-trained NER | Larger spaCy (560MB) |
| **Presidio** (Microsoft) | PII detection | Purpose-built system |
| **Fine-tuned DistilBERT** | Our model | Main contribution |
| **Fine-tuned BERT** | Our model | Comparison |
| **Fine-tuned RoBERTa** | Our model | Comparison |
| **BioClinicalBERT** | Clinical pre-training | Domain-specific baseline |

### 5.2 BioClinicalBERT

Fine-tune `emilyalsentzer/Bio_ClinicalBERT` — pre-trained on MIMIC-III clinical notes. This tests whether clinical pre-training improves PHI detection vs generic BERT.

Add to `MODEL_CONFIGS`:
```python
"bioclinicalbert": {
    "name": "emilyalsentzer/Bio_ClinicalBERT",
    "lr": 2e-5,
    "epochs": 10,
    "batch_size": 16,
}
```

### 5.3 Presidio Baseline

```python
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
results = analyzer.analyze(text=response, language="en")
```

---

## Phase 6: Latency Profiling

### 6.1 Inference Benchmarks

**Where**: `02_evaluation.ipynb`

For each model, measure on 1000 test examples:
- P50 / P95 / P99 latency
- Throughput (examples/second)
- Latency vs text length (scatter plot)
- Memory footprint (peak RSS)

### 6.2 CPU vs GPU

Compare inference on:
- Apple M-series CPU (your machine)
- Google Colab T4 GPU
- CPU-only server (Docker container)

---

## Phase 7: Generalization Testing

### 7.1 Out-of-Distribution Evaluation

Test the fine-tuned model on data it has never seen:

| Test Set | Source | Challenge |
|----------|--------|-----------|
| Real names (no digits) | Manually created | "John Smith", "María García" |
| Different address formats | Manually created | "P.O. Box 123", "123 Main St #4B" |
| Actual LLM responses | Run `naive` profile, collect answers | Real output format |
| i2b2 2014 de-id | Public dataset | Gold standard clinical NER |

### 7.2 LLM Response Evaluation

Generate actual LLM responses using the naive prompt, then run NER on them:

```bash
uv run evaluate --profile naive --limit 100 --no-save
# Collect raw_answer from each response
# Run NER on collected responses
# Compare vs ground truth
```

---

## Deliverables

### Tables for Paper

1. **Main Results Table**: Entity-level F1 per model (3 models × 2 entity types)
2. **Cross-Validation Table**: Mean ± std across 5 folds
3. **Ablation Table**: F1 for each data/training ablation
4. **Baseline Comparison Table**: All systems on same test set
5. **Latency Table**: P50/P95/P99 per model
6. **Error Analysis Table**: FN/FP categories with counts

### Figures for Paper

1. **Training curves**: Loss + F1 per epoch (existing W&B charts)
2. **Confusion matrix**: Entity-level, per model
3. **Error distribution**: Bar chart of error categories
4. **Latency vs text length**: Scatter plot
5. **Ablation impact**: Bar chart of F1 per ablation
6. **Cross-validation box plots**: F1 distribution per model

### Weave Artifacts

All results tracked in Weave for reproducibility:

| Object | Type |
|--------|------|
| `phi-ner-train` | Dataset (versioned: v1, v2) |
| `phi-ner-val` | Dataset |
| `phi-ner-test` | Dataset |
| `phi-ner-model` | Model artifact |
| `ner-eval-{model}` | Evaluation results |
| NER training runs | W&B Runs (loss curves, metrics) |

---

## Execution Order

```
Phase 1 → Update metrics (seqeval)               ~1 hour
Phase 2 → Cross-validation (5-fold × 3 models)   ~2 hours (Colab)
Phase 3 → Ablation study (12+ experiments)        ~4 hours (Colab)
Phase 4 → Error analysis                          ~2 hours
Phase 5 → Baseline comparison                     ~3 hours (Colab)
Phase 6 → Latency profiling                       ~1 hour
Phase 7 → Generalization testing                  ~2 hours
                                         Total:   ~15 hours
```

Most GPU-heavy work (Phases 2, 3, 5) runs on Colab. Error analysis and latency profiling run locally.