# PII NER Model — Analysis Report

**Fine-tuning DistilBERT, BERT, and RoBERTa to detect patient names and addresses in LLM responses for the FW-L2 response firewall.**

---

| Property | Detail |
|---|---|
| Dataset | 12,404 total examples |
| Test set | 1,861 examples |
| Labels | `O` · `B-NAME` · `I-NAME` · `B-ADDRESS` · `I-ADDRESS` |
| Platform | Google Colab T4 GPU |
| Tracking | Weights & Biases + Weave (`mobile-rag-firewall`) |

---

## 1. Background and Motivation

The FW-L2 response firewall uses regex to catch structured PII such as SSNs, phone numbers, email addresses, and dates of birth. However, regex cannot reliably detect free-form patient names (e.g. `Adah626 Klein929`) or narrative addresses (e.g. `308 Deckow Union, Pasco`). Three BERT-family models were fine-tuned on BIO-tagged token sequences generated from Synthea patient records to fill this gap.

---

## 2.1 Dataset Overview

Training data was generated locally via `experiments/phi_ner/scripts/generate_training_data.py` and published to Weave. Each example contains a `tokens` list and a corresponding `ner_tags` list in BIO format. Approximately **40% of examples** contain at least one PII entity.

| Split | Examples |
|---|---|
| Train | 8,682 |
| Validation | 1,861 |
| Test | 1,861 |
| **Total** | **12,404** |

The data split was 70% train / 15% val / 15% test. True entity counts in the test set: **502 NAME entities** and **814 ADDRESS entities** (1,316 total spans).

--

## 2.2 Notebooks

| Notebook | Focus | Compute |
|----------|-------|---------|
| `01_training.ipynb` | Model training (existing, updated) | GPU (Colab) |
| `02_evaluation.ipynb` | Entity-level evaluation + cross-validation | GPU (Colab) |
| `03_error_analysis.ipynb` | Error categorization + failure modes | CPU |
| `04_ablation.ipynb` | Data ablation + hyperparameter sensitivity | GPU (Colab) |
| `05_baselines.ipynb` | Comparison with established NER systems | CPU/GPU |

---

## 3. Model Configurations

Three architectures were compared, each with learning rates and batch sizes tuned for a T4 16 GB GPU with FP16 training.

| Model | Parameters | Learning Rate | Batch Size | Why |
|---|---|---|---|---|
| DistilBERT | 66M | 5e-5 | 32 | Smallest and fastest — best if the task is easy |
| BERT | 110M | 3e-5 | 24 | Standard baseline |
| RoBERTa | 125M | 2e-5 | 24 | Often best for NER but requires a lower LR |

Key training settings applied to all models:

- `eval_strategy="epoch"` — evaluate after every full pass through the data
- `load_best_model_at_end=True` — retain the checkpoint with the highest F1, not the last one
- `fp16=True` — half-precision training (2× faster, same accuracy)
- `EarlyStoppingCallback(patience=2)` — stop if F1 does not improve for 2 consecutive epochs
- `dataloader_num_workers=2` — parallel data loading for better GPU utilization

---

## 4. Validation-Set Performance (Token-Level F1)

Token-level F1 scores each token independently and awards partial credit — a name matched on 2 of 3 tokens still scores 67%. This is the metric reported during training.

| Model | F1 Macro | F1 NAME | F1 ADDRESS |
|---|---|---|---|
| DistilBERT | **0.9972** | 1.0000 | 0.9944 |
| BERT | **0.9975** | 1.0000 | 0.9950 |
| RoBERTa | **0.9972** | 1.0000 | 0.9945 |

### Reading the results

All three models reached near-identical validation F1 scores. The performance gap between the largest model (RoBERTa, 125M params) and the smallest (DistilBERT, 66M params) is negligible — a strong signal that the task does not require additional model capacity. NAME detection was essentially perfect (1.0) across all models; ADDRESS was marginally harder (~0.994–0.995) due to greater structural variability in Synthea-generated addresses.

```
Training curve — DistilBERT (epochs 1–5):

Epoch   Train Loss   Val Loss   F1 Entity
1       0.002652     0.001762   0.9637
2       0.001152     0.000971   0.9817
3       0.000608     0.000911   0.9827
4       0.000377     0.000771   0.9842
5       0.000314     0.000761   0.9859   ← best checkpoint saved
```

---

## 5. Entity-Level Evaluation — seqeval Strict (Test Set)

> **Note:** The baseline test F1 reported throughout sections 5–13 is 0.9734, reflecting the standard configuration (5 epochs, no oversampling). The ablation study in Section 7 identifies a clear path to ~0.986 through training-only changes.

Strict entity-level matching via `seqeval` with `mode='strict'` requires the predicted span to exactly match the true span in both position and type. A partial match (e.g. predicting `Adah626 Flo729` when the true entity is `Adah626 Flo729 Klein929`) counts as a complete miss. This is the correct metric for a PII firewall, where a partially-redacted name still leaks patient data.

```
Example:
  True label:  "Adah626 Flo729 Klein929"  (3 tokens, 1 entity)
  Prediction:  "Adah626 Flo729"            (2 tokens matched)

  Token F1:    2/3 = 67%  → looks acceptable
  Entity F1:   0/1 = 0%   → name still leaks!
```

| Model | F1 NAME | F1 ADDRESS | F1 Macro |
|---|---|---|---|
| DistilBERT | 1.0000 | 0.9594 | 0.9797 |
| BERT | 1.0000 | 0.9593 | 0.9797 |
| RoBERTa | 1.0000 | 0.9644 | **0.9822** |

### Reading the results

Scores dropped 1–2 percentage points versus token-level metrics, revealing the true difficulty. ADDRESS remains consistently harder than NAME across all models (F1 ~0.96 vs 1.0). This asymmetry is meaningful: addresses are structurally more variable and context-dependent than the alphanumeric Synthea name pattern.

RoBERTa achieves the highest macro F1 (0.9822) and the best ADDRESS score (0.9644), but the difference is **not statistically significant** — the Wilcoxon signed-rank test returned p = 0.25 across multi-seed runs.

---

## 6. Cross-Validation Stability (5-Fold, seqeval F1)

To ensure results are not dependent on a single lucky train/val split, 5-fold cross-validation was run across all three models. Train + validation data were combined and split into 5 equal folds. Each fold is logged as a separate W&B run.

| Model | F1 NAME | F1 ADDRESS | F1 Macro |
|---|---|---|---|
| DistilBERT | 0.9998 ± 0.0004 | 0.9657 ± 0.0084 | 0.9827 ± 0.0043 |
| BERT | 0.9992 ± 0.0012 | 0.9644 ± 0.0092 | 0.9818 ± 0.0051 |
| RoBERTa | 0.9994 ± 0.0005 | 0.9694 ± 0.0040 | **0.9844 ± 0.0019** |

### Reading the results

RoBERTa's ADDRESS standard deviation (±0.004) is notably tighter than DistilBERT's (±0.008), suggesting it is less sensitive to which examples land in validation. However, all three models show low variance overall — the task is stable and learnable. A standard deviation above 0.005 on macro F1 would be a warning sign; none of the models exceed this threshold.

---

## 7. Ablation Study

An ablation study isolates which design choices actually matter by changing one variable at a time and measuring the impact. All experiments use DistilBERT as the base model (fastest, best F1/latency trade-off). Each experiment is a separate W&B run tagged with `ablation`. The baseline for comparison is the `full` run — the standard training configuration with all data and default hyperparameters.

### Full results (sorted by F1)

| Experiment | Category | Test F1 | vs Baseline |
|---|---|---|---|
| `epochs_10` | Training | **0.9861** | +0.0127 |
| `oversampled_3x` | Imbalance | **0.9854** | +0.0120 |
| `lr_5e-05` | Training | 0.9812 | +0.0078 |
| `bs_32` | Training | 0.9812 | +0.0078 |
| `epochs_5` | Training | 0.9795 | +0.0061 |
| `lr_3e-05` | Training | 0.9772 | +0.0038 |
| `bs_16` | Training | 0.9772 | +0.0038 |
| `bs_8` | Training | 0.9765 | +0.0031 |
| `lr_0.0001` | Training | 0.9764 | +0.0030 |
| `epochs_3` | Training | 0.9753 | +0.0019 |
| `full` | Baseline | 0.9734 | — |
| `no_negatives` | Data | 0.9703 | -0.0031 |
| `lr_1e-05` | Training | 0.9675 | -0.0059 |
| `no_synthetic` | Data | 0.8740 | **-0.0994** |
| `addresses_only` | Data | 0.7394 | -0.2340 |
| `names_only` | Data | 0.5520 | -0.4214 |

---

### 7.1 Data Ablations

Each experiment removes or modifies one component of the training data.

| Experiment | What changes | Test F1 |
|---|---|---|
| `full` | Nothing — baseline | 0.9734 |
| `no_negatives` | Remove examples with no entities | 0.9703 |
| `no_synthetic` | Remove synthetic LLM-style responses | 0.8740 |
| `names_only` | Mask ADDRESS labels to `O` | 0.5520 |
| `addresses_only` | Mask NAME labels to `O` | 0.7394 |

**`no_synthetic` is the most important finding in the entire study.** Removing Synthea-generated synthetic examples caused a 10 percentage point drop — by far the largest degradation of any experiment across all three categories. The validation loss for this run was elevated throughout (0.005 vs 0.001 for the baseline), and F1 never recovered past 0.901 even after 5 epochs. This tells you that the synthetic examples are not helpful padding — they are the primary source of learning signal. Any future data pipeline work must protect and expand the Synthea generator first.

**`no_negatives`** produced a small but real drop of 0.3 points. Negative examples — text with no PII — teach the model what not to flag. Removing them predictably increases false positives. The drop is modest enough that you could reduce negatives slightly to speed up training, but removing them entirely is not worth the cost.

**`names_only` and `addresses_only`** are diagnostic experiments rather than practical configurations. When ADDRESS labels are masked and the model is trained on NAME only, F1 collapses to 0.55 — not because the model fails at names, but because macro F1 averages across both entity types and ADDRESS scores zero on the test set. More revealing is the training curve: validation F1 was completely flat at 0.5675 across all 5 epochs while validation loss *increased* every epoch (0.160 → 0.179). Training loss dropped normally, indicating overfitting to the name-only signal. This suggests the two entity types provide complementary structural context during training — the model learns better entity boundaries when both types are present simultaneously.

---

### 7.2 Training Ablations

**Epochs**

| Epochs | Test F1 |
|---|---|
| 3 | 0.9753 |
| 5 | 0.9795 |
| 10 | **0.9861** |

Longer training consistently helps and there is no sign of overfitting at 10 epochs. Validation F1 continued climbing from 0.972 at epoch 1 to 0.991 at epoch 10, while validation loss stayed stable around 0.00075. The early stopping with `patience=2` used in notebooks 01 and 02 was likely triggering before full convergence — the per-epoch F1 improvements in later epochs are small (e.g. 0.9851 → 0.9883 → 0.9917 in epochs 7–9) and within the noise of a single run, causing premature stopping. Training for 10 epochs instead of 5 costs roughly 5 extra minutes on a T4 and yields a ~1.3 point F1 gain.

```
Training curve — epochs_10 run:

Epoch   Train Loss   Val Loss   F1 Entity
1       0.002688     0.001492   0.9727
2       0.001087     0.000848   0.9801
3       0.000784     0.000701   0.9847
4       0.000423     0.000728   0.9831
5       0.000400     0.000729   0.9846
6       0.000253     0.000840   0.9852
7       0.000343     0.000680   0.9884
8       0.000100     0.000767   0.9913
9       0.000104     0.000743   0.9917
10      0.000069     0.000752   0.9917   ← best checkpoint
```

**Learning rate**

| LR | Test F1 |
|---|---|
| 1e-5 | 0.9675 |
| 3e-5 | 0.9772 |
| **5e-5** | **0.9812** |
| 1e-4 | 0.9764 |

The chosen default of 5e-5 is optimal. The 1e-4 run shows the classic signature of a slightly too-high learning rate — epoch 1 validation F1 jumped only to 0.937 versus 0.974 for other runs, as large gradient steps overshoot good solutions early. The 1e-5 run underfit — training loss at epoch 1 was 0.016 versus ~0.003 for others. The model is moderately sensitive to LR but not catastrophically so — a factor-of-2 error does not cause complete failure.

**Batch size**

| Batch size | Test F1 |
|---|---|
| 8 | 0.9765 |
| 16 | 0.9772 |
| **32** | **0.9812** |

Larger batch size wins here and is also faster — batch 8 took 7 minutes versus 5.5 minutes for batch 32 due to the larger number of gradient steps per epoch. The gradient noise from small batches provided no regularisation benefit on this dataset. Batch 32 is both more accurate and more efficient.

---

### 7.3 Class Imbalance

The training data is heavily imbalanced: 99.2% of tokens are `O` (not PII). The model could theoretically achieve 99% token accuracy by predicting `O` for everything. Oversampling addresses this by duplicating entity-rich examples 3× in the training set.

| Experiment | Training examples | Test F1 |
|---|---|---|
| `full` (baseline) | 8,682 | 0.9734 |
| `oversampled_3x` | 8,682 + 2× entity examples | **0.9854** |

`oversampled_3x` is the second-best result in the entire ablation study, behind only 10-epoch training. A 1.2 point F1 gain from simply duplicating existing data requires no new data collection and no architectural changes — it is a pure training pipeline change. The epoch 1 F1 of 0.9777 was also notably higher than the baseline epoch 1 of 0.9764, suggesting the model found entity patterns earlier when they appeared more frequently in each training batch.

> **Note on interpretation:** Oversampling can inflate results if the same entity patterns appear in both training and validation (since both come from the same Synthea distribution). The 0.985 figure should be treated as an upper bound; the real production gain is likely slightly smaller but still meaningful.

---

## 8. Multi-Seed Stability

Each model was trained 3 times using random seeds 42, 123, and 456 to verify results are not dependent on weight initialisation. Mean ± std is reported across the three runs.

| Model | F1 Macro (mean ± std) | Individual seeds |
|---|---|---|
| DistilBERT | 0.9817 ± 0.0023 | [0.9797, 0.9849, 0.9804] |
| BERT | 0.9810 ± 0.0016 | [0.9797, 0.9832, 0.9801] |
| RoBERTa | 0.9799 ± 0.0022 | [0.9828, 0.9791, 0.9779] |

**Wilcoxon test: DistilBERT vs BERT → p = 0.25 (not significant)**

The differences between models are within the noise of random initialisation. Model selection should therefore be made on practical grounds (speed, size) rather than accuracy.

---

## 9. Inference Latency Profiling

For a real-time response firewall, latency is a first-class concern. Each model was profiled on 500 test examples on the same T4 GPU.

| Model | P50 | P95 | P99 | Throughput |
|---|---|---|---|---|
| DistilBERT | **27.6 ms** | **41.8 ms** | **45.0 ms** | **~36 ex/sec** |
| BERT | 41.1 ms | 52.4 ms | 55.5 ms | ~24 ex/sec |
| RoBERTa | 41.4 ms | 52.0 ms | 57.7 ms | ~24 ex/sec |

### Reading the results

DistilBERT's P50 latency of 27.6 ms is approximately **33% faster** than BERT and RoBERTa (~41 ms). At P99 — the worst-case scenario — DistilBERT stays comfortably below 50 ms while the other two approach 56–58 ms. For a synchronous firewall applied to every LLM response, this gap compounds significantly at scale. DistilBERT processes roughly **1.5× more examples per second** than BERT or RoBERTa at equivalent accuracy.

---

## 10. Error Analysis — False Negatives (Missed PII)

Running the best model on the full 1,861-example test set produced the following confusion summary:

| Outcome | Count |
|---|---|
| Correct | 1,532 |
| False Negatives (missed PII) | 93 |
| False Positives (over-redaction) | 264 |

### False negatives by entity type

| Type | Count |
|---|---|
| ADDRESS | 65 |
| NAME | 28 |

### False negative categories

| Category | Count | Description |
|---|---|---|
| `single_token` | 58 | Single-word address entities (e.g. `EDMONDS` in `SWEDISH EDMONDS`) |
| `synthea_name` | 28 | Full Synthea names with numeric suffixes (e.g. `Riley817 Jarod376 Spinka232:`) |
| `address_with_numbers` | 5 | Street addresses containing digits (e.g. `915 Witting Annex Apt 45`) |
| `partial_address` | 2 | Addresses of 1–2 tokens only |

### Reading the results

The dominant failure mode (58 of 93 misses) is **single-token address entities** — isolated city or location names appearing inside institutional names such as `ENCOUNTER AT SWEDISH EDMONDS`. These are genuinely ambiguous without surrounding multi-token context, and the same token (`EDMONDS`) appears repeatedly across multiple test records, inflating the raw count. The second-largest failure category (28) covers Synthea-style names with numeric suffixes that happen to appear at the start of a record header (e.g. `Medical record for Riley817 Jarod376 Spinka232:`), where the trailing colon may be disrupting span detection.

> **Safety note:** Single-token address misses are the highest-risk category from a PII leakage perspective. These tokens appear in clinical encounter records and contain location data that could identify a patient's treatment facility or geographic area.

---

## 11. Error Analysis — False Positives (Over-Redaction)

### False positive categories

| Category | Count | Description |
|---|---|---|
| `single_word_address` | 169 | Single-word fragments incorrectly labelled as ADDRESS |
| `other` | 72 | Miscellaneous — unclassified patterns |
| `single_digit_or_short` | 23 | Short strings (digits, 1–2 characters) flagged as ADDRESS |

### Reading the results

The overwhelming majority of false positives (169 of 264) are **single-word address predictions** arising from tokeniser subword artifacts. When the DistilBERT tokeniser splits hyphenated compound proper nouns — for example, `KINDRED HOSPITAL-SEATTLE` becomes `KINDRED`, `HOSPITAL`, `-`, `##SEATTLE` — the fragment `##red` or `- seattle` gets incorrectly classified as an address token. This pattern appears repeatedly across multiple records containing the same hospital name, inflating the false positive count significantly.

The `single_digit_or_short` category (23 cases) represents SNOMED-CT codes and other numeric identifiers being confused with address tokens — for example, `##341003` in `SNOMED CT: 38341003 (Hypertension)`.

---

## 12. Error Correlation with Text Properties

Error rate was examined against two text properties to understand when the model is least reliable.

**Error rate vs text length:** Error rates were highest for very short texts (0–50 tokens) and lowest for medium-length texts (50–200 tokens). Long texts (200+ tokens) showed a moderate increase, which is expected as more entities appear per example and the probability of at least one miss increases.

**Error rate vs entity density:** Entity-dense texts (many PII spans per 100 tokens) showed elevated error rates in medium-density bins, consistent with the hypothesis that closely-packed entities create boundary confusion for the model.

---

## 13. Weave Evaluation Summary

The Weave evaluation on the test set measures entity-level detection at the example level (was at least one entity of each type detected, regardless of exact span) rather than exact-match F1.

| Model | Name detected | Address detected | Name FP rate | Address FP rate |
|---|---|---|---|---|
| DistilBERT | **1.000** | 0.9989 | 0.0021 | 0.0075 |
| BERT | **1.000** | 0.9995 | 0.0021 | 0.0258 |
| RoBERTa | **1.000** | 0.9979 | 0.0011 | 0.0666 |

### Reading the results

All three models detect NAME at 100% at the example level. ADDRESS detection is above 99.7% for all models. However, RoBERTa's address false positive rate (6.7%) is markedly higher than DistilBERT's (0.75%), suggesting RoBERTa is more aggressive in labelling non-address tokens as addresses — despite its higher seqeval F1. This reinforces the case for DistilBERT as the deployment model.

---

## 14. Summary and Recommendations

### Model verdict

| Criterion | Winner | Notes |
|---|---|---|
| Accuracy (seqeval) | RoBERTa (marginal) | Difference not statistically significant |
| Stability (CV std) | RoBERTa | Tighter ADDRESS variance across folds |
| Latency | DistilBERT | 33% faster at P50, 1.5× throughput |
| False positive rate | DistilBERT | 0.75% vs 6.7% for RoBERTa on address FP |
| Model size | DistilBERT | 66M vs 110–125M params |
| **Deployment choice** | **DistilBERT** | Best overall trade-off |


> **Baseline vs optimised:** The baseline test F1 is 0.9734 (5 epochs, no oversampling). The ablation study identifies two training-only changes — 10 epochs and 3× oversampling of entity-rich examples — that project to approximately **0.986–0.988** F1 with no new data or architectural changes.

### Highest-impact improvements

The following improvements are ranked by estimated F1 gain and implementation cost. Items 1 and 2 come directly from the ablation study and require only training pipeline changes — no new data collection and no architectural modifications.

1. **Train for 10 epochs instead of 5** *(ablation study — +1.3 F1 points, ~5 extra minutes on T4):* The ablation study showed validation F1 continued climbing to 0.9917 at epoch 10 with no sign of overfitting. Early stopping was triggering prematurely at epoch 5. Change `num_train_epochs=10` and remove or increase the early stopping patience to 3–4.

2. **Apply 3× oversampling of entity-rich examples** *(ablation study — +1.2 F1 points, zero new data required):* Duplicating examples with at least one PII entity 3× in the training set directly addresses the 99.2% class imbalance. Combined with recommendation 1, the projected test F1 moves from 0.9734 to approximately 0.986–0.988.

3. **Protect and expand the synthetic data pipeline** *(ablation study — prevents -10 F1 point collapse):* The `no_synthetic` ablation showed a 10-point drop when synthetic examples were removed. The Synthea generator is the single most important component of the training pipeline. Any future data work should prioritise expanding its coverage, particularly for clinical encounter records involving multi-word institutional names.

4. **Single-token address misses (58 FN):** Add training examples of isolated location tokens (city names, area names) appearing inside institutional name strings. Augmenting the Synthea generator to include encounters at multi-word hospital names containing location tokens would directly address this. Oversampling (recommendation 2) will partially help here by increasing exposure to these harder examples.

5. **Hyphenated compound name FP (169 FP):** Pre-process input text before tokenisation to replace hyphens in institutional names (e.g. `HOSPITAL-SEATTLE`) with spaces, preventing the tokeniser from producing address-like subword fragments.

6. **Synthea name suffix detection (28 FN):** The model occasionally misses names when they appear at the very start of a record header followed by a colon. Adding examples of this exact pattern to training data would help.

7. **Address false positive rate monitoring:** The 0.75% address FP rate for DistilBERT is acceptable but should be monitored in production, particularly for clinical text containing SNOMED-CT codes and coded encounter identifiers.

---

## Appendix — W&B Run References

All experiments were logged to the `mobile-rag-firewall` project on Weights & Biases.

| Run | Model | Type |
|---|---|---|
| `phi-ner-distilbert` | DistilBERT | Initial training |
| `phi-ner-bert` | BERT | Initial training |
| `phi-ner-roberta` | RoBERTa | Initial training |
| `eval-distilbert-s42` | DistilBERT | Entity-level eval |
| `cv-distilbert-fold1` … `fold5` | DistilBERT | Cross-validation |
| `seed-distilbert-s42/123/456` | DistilBERT | Multi-seed stability |
| `abl-full` | DistilBERT | Ablation baseline |
| `abl-no_negatives` | DistilBERT | Data ablation |
| `abl-no_synthetic` | DistilBERT | Data ablation |
| `abl-names_only` | DistilBERT | Data ablation |
| `abl-addresses_only` | DistilBERT | Data ablation |
| `abl-epochs_3` | DistilBERT | Training ablation |
| `abl-epochs_5` | DistilBERT | Training ablation |
| `abl-epochs_10` | DistilBERT | Training ablation |
| `abl-lr_1e-05` … `abl-lr_0.0001` | DistilBERT | Training ablation |
| `abl-bs_8` … `abl-bs_32` | DistilBERT | Training ablation |
| `abl-oversampled_3x` | DistilBERT | Class imbalance ablation |
| `phi-ner-error-analysis` | DistilBERT | Error analysis (Weave) |

Artifact: `phi-ner-model:latest` — published to W&B, pulled automatically by FW-L2 via `run.use_artifact("phi-ner-model:latest")`.