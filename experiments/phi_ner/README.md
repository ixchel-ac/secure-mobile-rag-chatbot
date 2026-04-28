# PHI NER Fine-tuning Experiment

Fine-tune BERT-like models to detect patient names and addresses in LLM responses, replacing spaCy in FW-L2.

## Goal

Replace spaCy `en_core_web_sm` (which misses Synthea names and partial addresses) with a fine-tuned model trained on our data.

## Models

| Model | Params | Size | Inference | Description |
|-------|--------|------|-----------|-------------|
| `distilbert-base-uncased` | 66M | ~260MB | ~5ms | Fastest, good baseline |
| `bert-base-uncased` | 110M | ~440MB | ~10ms | Standard BERT |
| `roberta-base` | 125M | ~500MB | ~10ms | Best NER accuracy |

## Entity Types

| Entity | Current (spaCy) | Target (fine-tuned) |
|--------|:---:|:---:|
| NAME | ~60% recall | >95% recall |
| ADDRESS | ~40% recall (fragments only) | >90% recall |

## Steps

### 1. Generate training data

```bash
python experiments/phi_ner/scripts/generate_training_data.py
```

Creates `data/train.json` and `data/val.json` from:
- Synthea chunks (real DEMOGRAPHICS text with PHI)
- Synthetic LLM responses (simulated naive prompt output)
- Negative examples (clean clinical text)

### 2. Train models

```bash
# Single model
python experiments/phi_ner/scripts/train.py --model distilbert

# All 3 models for comparison
python experiments/phi_ner/scripts/train.py --model all
```

Results logged to W&B: `wandb.ai/ricardo-morales-b/mobile-rag-firewall`

### 3. Evaluate and compare

```bash
python experiments/phi_ner/scripts/evaluate_model.py --model all
```

### 4. Integrate best model into FW-L2

Update `backend/app/firewall/fw_l2.py` to use the fine-tuned model instead of spaCy.

## Directory Structure

```
experiments/phi_ner/
├── README.md
├── data/
│   ├── train.json         # Generated training data
│   └── val.json           # Generated validation data
├── models/
│   ├── distilbert/best/   # Fine-tuned DistilBERT
│   ├── bert/best/         # Fine-tuned BERT
│   └── roberta/best/      # Fine-tuned RoBERTa
├── scripts/
│   ├── generate_training_data.py
│   ├── train.py
│   └── evaluate_model.py
└── configs/               # Hyperparameter configs (optional)
```