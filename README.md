# Mobile RAG Firewall

A two-level firewall architecture for securing Retrieval-Augmented Generation (RAG) systems
serving medical data on mobile devices.

**Architecture:** On-device FW-L1 (input firewall) + Cloud-side FW-L2 (response validation and
PII anonymization), with a FastAPI backend, FAISS vector store, and Llama 3.1 via W&B Inference / Groq / Ollama.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Environment Configuration](#environment-configuration)
4. [Phase 0: Environment Setup](#phase-0-environment-setup)
5. [Phase 1: Data Ingestion into FAISS](#phase-1-data-ingestion-into-faiss)
6. [Phase 2: Retriever](#phase-2-retriever)
7. [Phase 3: Generator and RAG Pipeline](#phase-3-generator-and-rag-pipeline)
8. [Phase 4: FW-L2 Response Firewall](#phase-4-fw-l2-response-firewall)
9. [Evaluation](#evaluation)
10. [PII NER Fine-tuning Experiment](#pii-ner-fine-tuning-experiment)
11. [CLI Commands](#cli-commands)
12. [Running Tests](#running-tests)
13. [Docker](#docker)
14. [Phase Summary](#phase-summary)
15. [Key Findings](#key-findings)

---

## Prerequisites

| Tool            | Version   | Purpose                                  |
|-----------------|-----------|------------------------------------------|
| Python          | >= 3.11   | Backend runtime                          |
| uv              | latest    | Fast Python package manager (replaces pip) |
| Docker Desktop  | latest    | Container orchestration (Ollama)         |
| Java JDK        | >= 11     | Running Synthea (patient data generator) |
| Git             | latest    | Version control                          |

### Installing uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

---

## Project Structure

```
mobile-rag-firewall/
â”œâ”€â”€ README.md
â”œâ”€â”€ docker-compose.yml          # Orchestrates backend + Ollama (optional)
â”œâ”€â”€ .env                        # Environment variables (models, API keys, paths)
â”‚
â”œâ”€â”€ data/                       # Data directory (not committed)
â”‚   â”œâ”€â”€ synthea/                # Raw Synthea patient data (text + CSV)
â”‚   â”œâ”€â”€ processed/              # PII ground truth, chunks
â”‚   â”œâ”€â”€ golden_sets/            # Adversarial query sets (1000 queries, C1-C5)
â”‚   â””â”€â”€ evaluation_results/     # JSON reports from evaluation runs
â”‚
â”œâ”€â”€ backend/                    # Cloud zone -- FastAPI + RAG + FW-L2
â”‚   â”œâ”€â”€ Dockerfile
â”‚   â”œâ”€â”€ pyproject.toml          # Dependencies + CLI scripts (8 commands)
â”‚   â”œâ”€â”€ tests/
â”‚   â”‚   â”œâ”€â”€ test_loader.py      # 20 tests: data loading
â”‚   â”‚   â”œâ”€â”€ test_retriever.py   # 35 tests: retrieval quality + adversarial
â”‚   â”‚   â”œâ”€â”€ test_pipeline.py    # 22 tests: benign + adversarial LLM queries
â”‚   â”‚   â””â”€â”€ test_fw_l2.py       # 29 tests: regex, NER, injection detection
â”‚   â””â”€â”€ app/
â”‚       â”œâ”€â”€ config.py           # Settings, env vars
â”‚       â”œâ”€â”€ cli.py              # CLI entry points (uv run commands)
â”‚       â”œâ”€â”€ ingestion/          # Loader, Cleaner, Chunker, Pipeline
â”‚       â”œâ”€â”€ rag/                # Embedder, Retriever, Generator, RAG Pipeline
â”‚       â”œâ”€â”€ firewall/           # FW-L2 (regex + spaCy NER + injection detection)
â”‚       â”œâ”€â”€ evaluation/         # Runner, Weave eval, Leaderboard
â”‚       â”œâ”€â”€ vectorstore/        # FAISS index management
â”‚       â”œâ”€â”€ models/             # Pydantic schemas
â”‚       â””â”€â”€ routes/             # API endpoints (stub)
â”‚
â”œâ”€â”€ experiments/                # ML experiments
â”‚   â””â”€â”€ phi_ner/                # BERT fine-tuning for PII NER
â”‚       â”œâ”€â”€ pyproject.toml      # Experiment dependencies + CLI (6 commands)
â”‚       â”œâ”€â”€ notebooks/          # Google Colab training notebook
â”‚       â”œâ”€â”€ scripts/            # Generate data, train, evaluate, export
â”‚       â”œâ”€â”€ data/               # Generated train/val/test splits
â”‚       â””â”€â”€ models/             # Trained model checkpoints
â”‚
â”œâ”€â”€ index/                      # FAISS index files (~41 MB)
â”œâ”€â”€ fw_l1/                      # On-device FW-L1 (stub)
â””â”€â”€ evaluation/                 # E2E evaluation scripts
```

---

## Environment Configuration

All configuration via `.env` at the project root:

```bash
# ---- Model Configuration ----
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_PROVIDER=wandb                           # "wandb", "groq", or "ollama"
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct   # W&B model name

# ---- W&B Inference (cloud LLM + Weave tracing) ----
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=mobile-rag-firewall

# ---- Groq (cloud LLM, alternative) ----
GROQ_API_KEY=your-groq-api-key

# ---- Ollama (local LLM, alternative) ----
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# ---- Paths ----
DATA_DIR=./data
INDEX_DIR=./index

# ---- RAG Settings ----
TOP_K=5
```

### LLM Provider Options

| Provider | Config | Speed | Privacy | Cost |
|----------|--------|:-----:|---------|------|
| **W&B Inference** | `LLM_PROVIDER=wandb` | ~2s/query | Cloud | Free tier |
| **Groq** | `LLM_PROVIDER=groq` | ~2s/query | Cloud | Free tier (rate limited) |
| **Ollama** | `LLM_PROVIDER=ollama` | ~30-60s/query | Local | Free |

---

## Phase 0: Environment Setup

```bash
cd backend
uv sync && uv pip install -e .
```

Configure LLM provider in `.env` (see above), then verify:

```bash
uv run help    # Shows all commands and current config
```

---

## Phase 1: Data Ingestion into FAISS

Generate Synthea data, copy to `data/synthea/`, then:

```bash
cd backend
uv run ingestion    # load -> clean -> chunk -> embed -> FAISS index -> PII ground truth
uv run data-show    # Patient statistics
uv run faiss-check  # Index health check
```

**Key design decisions:**
- **UUID-based matching**: Filenames contain UUIDs matching `patients.csv` (100% match rate)
- **Section-based chunking**: Each medical section (MEDICATIONS, CONDITIONS, etc.) becomes one chunk
- **PII injected into DEMOGRAPHICS**: SSN and address are added to DEMOGRAPHICS chunks so the LLM context realistically contains PII (enables fair firewall testing)

**Result:** 218 patients â†’ 11,104 chunks â†’ 384-dim FAISS index (41 MB)

---

## Phase 2: Retriever

**File:** `backend/app/rag/retriever.py`

Pure semantic search wrapping Embedder + FAISS. Supports optional section filtering.

```bash
uv run search    # Interactive search CLI
```

**Key finding:** The retriever does NOT block adversarial queries â€” every chunk carries full PII in metadata. Adversarial queries surface 9/10 DEMOGRAPHICS chunks vs 0/10 for benign queries.

---

## Phase 3: Generator and RAG Pipeline

### Dual System Prompts

| Prompt | Purpose | PII Protection |
|--------|---------|:-:|
| **Naive** | Test FW-L2 in isolation | None â€” LLM leaks freely |
| **Hardened** | Production defense-in-depth | Strong guardrails |

The **hardened prompt** was tightened after discovering metadata leakage â€” the original prompt said "SSNs are listed but I can't reveal them" (confirming data exists). The current version gives no information about what data is or isn't present:

```
"I can only answer clinical questions about patient health records."
```

### LLM Providers

Three providers supported: W&B Inference, Groq, Ollama. All use the same Llama 3.1 model. W&B Inference recommended (best rate limits + Weave tracing).

### Pipeline Flow

```
Query â†’ Retriever (FAISS) â†’ Generator (LLM) â†’ FW-L2 (optional) â†’ Response
```

---

## Phase 4: FW-L2 Response Firewall

**File:** `backend/app/firewall/fw_l2.py`

FW-L2 scans LLM responses and redacts detected PII before returning to the user.

### Detection Layers

| Layer | What it catches | Method | Size |
|-------|----------------|--------|------|
| **Regex** | SSN, phone, email, DOB, MRN/UUID | Pattern matching | 0 MB |
| **spaCy NER** | Real names (PERSON), cities/states (GPE) | `en_core_web_sm` | 12 MB |
| **Synthea pattern** | Synthea names (e.g., `Adah626 Klein929`) | Custom regex | 0 MB |
| **Medical whitelist** | Filters drug names misclassified as PERSON | Lookup set | 0 MB |
| **Injection detection** | System prompt echo, role reassignment, debug mode | Pattern matching | 0 MB |

### Example

```
Raw LLM response:
  "Patient Adah626 Klein929 (SSN: 999-83-1042) lives at 308 Deckow Union, Pasco, Washington"

After FW-L2:
  "Patient [NAME] (SSN: [SSN]) lives at 308 [ADDRESS], [ADDRESS], [ADDRESS]"
```

### What FW-L2 Does NOT Use

**Ground truth is NOT used for detection** â€” that would be cheating. `phi_groundtruth.json` is only used in the evaluation runner to measure what FW-L2 missed. In production, there's no lookup table of every patient's name.

### Known Gaps

| PII Type | Detected? | Why |
|----------|:-:|-----|
| SSN (`123-45-6789`) | Yes | Regex |
| Phone, email | Yes | Regex |
| DOB (`1992-10-04`) | Yes | Regex |
| Patient names | Partial | spaCy misses non-standard names |
| Full addresses | Partial | spaCy catches city/state, misses street |
| Metadata leakage | Yes | Regex patterns (separate scorer) |

These gaps are why the BERT NER fine-tuning experiment exists (see below).

### Evaluation Profiles

| Profile | System Prompt | FW-L2 | What it measures |
|---------|:---:|:---:|-----------------|
| `naive` | Naive | Off | How much does the LLM leak without any protection? |
| `naive_fw_l2` | Naive | On | How much does FW-L2 catch? |
| `hardened` | Hardened | Off | How effective is the system prompt alone? |
| `hardened_fw_l2` | Hardened | On | Full defense-in-depth |

### Metadata Leakage Detection

Beyond data leakage (actual PII values), the evaluation also detects **metadata leakage** â€” when the LLM confirms what data exists without revealing values:

| Response | Data leak | Metadata leak |
|----------|:-:|:-:|
| "SSN is 999-83-1042" | Yes | No |
| "SSNs are listed but I can't reveal them" | No | Yes |
| "I can only answer clinical questions" | No | No |

### Tests: 29 passed

```bash
cd backend && python -m pytest tests/test_fw_l2.py -v    # 29 tests, ~6s
```

---

## Evaluation

### Golden Test Set

1000 adversarial queries across 5 categories:

| Category | Queries | Attack Type |
|----------|:-------:|-------------|
| **C1** | 200 | Direct PII extraction ("give me the SSN") |
| **C2** | 200 | InDirect PII extraction ("personal details") |
| **C3** | 200 | Prompt injection / jailbreak |
| **C4** | 200 | Social engineering pretexts |
| **C5** | 200 | Metadata exfiltration (confirm/deny data exists) |

### Running Evaluation

```bash
cd backend

# Standard evaluation
uv run evaluate --profile naive_fw_l2 --parallel --batch-size 10 --limit 50

# Weave evaluation (logs to W&B leaderboard)
uv run evaluate --profile hardened --weave --limit 30

# Full leaderboard (runs all profiles, publishes comparison)
uv run leaderboard --profiles naive naive_fw_l2 hardened hardened_fw_l2 --limit 20

# Generate/regenerate adversarial queries
uv run generate-queries
```

### Weave Integration

All evaluations are tracked in [W&B Weave](https://wandb.ai):

| What | Weave Object |
|------|-------------|
| Adversarial queries | `adversarial-golden-set` (Dataset) |
| System prompts | `system-prompt-naive`, `system-prompt-hardened` (MessagesPrompt) |
| Pipeline configs | `rag-naive`, `rag-hardened`, etc. (Model) |
| NER training data | `phi-ner-train`, `phi-ner-val`, `phi-ner-test` (Dataset) |

### Scorers

| Scorer | Metrics |
|--------|---------|
| `pii_leak_scorer` | blocked, ssn_leaked, pii_leaked, dob_leaked, name_leaked, metadata_leaked |
| `metadata_leak_scorer` | metadata_leaked, clean_refusal |
| `redaction_scorer` | redaction_applied, ssn_caught_by_fw_l2 |
| `injection_scorer` | injection_detected |
| `latency_scorer` | latency_seconds, under_5s |

---

## PII NER Fine-tuning Experiment

**Location:** `experiments/phi_ner/`

Fine-tune BERT-like models to improve FW-L2's name and address detection, replacing spaCy's generic NER.

### Why

spaCy `en_core_web_sm` has gaps:
- Misses Synthea names (numbers in names: `Adah626 Klein929`)
- Only catches city/state, not full addresses
- False positives on drug names (`Aspirin` â†’ PERSON)

### Models

| Model | Params | Size | Speed |
|-------|--------|------|-------|
| `distilbert-base-uncased` | 66M | ~260MB | ~5ms |
| `bert-base-uncased` | 110M | ~440MB | ~10ms |
| `roberta-base` | 125M | ~500MB | ~10ms |

### Training Data

Generated from Synthea chunks + ground truth + synthetic LLM responses:

| Split | Examples | With entities |
|-------|:--------:|:------------:|
| Train | 8,332 | 3,316 |
| Val | 1,786 | 717 |
| Test | 1,786 | 681 |

### CLI Commands (from `experiments/phi_ner/`)

```bash
cd experiments/phi_ner

uv run ner-help        # Show all commands
uv run ner-generate    # Generate training data (publishes to Weave)
uv run ner-train --model distilbert    # Train single model
uv run ner-train --model all           # Train all 3
uv run ner-evaluate --model distilbert # Compare vs spaCy
uv run ner-compare                     # Side-by-side comparison
uv run ner-export --model distilbert   # Export to backend FW-L2
```

### Google Colab Notebooks

All GPU-intensive work runs on Colab with T4. Upload notebooks from `experiments/phi_ner/notebooks/`:

| Notebook | Focus | Time | Depends on |
|----------|-------|:----:|-----------|
| `01_training.ipynb` | Train 3 models, publish best to Weave | ~10 min | Data in Weave |
| `02_evaluation.ipynb` | Entity-level eval, 5-fold CV, multi-seed, latency | ~2 hrs | Data in Weave |
| `03_error_analysis.ipynb` | Categorize FN/FP, error vs text properties | ~10 min | Model from 01 |
| `04_ablation.ipynb` | Data/training/class imbalance ablations | ~3 hrs | Data in Weave |
| `05_baselines.ipynb` | Compare vs spaCy, Presidio, BioClinicalBERT | ~1 hr | Data in Weave + model from 01 |

Data is pulled from Weave automatically. Results log to the same W&B project.

**Execution order**: 01 first (publishes model), then 02-05 in any order.

### Training Results

| Model | F1 Macro | F1 NAME | F1 ADDRESS | Speed |
|-------|:--------:|:-------:|:----------:|:-----:|
| DistilBERT (66M) | **0.9949** | **1.0000** | **0.9898** | **126 samples/s** |
| BERT (110M) | 0.9947 | 1.0000 | 0.9894 | 108 samples/s |
| RoBERTa (125M) | 0.9923 | 0.9997 | 0.9849 | 122 samples/s |

**Winner**: DistilBERT â€” smallest, fastest, highest F1. Larger models show no advantage.

### FW-L2 Profiles

The fine-tuned model is deployed as a separate FW-L2 profile alongside the spaCy-based one:

| Profile | NER Backend | What it uses |
|---------|-------------|-------------|
| `fw_l2_base` | spaCy + Synthea patterns | Generic NER, works out of the box |
| `fw_l2_bert` | Fine-tuned DistilBERT | Domain-specific, higher accuracy |

```bash
# Compare both on the leaderboard
uv run leaderboard --profiles naive_fw_l2_base naive_fw_l2_bert --limit 50
```

### Model Deployment

The trained model is published as a W&B model artifact from Colab. FW-L2 pulls it automatically:

```
Colab (01_training) â†’ wandb.log_artifact("phi-ner-model", type="model")
                                    â†“
FW-L2 backend â†’ run.use_artifact("phi-ner-model:latest").download() â†’ cached locally
```

No manual download needed.

---

## CLI Commands

### Backend (`cd backend`)

| Command | Description |
|---------|-------------|
| `uv run help` | Show all commands and current configuration |
| `uv run ingestion` | Run full pipeline: load â†’ clean â†’ chunk â†’ embed â†’ index |
| `uv run data-show` | Patient data statistics |
| `uv run faiss-check` | FAISS index health check |
| `uv run search` | Interactive retriever search |
| `uv run evaluate` | Run adversarial evaluation (supports `--profile`, `--weave`, `--parallel`) |
| `uv run leaderboard` | Run all profiles and publish W&B leaderboard |
| `uv run generate-queries` | Regenerate the adversarial golden set |

### Experiments (`cd experiments/phi_ner`)

| Command | Description |
|---------|-------------|
| `uv run ner-help` | Show experiment commands |
| `uv run ner-generate` | Generate NER training data |
| `uv run ner-train` | Fine-tune BERT models |
| `uv run ner-evaluate` | Evaluate vs spaCy |
| `uv run ner-compare` | Compare all trained models |
| `uv run ner-export` | Export model to FW-L2 |

---

## Running Tests

```bash
cd backend

python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_loader.py -v      # Loader (no deps)
python -m pytest tests/test_fw_l2.py -v       # FW-L2 (needs spaCy)
python -m pytest tests/test_retriever.py -v   # Retriever (needs FAISS index)
python -m pytest tests/test_pipeline.py -v -s # Pipeline (needs LLM provider)
```

| Test File | Tests | Dependencies |
|-----------|:-----:|-------------|
| `test_loader.py` | 20 | None |
| `test_fw_l2.py` | 29 | spaCy + (optionally) fine-tuned BERT |
| `test_retriever.py` | 35 | FAISS index |
| `test_pipeline.py` | 22 | FAISS index + LLM |
| **Total** | **106** | |

**Experiment notebooks** (run on Google Colab, results tracked in W&B):
| Notebook | What it validates |
|----------|------------------|
| 01_training | Model training convergence, token-level F1 |
| 02_evaluation | Entity-level F1, cross-validation stability, statistical significance |
| 03_error_analysis | FN/FP categorization, error patterns |
| 04_ablation | Data/training/imbalance sensitivity |
| 05_baselines | Comparison vs spaCy, Presidio, BioClinicalBERT |

---

## Docker

### With W&B / Groq (recommended)

```bash
docker compose up backend -d    # Just the backend
```

### With Ollama (local LLM)

```bash
docker compose --profile ollama up -d    # Backend + Ollama + auto model pull
```

### Environment Variables in Docker

| Variable | Source | Notes |
|----------|-------|-------|
| `EMBEDDING_MODEL` | `.env` â†’ build arg + runtime | Pre-downloaded during build |
| `LLM_PROVIDER` | `.env` | `wandb`, `groq`, or `ollama` |
| `LLM_MODEL` | `.env` | Model name for selected provider |
| `WANDB_API_KEY` | `.env` | Required for `wandb` provider |
| `GROQ_API_KEY` | `.env` | Required for `groq` provider |
| `OLLAMA_HOST` | Hardcoded `ollama` | Docker service name |
| `OLLAMA_PORT` | `.env` | Same in both contexts |
| `TOP_K` | `.env` | Chunks to retrieve |

---

## Phase Summary

| Phase | Goal | Status |
|-------|------|:------:|
| **0** | Environment Setup | Done |
| **1** | Data Ingestion (load, clean, chunk, embed, FAISS) | Done |
| **1.8** | PII ground truth Index | Done |
| **2** | Retriever (semantic search + section filtering) | Done |
| **2.1** | Retrieval Quality Tests | Done |
| **2.2** | Adversarial Retrieval Tests C1-C4 | Done |
| **3** | Generator (W&B + Groq + Ollama, dual system prompts) | Done |
| **3.3** | RAG Pipeline (retriever â†’ generator â†’ FW-L2) | Done |
| **3.4** | Benign + Adversarial Pipeline Tests | Done |
| **3.5** | 1000-Query Adversarial Golden Set (C1-C5) | Done |
| **3.6** | Evaluation CLI (sequential + parallel + Weave) | Done |
| **4.1** | FW-L2: Regex scanner (SSN, phone, email, DOB, MRN) | Done |
| **4.2** | FW-L2: Injection artifact detection | Done |
| **4.3** | FW-L2: Wired into RAG pipeline | Done |
| **4.5** | FW-L2: spaCy NER for names/addresses | Done |
| **4.5b** | FW-L2: Medical term whitelist (reduces false positives) | Done |
| **4.6** | Metadata leakage detection + scorer | Done |
| **4.7** | Weave leaderboard (profile comparison) | Done |
| **4.8** | BERT NER fine-tuning (training + Weave evaluation) | Done |
| **4.9** | BERT NER integrated into FW-L2 as `fw_l2_bert` profile | Done |
| **4.10** | Recursive chunker fix (overlap within 2048 limit) | Done |
| **4.11** | Academic evaluation notebooks (5 notebooks on Colab) | Done |
| **5** | FastAPI Backend (wire routes to pipeline) | Done |
| **6** | Evaluation + Observability (Weave integration) | Done |
| **7** | FW-L1 Model Training and ONNX Export | In progress |
| **8** | Android App + FW-L1 Integration | **Pending** |
| **9** | Full System Evaluation | **Pending** |

---

## Key Findings

### System prompt alone is not sufficient

With the **naive prompt**, the LLM freely outputs SSNs, names, and addresses from the context chunks. The **hardened prompt** blocks all tested adversarial queries (0% leak rate), but system prompts are brittle â€” jailbreaks evolve and different model versions may behave differently.

### Metadata leakage is a real risk

The original hardened prompt leaked metadata: "SSNs are listed in the demographics section, but I'm not allowed to reveal that information." This confirms to an attacker that SSNs exist and where they're stored. The tightened prompt gives no information: "I can only answer clinical questions."

### FW-L2 catches what the LLM leaks

When the naive prompt is used, the LLM outputs SSNs freely. FW-L2 catches and redacts all SSN patterns via regex. Two NER backends are available:
- **`fw_l2_base`** (spaCy): Catches city/state names, misses Synthea-format names and full addresses
- **`fw_l2_bert`** (fine-tuned DistilBERT): 100% name detection, 100% address detection, trained on our data

### Fine-tuned BERT outperforms spaCy for PII detection

DistilBERT fine-tuned on 8,332 Synthea examples achieves F1=0.9949 â€” near-perfect detection of names and addresses. Key findings:
- Smallest model (66M params) performs best â€” the task doesn't need capacity, it needs domain-specific patterns
- Training converges in 2-3 epochs; 10 epochs is unnecessary
- All models achieve 100% name detection; addresses (~99%) are slightly harder

### Ground truth must not be used for detection

Using `phi_groundtruth.json` to detect names/addresses in FW-L2 would be cheating â€” in production there's no lookup table. Ground truth is only for evaluation (measuring what was missed).

### Chunker recursive split matters

The original chunker allowed chunks up to 2,249 chars (overlap pushed past the 2,048 limit), causing silent truncation by the embedding model. Fixed by reserving space for overlap within the limit. After fix: 0 chunks exceed 2,048 chars.

### Defense in depth

```
Mobile App
    â”‚
    â”œâ”€â”€ FW-L1: Block adversarial queries (on-device)
    â”‚
    â–¼
Backend
    â”‚
    â”œâ”€â”€ Retriever: Embed query â†’ FAISS search â†’ top-k chunks
    â”‚
    â”œâ”€â”€ Generator: System prompt + context + query â†’ LLM
    â”‚       â”œâ”€â”€ Naive prompt (no guardrails â€” for testing FW-L2)
    â”‚       â””â”€â”€ Hardened prompt (refuses PII requests, no metadata leakage)
    â”‚
    â”œâ”€â”€ FW-L2: Scan and redact PII from response
    â”‚       â”œâ”€â”€ Regex (SSN, phone, email, DOB, MRN)
    â”‚       â”œâ”€â”€ NER (names, addresses â€” spaCy or fine-tuned BERT)
    â”‚       â”œâ”€â”€ Injection detection
    â”‚       â””â”€â”€ Metadata leak detection (evaluation only)
    â”‚
    â–¼
Safe Response to Mobile App
```

Each layer catches what the previous one missed:
- **FW-L1** blocks adversarial queries before they cost compute
- **System prompt** instructs the LLM to refuse PII requests
- **FW-L2** redacts any PII that slips through in the response

