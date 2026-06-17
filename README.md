# SQDeChain: Enhancing Tourist Trust and Destination Image Through a Hybrid AI–Blockchain Framework for Tourism 5.0

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo%2017369377-blue.svg)](https://zenodo.org/records/17369377)
[![Repository](https://img.shields.io/badge/GitHub-rajavavek%2FSQDeChain-black.svg)](https://github.com/rajavavek/SQDeChain)


Research code repository for a Tourism 5.0 framework that combines lightweight multimodal artificial intelligence with shard-aware Byzantine Fault Tolerant blockchain verification. The pipeline computes three tourism key performance indicators (KPIs): **Service Quality Trust Index (SQTI)**, **Destination Image Reliability Score (DIRS)**, and **Tokenised Loyalty Propensity (TLP)**. These KPI bundles are then hashed and verified using a simulated **ClusterPioneer BFT** consensus layer.

> **Repository status:** This repository is a research prototype for reproducibility, benchmarking, and academic publication support. It is not a production blockchain deployment.

---

## Table of contents

- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start: smoke test](#quick-start-smoke-test)
- [Dataset](#dataset)
- [Full experiment workflow](#full-experiment-workflow)
- [Key configuration values](#key-configuration-values)
- [Main scripts](#main-scripts)
- [Outputs](#outputs)
- [Reproducibility notes](#reproducibility-notes)
- [GitHub upload checklist](#github-upload-checklist)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Method overview

The implementation follows the SQDeChain/SQcoin paper pipeline:

```text
Text reviews       -> DistilDeBERTa-compatible text model -> SQTI
Destination images -> MobileViT-compatible image model     -> DIRS
Behaviour logs     -> GRU sequence model                   -> TLP
SQTI + DIRS + TLP  -> deterministic KPI hash                -> ClusterPioneer BFT shard simulation
Verified KPIs      -> token allocation + fairness + adversarial evaluation
```

Core components:

- **Multimodal AI:** text, image, and behavioural models for tourism analytics.
- **KPI computation:** SQTI, DIRS, TLP, token allocation, Gini fairness, and deterministic KPI hashing.
- **Blockchain simulation:** reputation-aware validator selection, shard routing, BFT quorum, finality, and tokenomics.
- **Experiments:** smoke tests, Zenodo-data configuration, ablation study, benchmark tables, adversarial tests, and publication-style figures.

---

## Repository structure

```text
.
├── configs/
│   ├── smoke.yaml                 # Lightweight smoke-test configuration
│   ├── default.yaml               # Default experiment configuration
│   └── zenodo_17369377.yaml       # Full-data configuration template
├── data/
│   ├── sample/                    # Small synthetic sample for smoke tests
│   └── README.md                  # Data handling notes
├── docs/
│   ├── DATA_SCHEMA.md             # Required columns and derived fields
│   ├── METHOD_ALIGNMENT.md        # Paper-to-code mapping
│   ├── REPRODUCIBILITY.md         # Reproducibility protocol
│   ├── GITHUB_UPLOAD_CHECKLIST.md # GitHub release checklist
│   └── paper/                     # Local paper copy, if allowed by the journal/submission policy
├── results/smoke/                 # Example output files from the smoke run
├── scripts/                       # Command-line entry points
├── src/sqdechain/                 # Python package source code
├── tests/                         # Unit tests
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rajavavek/SQDeChain.git
cd SQDeChain
```

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

For the lightweight smoke-test pipeline:

```bash
pip install -r requirements.txt
pip install -e .
```

For full transformer and MobileViT training:

```bash
pip install -e ".[full]"
```

For development and tests:

```bash
pip install -e ".[dev,full]"
```

---




The data schema is documented in [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

> **Important:** Do not commit large raw data files, model checkpoints, or generated artifacts directly to GitHub. Keep the dataset on Zenodo or another external data repository, and keep only scripts, configuration files, and documentation in GitHub.

---

## Full experiment workflow

### 1. Prepare and validate the data

```bash
python scripts/prepare_data.py --config configs/zenodo_17369377.yaml --outdir data/processed
```

### 2. Train or score each modality

```bash
python scripts/train_text.py --config configs/zenodo_17369377.yaml --outdir results/zenodo
python scripts/train_image.py --config configs/zenodo_17369377.yaml --outdir results/zenodo
python scripts/train_behavior.py --config configs/zenodo_17369377.yaml --outdir results/zenodo
```

### 3. Compute KPIs and hashes

```bash
python scripts/run_kpi_pipeline.py \
  --config configs/zenodo_17369377.yaml \
  --score-dir results/zenodo \
  --outdir results/zenodo
```

### 4. Run blockchain simulation and benchmarks

```bash
python scripts/simulate_blockchain.py --config configs/zenodo_17369377.yaml --outdir results/zenodo
python scripts/run_benchmarks.py --config configs/zenodo_17369377.yaml --outdir results/zenodo
```

### 5. Run ablation and generate figures

```bash
python scripts/run_ablation.py --config configs/zenodo_17369377.yaml --outdir results/zenodo
python scripts/make_publication_figures.py --results results/zenodo
```

---

## Key configuration values

The full-data configuration is stored in `configs/zenodo_17369377.yaml`. Important defaults include:

| Component | Setting | Default |
|---|---:|---:|
| Split | train / validation / test | 70 / 15 / 15 |
| Text model | max sequence length | 256 |
| Text model | learning rate | 2e-5 |
| Text model | batch size | 32 |
| Text model | epochs | 5 |
| Image model | input resolution | 224 x 224 |
| Image model | learning rate | 1e-4 |
| Image model | batch size | 64 |
| Image model | epochs | 20 |
| GRU | hidden units | 128 |
| GRU | sequence length | 30 |
| GRU | dropout | 0.30 |
| GRU | learning rate | 1e-3 |
| Blockchain | shard counts | 2, 4, 6, 8 |
| Blockchain | committee size | 11 |
| Blockchain | Byzantine faults | 3 |

---

## Main scripts

| Script | Purpose |
|---|---|
| `scripts/download_zenodo.py` | Download Zenodo record files and verify checksums where available. |
| `scripts/make_synthetic_data.py` | Generate a small local dataset for testing. |
| `scripts/prepare_data.py` | Validate and normalize input files. |
| `scripts/train_text.py` | Train or score the text model for SQTI. |
| `scripts/train_image.py` | Train or score the image model for DIRS. |
| `scripts/train_behavior.py` | Train or score the GRU model for TLP. |
| `scripts/run_kpi_pipeline.py` | Compute SQTI, DIRS, TLP, token allocation, and KPI hashes. |
| `scripts/simulate_blockchain.py` | Run ClusterPioneer BFT scalability and latency simulation. |
| `scripts/run_benchmarks.py` | Generate benchmark and adversarial-test tables. |
| `scripts/run_ablation.py` | Measure the contribution of each modality. |
| `scripts/make_publication_figures.py` | Generate publication-style figures from CSV outputs. |

---

## Outputs

The pipeline writes machine-readable results and figures:

| Output | Description |
|---|---|
| `text_scores.csv` | Text-model quality and sentiment outputs. |
| `image_scores.csv` | Image reliability and provenance outputs. |
| `behavior_scores.csv` | GRU loyalty predictions. |
| `kpi_scores.csv` | SQTI, DIRS, TLP, token values, and deterministic KPI hashes. |
| `scalability_latency.csv` | Shard-level throughput, latency, and finality results. |
| `token_fairness.csv` | Token distribution and Gini fairness metrics. |
| `adversarial_tests.csv` | Sybil, poisoning, and Byzantine attack metrics. |

---










Please include:

1. operating system,
2. Python version,
3. command that failed,
4. full error message,
5. whether the smoke dataset or the Zenodo dataset was used.
