# Reproducibility Protocol

## 1. Environment

Recommended:

- Python 3.10+
- PyTorch 2.x
- Hugging Face Transformers for text model training
- timm for MobileViT
- CUDA GPU for full dataset training

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Dataset

Download the Zenodo record:

```bash
python scripts/download_zenodo.py --record-id 17369377 --output data/raw/zenodo_17369377
```

Confirm the files match `docs/DATA_SCHEMA.md`. If filenames differ, update `configs/zenodo_17369377.yaml`.

## 3. Seeds and repetitions

All scripts use `project.seed`. The full experiment configuration uses five repeated runs, bootstrap confidence intervals, and Bonferroni-corrected comparisons.

## 4. Outputs

Each experiment writes:

- `*.csv`: machine-readable tables
- `*.json`: metadata and configuration snapshots
- `*.png`: publication figures

## 5. Verifying KPI hashes

`run_kpi_pipeline.py` writes `kpi_scores.csv`, including `kpi_hash`. The hash is computed over canonical JSON fields and is deterministic for the same input values.

## 6. Hardware differences

Full transformer/MobileViT training can vary by GPU, library versions, and nondeterministic CUDA kernels. For strict reproducibility, pin package versions, use deterministic PyTorch settings, and record `run_metadata.json`.
