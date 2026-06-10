# Data Directory

- `sample/`: small synthetic dataset for smoke tests and CI.
- `raw/`: place downloaded Zenodo files here. This directory is ignored by Git.
- `processed/`: normalized files produced by `scripts/prepare_data.py`. This directory is ignored by Git.

Run:

```bash
python scripts/download_zenodo.py --record-id 17369377 --output data/raw/zenodo_17369377
python scripts/prepare_data.py --config configs/zenodo_17369377.yaml
```
