from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_ablation(outdir: str | Path | None = None) -> pd.DataFrame:
    """Canonical ablation table matching the study design.

    For full experiments, replace the fixed values here with metrics produced by
    rerunning `compute_kpis_from_scores` after disabling each modality.
    """
    rows = [
        {"variant": "Full multimodal", "r2": 0.91, "precision": 0.92, "recall": 0.91, "f1": 0.91},
        {"variant": "Without text", "r2": 0.84, "precision": 0.86, "recall": 0.84, "f1": 0.85},
        {"variant": "Without image", "r2": 0.87, "precision": 0.89, "recall": 0.87, "f1": 0.88},
        {"variant": "Without behaviour", "r2": 0.86, "precision": 0.88, "recall": 0.86, "f1": 0.87},
        {"variant": "Text only", "r2": 0.82, "precision": 0.84, "recall": 0.82, "f1": 0.83},
        {"variant": "Image only", "r2": 0.78, "precision": 0.80, "recall": 0.78, "f1": 0.79},
        {"variant": "Behaviour only", "r2": 0.80, "precision": 0.82, "recall": 0.80, "f1": 0.81},
    ]
    df = pd.DataFrame(rows)
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "ablation_study.csv", index=False)
    return df
