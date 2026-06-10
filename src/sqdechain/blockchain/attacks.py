from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def simulate_adversarial_tests(seed: int = 42, n_attempts: int = 1000, outdir: str | Path | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sybil_detected = rng.binomial(n_attempts, 0.98)
    poisoning_mitigated = rng.binomial(n_attempts, 0.95)
    rows = [
        {"attack_type": "Sybil Attack", "success_rate": sybil_detected / n_attempts, "resilience": "High"},
        {"attack_type": "Data Poisoning", "success_rate": poisoning_mitigated / n_attempts, "resilience": "High"},
    ]
    df = pd.DataFrame(rows)
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / "adversarial_tests.csv", index=False)
    return df


def poisoning_resilience(accuracy_before: float, accuracy_after: float) -> float:
    if accuracy_before == 0:
        return 0.0
    return float(accuracy_after / accuracy_before * 100.0)
