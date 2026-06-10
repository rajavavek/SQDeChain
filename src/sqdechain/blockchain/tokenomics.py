from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from sqdechain.kpis import allocate_tokens, gini


def distribute_tokens(kpi_df: pd.DataFrame, reward_lambda: float = 100.0, token_cap: float = 100.0, outdir: str | Path | None = None) -> pd.DataFrame:
    out = kpi_df.copy()
    out["tokens"] = out["tlp"].apply(lambda x: allocate_tokens(x, reward_lambda, token_cap))
    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir / "token_distribution.csv", index=False)
        pd.DataFrame([{"gini": gini(out["tokens"])}]).to_csv(outdir / "token_fairness.csv", index=False)
    return out
