#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from sqdechain.config import load_config, save_config_snapshot
from sqdechain.experiments.pipeline import compute_kpis_from_scores


def main():
    p = argparse.ArgumentParser(description="Compute SQTI, DIRS, TLP, KPI hashes, and tokens.")
    p.add_argument("--config", required=True)
    p.add_argument("--score-dir", default=None, help="Directory containing text_scores.csv/image_scores.csv/behavior_scores.csv. Defaults to outdir.")
    p.add_argument("--outdir", default="results/kpi")
    args = p.parse_args()
    cfg = load_config(args.config)
    save_config_snapshot(cfg, args.outdir)
    score_dir = args.score_dir or args.outdir
    df = compute_kpis_from_scores(cfg, score_dir, args.outdir)
    print(df.head())


if __name__ == "__main__":
    main()
