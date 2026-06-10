#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from sqdechain.config import load_config, set_seed
from sqdechain.data.loaders import read_table, ensure_behavior_columns
from sqdechain.models.behavior import train_behavior_model


def main():
    p = argparse.ArgumentParser(description="Train/score GRU behavioural model for TLP.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", default="results/behavior")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    df = ensure_behavior_columns(read_table(cfg["data"]["behavior_logs"]))
    metrics = train_behavior_model(df, cfg, args.outdir)
    print(metrics)


if __name__ == "__main__":
    main()
