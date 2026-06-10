#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from pathlib import Path
from sqdechain.config import load_config, set_seed
from sqdechain.data.loaders import read_table, ensure_text_columns
from sqdechain.models.text import train_text_model


def main():
    p = argparse.ArgumentParser(description="Train/score text review model for SQTI.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", default="results/text")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    df = ensure_text_columns(read_table(cfg["data"]["text_reviews"]))
    metrics = train_text_model(df, cfg, args.outdir)
    print(metrics)


if __name__ == "__main__":
    main()
