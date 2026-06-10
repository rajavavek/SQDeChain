#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from pathlib import Path
import shutil

from sqdechain.config import load_config
from sqdechain.data.loaders import read_table, ensure_text_columns, ensure_image_columns, ensure_behavior_columns


def main():
    p = argparse.ArgumentParser(description="Validate and normalize SQDeChain data files.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", default="data/processed")
    args = p.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    text = ensure_text_columns(read_table(data["text_reviews"]))
    image = ensure_image_columns(read_table(data["image_manifest"]))
    behavior = ensure_behavior_columns(read_table(data["behavior_logs"]))
    text.to_csv(outdir / "text_reviews.csv", index=False)
    image.to_csv(outdir / "image_manifest.csv", index=False)
    behavior.to_csv(outdir / "behavior_logs.csv", index=False)
    print(f"Prepared data written to {outdir}")


if __name__ == "__main__":
    main()
