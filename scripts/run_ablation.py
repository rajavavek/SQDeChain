#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from sqdechain.experiments.ablation import run_ablation


def main():
    p = argparse.ArgumentParser(description="Run SQDeChain multimodal ablation study.")
    p.add_argument("--config", required=False)
    p.add_argument("--outdir", default="results/ablation")
    args = p.parse_args()
    print(run_ablation(args.outdir))


if __name__ == "__main__":
    main()
