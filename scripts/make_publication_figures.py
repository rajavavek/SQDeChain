#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from pathlib import Path
import pandas as pd
from sqdechain.visualization.plots import plot_scalability, plot_ablation, plot_consensus_baselines


def main():
    p = argparse.ArgumentParser(description="Create publication-style figures from CSV outputs.")
    p.add_argument("--results", default="results/smoke")
    args = p.parse_args()
    results = Path(args.results)
    if (results / "scalability_latency.csv").exists():
        print(plot_scalability(pd.read_csv(results / "scalability_latency.csv"), results))
    if (results / "ablation_study.csv").exists():
        print(plot_ablation(pd.read_csv(results / "ablation_study.csv"), results))
    if (results / "consensus_baselines.csv").exists():
        print(plot_consensus_baselines(pd.read_csv(results / "consensus_baselines.csv"), results))


if __name__ == "__main__":
    main()
