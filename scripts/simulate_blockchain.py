#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from sqdechain.config import load_config
from sqdechain.blockchain.simulator import run_scalability_simulation, run_consensus_baselines


def main():
    p = argparse.ArgumentParser(description="Run ClusterPioneer BFT scalability simulation.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", default="results/blockchain")
    args = p.parse_args()
    cfg = load_config(args.config)
    print(run_scalability_simulation(cfg, args.outdir))
    print(run_consensus_baselines(args.outdir))


if __name__ == "__main__":
    main()
