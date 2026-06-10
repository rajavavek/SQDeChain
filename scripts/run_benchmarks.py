#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from sqdechain.config import load_config
from sqdechain.blockchain.simulator import run_consensus_baselines, run_end_to_end_benchmarks
from sqdechain.blockchain.attacks import simulate_adversarial_tests


def main():
    p = argparse.ArgumentParser(description="Run SQDeChain benchmark tables and adversarial tests.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", default="results/benchmarks")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("project", {}).get("seed", 42))
    print(run_consensus_baselines(args.outdir))
    print(run_end_to_end_benchmarks(args.outdir))
    print(simulate_adversarial_tests(seed=seed, outdir=args.outdir))


if __name__ == "__main__":
    main()
