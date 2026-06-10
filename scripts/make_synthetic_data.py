#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from sqdechain.data.synthetic import make_synthetic_dataset


def main():
    p = argparse.ArgumentParser(description="Create synthetic SQDeChain sample data.")
    p.add_argument("--output", default="data/sample")
    p.add_argument("--n-reviews", type=int, default=500)
    p.add_argument("--n-images", type=int, default=80)
    p.add_argument("--n-users", type=int, default=200)
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_synthetic_dataset(args.output, args.n_reviews, args.n_images, args.n_users, args.sequence_length, args.seed)
    print(f"Synthetic dataset written to {args.output}")


if __name__ == "__main__":
    main()
