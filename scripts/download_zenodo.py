#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
import os
from sqdechain.data.zenodo import download_record


def main():
    p = argparse.ArgumentParser(description="Download a Zenodo record and verify checksums where available.")
    p.add_argument("--record-id", default="17369377")
    p.add_argument("--output", default="data/raw/zenodo_17369377")
    p.add_argument("--token", default=os.getenv("ZENODO_TOKEN"), help="Optional Zenodo API token, or set ZENODO_TOKEN.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    files = download_record(args.record_id, args.output, token=args.token, overwrite=args.overwrite)
    print("Downloaded files:")
    for path in files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
