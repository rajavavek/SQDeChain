#!/usr/bin/env python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from pathlib import Path
import json
import time
import pandas as pd

from sqdechain.config import load_config, set_seed, save_config_snapshot
from sqdechain.data.loaders import read_table, ensure_text_columns, ensure_image_columns, ensure_behavior_columns
from sqdechain.models.text import train_text_model
from sqdechain.models.image import train_image_model
from sqdechain.models.behavior import train_behavior_model
from sqdechain.experiments.pipeline import compute_kpis_from_scores
from sqdechain.blockchain.simulator import run_scalability_simulation, run_consensus_baselines, run_end_to_end_benchmarks
from sqdechain.blockchain.attacks import simulate_adversarial_tests
from sqdechain.experiments.ablation import run_ablation
from sqdechain.visualization.plots import plot_scalability, plot_ablation, plot_consensus_baselines


def main():
    p = argparse.ArgumentParser(description="Run the end-to-end SQDeChain reproducibility pipeline.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", default="results/smoke")
    args = p.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, outdir)

    print("[1/8] Loading data...", flush=True)
    text = ensure_text_columns(read_table(cfg["data"]["text_reviews"]))
    image = ensure_image_columns(read_table(cfg["data"]["image_manifest"]))
    behavior = ensure_behavior_columns(read_table(cfg["data"]["behavior_logs"]))

    print("[2/8] Training/scoring text model...", flush=True)
    metrics = []
    metrics.append({"component": "text", **train_text_model(text, cfg, outdir)})
    print("[3/8] Training/scoring image model...", flush=True)
    metrics.append({"component": "image", **train_image_model(image, cfg, outdir)})
    print("[4/8] Training/scoring behavior model...", flush=True)
    metrics.append({"component": "behavior", **train_behavior_model(behavior, cfg, outdir)})
    pd.DataFrame(metrics).to_csv(outdir / "model_metrics_summary.csv", index=False)

    print("[5/8] Computing KPI bundles and tokens...", flush=True)
    compute_kpis_from_scores(cfg, outdir, outdir)
    print("[6/8] Running blockchain simulations...", flush=True)
    scalability = run_scalability_simulation(cfg, outdir)
    baselines = run_consensus_baselines(outdir)
    run_end_to_end_benchmarks(outdir)
    simulate_adversarial_tests(seed=int(cfg.get("project", {}).get("seed", 42)), outdir=outdir)
    print("[7/8] Running ablation and creating figures...", flush=True)
    ablation = run_ablation(outdir)
    plot_scalability(scalability, outdir)
    plot_consensus_baselines(baselines, outdir)
    plot_ablation(ablation, outdir)

    print("[8/8] Writing metadata...", flush=True)
    metadata = {"runtime_seconds": round(time.time() - t0, 3), "config": args.config, "outdir": args.outdir}
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed SQDeChain pipeline in {metadata['runtime_seconds']}s. Outputs: {outdir}")


if __name__ == "__main__":
    main()
