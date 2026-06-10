from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_scalability(df: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure_scalability_latency.png"
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["number_of_shards"], df["throughput_tps"], marker="o", label="Throughput (TPS)")
    ax1.set_xlabel("Number of shards")
    ax1.set_ylabel("Throughput (TPS)")
    ax2 = ax1.twinx()
    ax2.plot(df["number_of_shards"], df["consensus_latency_ms"], marker="s", label="Consensus latency (ms)")
    ax2.set_ylabel("Consensus latency (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_ablation(df: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure_ablation.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["variant"], df["r2"])
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_consensus_baselines(df: pd.DataFrame, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure_consensus_baselines.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["protocol"], df["throughput_tps"])
    ax.set_ylabel("Throughput (TPS)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
