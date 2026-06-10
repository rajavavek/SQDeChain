# Method Alignment with the SQDeChain Paper

This document maps the implementation to the paper's methods and experiments.

## Multimodal AI

| Paper component | Repository implementation |
|---|---|
| DistilDeBERTa text sentiment / service quality | `src/sqdechain/models/text.py`, `scripts/train_text.py` |
| MobileViT image feature/reliability model | `src/sqdechain/models/image.py`, `scripts/train_image.py` |
| GRU behavioural loyalty model | `src/sqdechain/models/behavior.py`, `scripts/train_behavior.py` |
| SQTI, DIRS, TLP equations | `src/sqdechain/kpis.py` |
| KPI bundle hash | `src/sqdechain/kpis.py::hash_kpi_bundle` |

## Blockchain / Consensus

| Paper component | Repository implementation |
|---|---|
| Reputation-aware validator selection (RACS) | `src/sqdechain/blockchain/racs.py` |
| ClusterPioneer shard BFT | `src/sqdechain/blockchain/consensus.py`, `simulator.py` |
| Byzantine quorum `2f + 1` | `src/sqdechain/blockchain/consensus.py` |
| Token distribution and Gini fairness | `src/sqdechain/blockchain/tokenomics.py`, `src/sqdechain/kpis.py` |
| Sybil / poisoning / Byzantine tests | `src/sqdechain/blockchain/attacks.py` |

## Experiments

| Paper result category | Repository script |
|---|---|
| Simulation throughput and latency | `scripts/simulate_blockchain.py` |
| Real-world/testbed-style metrics | `scripts/run_benchmarks.py` |
| Predictive metrics (R², precision, recall, F1) | `scripts/train_*.py`, `src/sqdechain/metrics.py` |
| Token fairness | `scripts/run_kpi_pipeline.py`, `scripts/simulate_blockchain.py` |
| Ablation | `scripts/run_ablation.py` |
| Publication plots/tables | `scripts/make_publication_figures.py` |

## Important reproducibility note

The exact numerical values in the paper depend on the exact raw Zenodo files, model checkpoints, hardware, and random seeds. This repository provides the complete pipeline and a smoke dataset; for exact paper reproduction, download the Zenodo record, set the file paths in `configs/zenodo_17369377.yaml`, and run the full experiment commands in `README.md`.
