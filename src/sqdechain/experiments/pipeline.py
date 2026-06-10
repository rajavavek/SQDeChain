from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sqdechain.data.loaders import read_table, ensure_text_columns, ensure_image_columns, ensure_behavior_columns
from sqdechain.kpis import (
    normalize_1_5_rating,
    service_quality_trust_index,
    destination_image_reliability_score,
    tokenised_loyalty_propensity,
    build_kpi_bundle,
    hash_kpi_bundle,
)
from sqdechain.blockchain.tokenomics import distribute_tokens


def compute_kpis_from_scores(cfg: Dict, score_dir: str | Path, outdir: str | Path) -> pd.DataFrame:
    score_dir = Path(score_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    kpi_cfg = cfg.get("kpi", {})

    text = ensure_text_columns(read_table(data_cfg["text_reviews"]))
    image = ensure_image_columns(read_table(data_cfg["image_manifest"]))
    behavior = ensure_behavior_columns(read_table(data_cfg["behavior_logs"]))

    text_scores_path = score_dir / "text_scores.csv"
    image_scores_path = score_dir / "image_scores.csv"
    behavior_scores_path = score_dir / "behavior_scores.csv"

    if text_scores_path.exists():
        text_scores = pd.read_csv(text_scores_path)
        text = text.merge(text_scores[["review_id", "text_quality_score"]], on="review_id", how="left")
        fallback_q = pd.Series(normalize_1_5_rating(text["rating"]), index=text.index)
        text["q_i"] = text["text_quality_score"].fillna(fallback_q)
    else:
        text["q_i"] = normalize_1_5_rating(text["rating"])
    text["w_i"] = text["credibility"].astype(float) * text["recency_weight"].astype(float)

    if image_scores_path.exists():
        image_scores = pd.read_csv(image_scores_path)
        image = image.merge(image_scores[["image_id", "visual_similarity"]], on="image_id", how="left")
    image["visual_similarity"] = image.get("visual_similarity", image.get("label", 0.5)).fillna(0.5)

    if behavior_scores_path.exists():
        behavior_scores = pd.read_csv(behavior_scores_path)
    else:
        behavior_scores = behavior.groupby("user_id").agg(loyalty_propensity=("loyalty_label", "mean")).reset_index()

    sqti_entity = text.groupby("entity_id").apply(lambda g: service_quality_trust_index(g["q_i"], g["w_i"], alpha=float(kpi_cfg.get("sqti_alpha", 0.8)))).rename("sqti").reset_index()
    dirs_entity = image.groupby("entity_id").apply(lambda g: destination_image_reliability_score(
        g["visual_similarity"], g["text_polarity"], g["user_credibility"], beta=kpi_cfg.get("dirs_beta", [0.5, 0.3, 0.2])
    )).rename("dirs").reset_index()
    behavior_user = behavior.groupby("user_id").agg(visit_frequency=("visit_frequency", "mean"), duration=("duration", "mean")).reset_index()
    behavior_user = behavior_user.merge(behavior_scores, on="user_id", how="left")

    # Cross product of user/entity subset for demonstration; real pipeline aligns by session/booking if available.
    entities = sqti_entity.merge(dirs_entity, on="entity_id", how="outer").fillna(0.5)
    rows = []
    for _, user in behavior_user.iterrows():
        # Assign stable entity by user hash for deterministic alignment.
        eidx = abs(hash(str(user["user_id"]))) % len(entities)
        ent = entities.iloc[eidx]
        tlp_base = tokenised_loyalty_propensity(
            float(user["visit_frequency"]),
            float(user["duration"]),
            float(ent["sqti"]),
            float(ent["dirs"]),
            gamma=kpi_cfg.get("tlp_gamma", [0.25, 0.20, 0.30, 0.25]),
        )
        tlp = float(np.clip((tlp_base + float(user.get("loyalty_propensity", tlp_base))) / 2.0, 0, 1))
        bundle = build_kpi_bundle(user["user_id"], ent["entity_id"], float(ent["sqti"]), float(ent["dirs"]), tlp)
        bundle["kpi_hash"] = hash_kpi_bundle(bundle)
        rows.append(bundle)
    kpis = pd.DataFrame(rows)
    kpis.to_csv(outdir / "kpi_scores.csv", index=False)
    distribute_tokens(kpis, reward_lambda=float(kpi_cfg.get("token_lambda", 100.0)), token_cap=float(kpi_cfg.get("token_cap", 100.0)), outdir=outdir)
    return kpis
