from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Sequence

import numpy as np


def normalize_1_5_rating(rating: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(rating), dtype=float)
    return np.clip((arr - 1.0) / 4.0, 0.0, 1.0)


def service_quality_trust_index(q: Iterable[float], w: Iterable[float], alpha: float = 0.8) -> float:
    q_arr = np.asarray(list(q), dtype=float)
    w_arr = np.asarray(list(w), dtype=float)
    if q_arr.size == 0:
        return float("nan")
    if np.any(w_arr < 0):
        raise ValueError("SQTI weights must be non-negative")
    if np.sum(w_arr) == 0:
        w_arr = np.ones_like(q_arr)
    sqti = float(np.sum(w_arr * q_arr) / np.sum(w_arr))
    median = float(np.median(q_arr))
    return float(alpha * sqti + (1.0 - alpha) * median)


def destination_image_reliability_score(
    visual_similarity: Iterable[float],
    text_polarity: Iterable[float],
    user_credibility: Iterable[float],
    beta: Sequence[float] = (0.5, 0.3, 0.2),
) -> float:
    b1, b2, b3 = beta
    if not np.isclose(b1 + b2 + b3, 1.0):
        raise ValueError("DIRS beta weights must sum to 1")
    v = np.asarray(list(visual_similarity), dtype=float)
    p = np.asarray(list(text_polarity), dtype=float)
    c = np.asarray(list(user_credibility), dtype=float)
    if len(v) == 0:
        return float("nan")
    score = b1 * v + b2 * p + b3 * c
    return float(np.mean(np.clip(score, 0.0, 1.0)))


def tokenised_loyalty_propensity(
    visit_frequency: float,
    duration: float,
    sqti: float,
    dirs: float,
    gamma: Sequence[float] = (0.25, 0.20, 0.30, 0.25),
) -> float:
    if not np.isclose(sum(gamma), 1.0):
        raise ValueError("TLP gamma weights must sum to 1")
    g1, g2, g3, g4 = gamma
    raw = g1 * visit_frequency + g2 * duration + g3 * sqti + g4 * dirs
    return float(np.clip(raw, 0.0, 1.0))


def allocate_tokens(tlp: float, reward_lambda: float = 100.0, token_cap: float = 100.0) -> float:
    return float(min(reward_lambda * float(tlp), token_cap))


def gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float).flatten()
    if arr.size == 0:
        return float("nan")
    if np.amin(arr) < 0:
        arr = arr - np.amin(arr)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff_sum / (2 * arr.size**2 * mean))


def hash_kpi_bundle(bundle: Dict) -> str:
    """Deterministic SHA-256 hash of a KPI bundle for blockchain validation."""
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_kpi_bundle(user_id: str, entity_id: str, sqti: float, dirs: float, tlp: float, raw_hash: str = "") -> Dict:
    return {
        "user_id": str(user_id),
        "entity_id": str(entity_id),
        "sqti": round(float(sqti), 8),
        "dirs": round(float(dirs), 8),
        "tlp": round(float(tlp), 8),
        "raw_hash": raw_hash,
    }
