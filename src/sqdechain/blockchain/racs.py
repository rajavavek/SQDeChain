from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import math

import numpy as np


@dataclass
class Validator:
    validator_id: str
    stake: float
    accuracy_history: List[float]
    uptime_ratio: float
    slashing_events: int
    region: str
    slashed_in_last_epoch: bool = False


def reputation_score(v: Validator, gamma: float = 0.90, beta: float = 0.10, lambda_slash: float = 0.20) -> float:
    hist = np.asarray(v.accuracy_history, dtype=float)
    if hist.size == 0:
        acc = 0.5
    else:
        weights = np.asarray([gamma ** (hist.size - i - 1) for i in range(hist.size)], dtype=float)
        acc = float(np.sum(weights * hist) / np.sum(weights))
    score = acc + beta * math.log(1.0 + max(v.uptime_ratio, 0.0)) - lambda_slash * v.slashing_events
    return float(np.clip(score, 0.0, 1.0))


def select_committee(
    validators: Iterable[Validator],
    committee_size: int = 11,
    alpha: float = 0.50,
    beta: float = 0.10,
    penalty_factor: float = 0.50,
    max_per_region: int = 4,
) -> List[Validator]:
    vals = list(validators)
    if not vals:
        return []
    stakes = np.asarray([max(v.stake, 0.0) for v in vals], dtype=float)
    stake_min, stake_max = float(stakes.min()), float(stakes.max())
    denom = stake_max - stake_min if stake_max > stake_min else 1.0
    scored = []
    for v, s in zip(vals, stakes):
        stake_norm = (s - stake_min) / denom
        rep = reputation_score(v, beta=beta)
        score = alpha * stake_norm + (1.0 - alpha) * rep
        if v.slashed_in_last_epoch:
            score *= penalty_factor
        score *= 1.0 + beta * math.log(1.0 + max(v.uptime_ratio, 0.0))
        scored.append((score, v))
    scored.sort(key=lambda item: item[0], reverse=True)
    committee: List[Validator] = []
    region_counts = {}
    leftovers = []
    for _, v in scored:
        count = region_counts.get(v.region, 0)
        if count < max_per_region and len(committee) < committee_size:
            committee.append(v)
            region_counts[v.region] = count + 1
        else:
            leftovers.append(v)
    for v in leftovers:
        if len(committee) >= committee_size:
            break
        committee.append(v)
    return committee[:committee_size]


def make_validators(n: int, seed: int = 42) -> List[Validator]:
    rng = np.random.default_rng(seed)
    regions = ["APAC", "EU", "NA", "MEA", "LATAM"]
    validators = []
    for i in range(n):
        validators.append(Validator(
            validator_id=f"val_{i:04d}",
            stake=float(rng.lognormal(mean=2.0, sigma=0.5)),
            accuracy_history=rng.uniform(0.80, 1.0, size=5).tolist(),
            uptime_ratio=float(rng.uniform(0.85, 1.0)),
            slashing_events=int(rng.choice([0, 0, 0, 1])),
            region=str(rng.choice(regions)),
            slashed_in_last_epoch=bool(rng.random() < 0.05),
        ))
    return validators
