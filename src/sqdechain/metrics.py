from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, r2_score, accuracy_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def regression_r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(r2_score(np.asarray(list(y_true)), np.asarray(list(y_pred))))


def classification_metrics(y_true, y_pred, average: str = "weighted") -> ClassificationMetrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def bootstrap_ci(values: Iterable[float], confidence: float = 0.95, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    alpha = 1.0 - confidence
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_t_test_bonferroni(a, b, m: int, alpha: float = 0.05):
    from scipy.stats import ttest_rel

    stat, p = ttest_rel(a, b)
    return {
        "t_statistic": float(stat),
        "p_value": float(p),
        "bonferroni_alpha": float(alpha / max(m, 1)),
        "significant": bool(p < alpha / max(m, 1)),
    }
