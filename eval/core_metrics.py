"""Evaluation metrics for STE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _to_bool(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == bool:
        return x
    return x.astype(float) > 0.5


def f1_score_core(pred_core: np.ndarray, true_core: np.ndarray) -> float:
    """F1 for binary core membership vectors."""
    p = _to_bool(pred_core)
    t = _to_bool(true_core)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, np.logical_not(t)).sum()
    fn = np.logical_and(np.logical_not(p), t).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


def jaccard_index(pred_core: np.ndarray, true_core: np.ndarray) -> float:
    p = _to_bool(pred_core)
    t = _to_bool(true_core)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE over scalar probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    # Clip to avoid binning issues
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.shape[0]

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y[mask])
        conf = np.mean(p[mask])
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def top_k_f1(ranking: np.ndarray, true_core: np.ndarray) -> float:
    """Convert ranking (rank positions) to a top-|C*| set and compute F1."""
    r = np.asarray(ranking)
    t = _to_bool(true_core)
    k = int(t.sum())
    k = max(k, 1)
    pred = (r < k)
    return f1_score_core(pred.astype(float), t.astype(float))


def top_1_in_core(ranking: np.ndarray, true_core: np.ndarray) -> float:
    r = np.asarray(ranking)
    t = _to_bool(true_core)
    best = int(np.argmin(r))
    return float(t[best])


@dataclass
class CoreMetrics:
    f1: float
    jaccard: float

    @staticmethod
    def from_vectors(pred: np.ndarray, true: np.ndarray) -> 'CoreMetrics':
        return CoreMetrics(
            f1=f1_score_core(pred, true),
            jaccard=jaccard_index(pred, true)
        )
