#!/usr/bin/env python3
"""NeurIPS-grade experiment suite for Soft Tournament Equilibrium (STE).

This script is intentionally self-contained. It supports:

1. Controlled planted-core synthetic recovery with multiple seeds.
2. Ablations over edge estimator, reachability operator, path length, and temperature.
3. Bootstrap recovery/stability diagnostics.
4. Runtime scaling.
5. Human-preference / Chatbot-Arena-style real-data analysis.
6. AgentBench-style execution-log analysis.
7. Paper asset generation: CSV summaries, LaTeX tables, and figures.

The suite is designed to answer reviewer-style questions:
- Does the method recover a known tournament-theoretic core?
- Does it beat strong ranking/rating baselines under a fair set conversion?
- What fails under extreme sparsity/noise?
- Which operator/estimator choices matter?
- Is the evidence robust across seeds and bootstrap resamples?
- Do real human preference data contain high-confidence cycles?

Dependencies:
    numpy pandas scipy scikit-learn matplotlib pyyaml

Optional dependencies:
    tqdm networkx

Usage examples:
    python -m ste_neurips.neurips_suite synthetic --config configs/synthetic_mac.yaml
    python -m ste_neurips.neurips_suite real-arena --input data/arena.csv --out outputs/arena
    python -m ste_neurips.neurips_suite agentbench --input data/agentbench_scores.csv --out outputs/agentbench
    python -m ste_neurips.neurips_suite summarize --out outputs/neurips_main
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from scipy.optimize import minimize
    from scipy.special import betainc, expit, logit
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
except Exception as exc:  # pragma: no cover
    minimize = None
    betainc = None
    expit = None
    logit = None
    sparse = None
    lsqr = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
except Exception as exc:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None
    brier_score_loss = None
    log_loss = None

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None

try:
    import trueskill as _trueskill
except Exception:  # pragma: no cover
    _trueskill = None



# ---------------------------------------------------------------------------
# Configuration and generic helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_csv_ints(s: str) -> List[int]:
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_csv_floats(s: str) -> List[float]:
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def get_cfg(cfg: Mapping, key: str, default):
    return cfg.get(key, default)

def df_to_markdown(df: pd.DataFrame, **kwargs) -> str:
    """Return a Markdown table without crashing if tabulate is absent.

    Pandas delegates DataFrame.to_markdown to the optional `tabulate` package.
    We include tabulate in requirements, but this fallback keeps smoke tests and
    reviewer artifact generation robust in minimal environments.
    """
    try:
        return df.to_markdown(**kwargs)
    except ImportError:
        index = kwargs.get("index", True)
        return df.to_string(index=index)




def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0, mean
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    half = 1.96 * se
    return mean, half, se


def entropy_from_counts(items: Sequence[int], n: int) -> float:
    if len(items) == 0:
        return float("nan")
    counts = np.bincount(np.asarray(items, dtype=int), minlength=n).astype(float)
    p = counts[counts > 0] / float(len(items))
    return float(-(p * np.log2(p)).sum())


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo = np.nanmin(scores)
    hi = np.nanmax(scores)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - lo) / (hi - lo)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Hard tournament solutions
# ---------------------------------------------------------------------------


def majority_adjacency(P: np.ndarray) -> np.ndarray:
    A = (np.asarray(P) > 0.5).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def adjacency_from_counts(wins: np.ndarray, comps: np.ndarray, tie_policy: str = "zero") -> np.ndarray:
    n = wins.shape[0]
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j or comps[i, j] <= 0:
                continue
            if wins[i, j] > wins[j, i]:
                A[i, j] = 1
            elif wins[i, j] == wins[j, i] and tie_policy == "random":
                # deterministic pseudo-random tie break for reproducibility
                A[i, j] = int((i * 1315423911 + j * 2654435761) % 2 == 0)
    np.fill_diagonal(A, 0)
    return A


def top_cycle_from_adj(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=bool)
    n = A.shape[0]
    tc = np.zeros(n, dtype=np.int32)
    for s in range(n):
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v in np.nonzero(A[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        tc[s] = int(seen.all())
    return tc


def uncovered_from_adj(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=bool)
    n = A.shape[0]
    covered = np.zeros(n, dtype=bool)
    for c in range(n):
        for a in range(n):
            if c == a or not A[c, a]:
                continue
            beaten_by_a = np.nonzero(A[a])[0]
            if np.all(A[c, beaten_by_a]):
                covered[a] = True
    return (~covered).astype(np.int32)


def copeland_scores_from_counts(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    n = wins.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            if wins[i, j] > wins[j, i]:
                scores[i] += 1.0
                scores[j] -= 1.0
            elif wins[j, i] > wins[i, j]:
                scores[j] += 1.0
                scores[i] -= 1.0
    return scores


# ---------------------------------------------------------------------------
# Planted core synthetic generator
# ---------------------------------------------------------------------------


def cyclic_orientation(i: int, j: int, s: int) -> bool:
    if s <= 1:
        return False
    d = (j - i) % s
    return 1 <= d <= (s - 1) // 2


@dataclass
class PlantedTournament:
    P: np.ndarray
    A: np.ndarray
    true_tc: np.ndarray
    true_uc: np.ndarray
    planted_core: np.ndarray
    permutation: np.ndarray
    metadata: Dict


def make_planted_core_tournament(
    n: int,
    core_size: int,
    margin_core: float = 0.22,
    margin_out: float = 0.26,
    outside_scale: float = 1.0,
    seed: int = 0,
    randomize_labels: bool = True,
    mode: str = "cyclic_core",
) -> PlantedTournament:
    """Generate a probabilistic tournament with known Smith/top-cycle core.

    mode='transitive' returns a singleton Condorcet core.
    mode='cyclic_core' returns a planted strongly connected top cycle.
    """
    if core_size < 1 or core_size > n:
        raise ValueError("core_size must be in [1,n]")
    if mode == "cyclic_core" and core_size > 1 and core_size % 2 == 0:
        raise ValueError("Use odd core_size for the regular cyclic core.")

    rng = np.random.default_rng(seed)
    P = np.full((n, n), 0.5, dtype=np.float64)

    if mode == "transitive":
        strengths = np.linspace(2.5, -2.5, n)
        strengths += rng.normal(scale=0.02, size=n)
        for i in range(n):
            for j in range(i + 1, n):
                p = 1.0 / (1.0 + math.exp(-(strengths[i] - strengths[j])))
                p = 0.5 + np.sign(p - 0.5) * max(abs(p - 0.5), margin_out)
                P[i, j] = float(np.clip(p, 0.51, 0.99))
                P[j, i] = 1.0 - P[i, j]
        core_size = 1
    else:
        core = np.arange(core_size, dtype=int)
        outside = np.arange(core_size, n, dtype=int)

        for ii, a in enumerate(core):
            for jj, b in enumerate(core):
                if a == b or core_size == 1:
                    continue
                if cyclic_orientation(ii, jj, core_size):
                    P[a, b] = 0.5 + margin_core
                    P[b, a] = 0.5 - margin_core

        # Every core member beats every outsider, making the Smith/top-cycle set exactly C.
        for c in core:
            for o in outside:
                jitter = rng.uniform(-0.02, 0.02)
                p = float(np.clip(0.5 + margin_out + jitter, 0.51, 0.99))
                P[c, o] = p
                P[o, c] = 1.0 - p

        if len(outside) > 1:
            strengths = np.linspace(1.5, -1.5, len(outside)) * outside_scale
            strengths += rng.normal(scale=0.05, size=len(outside))
            for ii, a in enumerate(outside):
                for jj, b in enumerate(outside):
                    if a == b:
                        continue
                    p = 1.0 / (1.0 + math.exp(-(strengths[ii] - strengths[jj])))
                    p = 0.5 + np.sign(p - 0.5) * max(abs(p - 0.5), 0.08)
                    P[a, b] = float(np.clip(p, 0.01, 0.99))

    np.fill_diagonal(P, 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            P[j, i] = 1.0 - P[i, j]

    perm = np.arange(n)
    if randomize_labels:
        perm = rng.permutation(n)
        P = P[np.ix_(perm, perm)]

    A = majority_adjacency(P)
    true_tc = top_cycle_from_adj(A)
    true_uc = uncovered_from_adj(A)
    planted_core = (perm < core_size).astype(np.int32) if randomize_labels else np.array([1 if i < core_size else 0 for i in range(n)], dtype=np.int32)
    return PlantedTournament(
        P=P,
        A=A,
        true_tc=true_tc,
        true_uc=true_uc,
        planted_core=planted_core,
        permutation=perm,
        metadata={"mode": mode, "n": n, "core_size": int(core_size), "margin_core": margin_core, "margin_out": margin_out},
    )


def sample_counts(
    P: np.ndarray,
    m_per_pair: int,
    missing_rate: float,
    label_noise: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample aggregate wins and comparison counts from a probabilistic tournament."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    wins = np.zeros((n, n), dtype=np.int32)
    comps = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < missing_rate:
                continue
            p = P[i, j]
            if label_noise > 0:
                p = (1.0 - label_noise) * p + label_noise * (1.0 - p)
            w = int(rng.binomial(m_per_pair, p))
            wins[i, j] = w
            wins[j, i] = m_per_pair - w
            comps[i, j] = comps[j, i] = m_per_pair
    return wins, comps




def normalized_smin(z: np.ndarray, gamma: float, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    gamma = max(float(gamma), 1e-12)
    m = z.shape[axis]
    x = -z / gamma
    xmax = np.max(x, axis=axis, keepdims=True)
    lse = np.squeeze(xmax, axis=axis) + np.log(np.sum(np.exp(x - xmax), axis=axis))
    return -gamma * (lse - math.log(float(m)))


def smooth_max_bounded(z: np.ndarray, gamma: float, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    gamma = max(float(gamma), 1e-12)
    x = z / gamma
    x = x - np.max(x, axis=axis, keepdims=True)
    w = np.exp(x)
    w = w / np.sum(w, axis=axis, keepdims=True)
    return np.sum(w * z, axis=axis)

# ---------------------------------------------------------------------------
# STE edge estimators and reachability operators
# ---------------------------------------------------------------------------


def posterior_edge_from_counts(wins: np.ndarray, comps: np.ndarray, prior: float = 0.5) -> np.ndarray:
    """Beta-posterior evidence that an edge has majority direction.

    D_ab = max(0, 2*Pr(theta_ab > 0.5 | data)-1), so missing or ambiguous
    pairs contribute near-zero directed evidence instead of fake bidirectional
    reachability.
    """
    if betainc is None:
        raise RuntimeError("scipy.special.betainc is required for posterior_edge.")
    n = wins.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            a = prior + wins[i, j]
            b = prior + wins[j, i]
            # Regularized incomplete beta I_x(a,b) is beta CDF at x.
            cdf_half = float(betainc(a, b, 0.5))
            pr_i_majority = 1.0 - cdf_half
            D[i, j] = max(0.0, 2.0 * pr_i_majority - 1.0)
            D[j, i] = max(0.0, 2.0 * (1.0 - pr_i_majority) - 1.0)
    np.fill_diagonal(D, 0.0)
    return D



def posterior_majority_probs_from_counts(wins: np.ndarray, comps: np.ndarray, prior: float = 0.5) -> np.ndarray:
    """Posterior probability that each directed edge has majority direction.

    Entry Pi[i,j] is Pr(theta_ij > 1/2 | wins/comparisons) under a symmetric
    Beta(prior, prior) prior. Missing pairs are deliberately left at 0.5.
    This is the edge-uncertainty model used by the posterior-edge STE
    reporting estimator in the paper.
    """
    if betainc is None:
        raise RuntimeError("scipy.special.betainc is required for posterior majority probabilities.")
    n = wins.shape[0]
    Pi = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            a = prior + wins[i, j]
            b = prior + wins[j, i]
            pr_i_majority = 1.0 - float(betainc(a, b, 0.5))
            Pi[i, j] = float(np.clip(pr_i_majority, 0.0, 1.0))
            Pi[j, i] = 1.0 - Pi[i, j]
    np.fill_diagonal(Pi, 0.5)
    return Pi


def sample_tournament_from_edge_probs(Pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one hard tournament from pairwise edge-majority probabilities."""
    n = Pi.shape[0]
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < Pi[i, j]:
                A[i, j] = 1
            else:
                A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def posterior_membership_scores(
    wins: np.ndarray,
    comps: np.ndarray,
    solution: str = "uc",
    samples: int = 200,
    seed: int = 0,
    prior: float = 0.5,
) -> np.ndarray:
    """Monte Carlo posterior-edge STE membership scores.

    This implements Eq. (posterior-edge) in the NeurIPS draft: sample hard
    tournaments from the edge-direction posterior and average hard TC/UC
    membership indicators. Scores are membership frequencies under the stated
    edge-uncertainty model, not universal probabilities of absolute quality.
    """
    n = wins.shape[0]
    solution = solution.lower().strip()
    if samples <= 0:
        raise ValueError("posterior samples must be positive")
    Pi = posterior_majority_probs_from_counts(wins, comps, prior=prior)
    rng = np.random.default_rng(seed)
    acc = np.zeros(n, dtype=np.float64)
    for _ in range(samples):
        A = sample_tournament_from_edge_probs(Pi, rng)
        if solution == "tc":
            acc += top_cycle_from_adj(A)
        elif solution == "uc":
            acc += uncovered_from_adj(A)
        else:
            raise ValueError("solution must be 'tc' or 'uc'")
    return acc / float(samples)


def parse_ste_method(method: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Parse STE method names.

    Returns (edge_estimator, solution, posterior_mode).
    solution is 'tc' or 'uc'. posterior_mode means sampled hard-tournament
    posterior membership rather than a plug-in soft operator.
    """
    m = method.lower().strip()
    if m in {"ste", "ste_posterior_edge", "posterior_edge", "posterior_edge_uc"}:
        return "posterior_edge", "uc", True
    if m in {"ste_posterior_edge_uc", "ste_post_uc"}:
        return "posterior_edge", "uc", True
    if m in {"ste_posterior_edge_tc", "ste_post_tc"}:
        return "posterior_edge", "tc", True

    # Plug-in aliases. Bare ste_weighted_mean is kept as UC to match the paper's
    # main finite-sample table, which reports STE plug-in UC.
    aliases = {
        "ste_plugin_uc": ("weighted_mean", "uc", False),
        "ste_plugin_tc": ("weighted_mean", "tc", False),
        "ste_weighted_mean": ("weighted_mean", "uc", False),
        "ste_weighted_mean_uc": ("weighted_mean", "uc", False),
        "ste_weighted_mean_tc": ("weighted_mean", "tc", False),
        "ste_missing_as_half": ("missing_as_half", "uc", False),
        "ste_missing_as_half_uc": ("missing_as_half", "uc", False),
        "ste_missing_as_half_tc": ("missing_as_half", "tc", False),
        "ste_hard_majority_uc": ("hard_majority", "uc", False),
        "ste_hard_majority_tc": ("hard_majority", "tc", False),
        "weighted_mean": ("weighted_mean", "uc", False),
        "missing_as_half": ("missing_as_half", "uc", False),
    }
    if m in aliases:
        return aliases[m]

    # Generic pattern: ste_<estimator>_<tc|uc>
    mm = re.match(r"^ste_(posterior_edge|weighted_mean|missing_as_half|hard_majority)_(tc|uc)$", m)
    if mm:
        estimator, sol = mm.group(1), mm.group(2)
        return estimator, sol, estimator == "posterior_edge"
    return None, None, False


def target_for_method(method: str, true_tc: np.ndarray, true_uc: np.ndarray) -> np.ndarray:
    """Choose the tournament-solution target corresponding to a method name."""
    m = method.lower().strip()
    if m.endswith("_tc") or m == "hard_tc":
        return true_tc
    if m.endswith("_uc") or m in {"hard_uc", "ste", "ste_posterior_edge", "posterior_edge", "ste_weighted_mean", "weighted_mean"}:
        return true_uc
    # Ranking/rating baselines are compared to the refined core by default.
    return true_uc

def smoothed_mean_P(wins: np.ndarray, comps: np.ndarray, prior: float = 0.5) -> np.ndarray:
    n = wins.shape[0]
    P = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                p = 0.5
            else:
                p = (wins[i, j] + prior) / (comps[i, j] + 2.0 * prior)
            P[i, j] = p
            P[j, i] = 1.0 - p
    np.fill_diagonal(P, 0.5)
    return P


def weighted_mean_edge_from_counts(wins: np.ndarray, comps: np.ndarray, tau: float = 0.035, prior: float = 0.5, conf_scale: float = 5.0) -> np.ndarray:
    P = smoothed_mean_P(wins, comps, prior=prior)
    conf = comps.astype(float) / (comps.astype(float) + conf_scale)
    D = conf * sigmoid((P - 0.5) / max(tau, 1e-8))
    np.fill_diagonal(D, 0.0)
    return D


def missing_as_half_edge_from_counts(wins: np.ndarray, comps: np.ndarray, tau: float = 0.035, prior: float = 0.5) -> np.ndarray:
    P = smoothed_mean_P(wins, comps, prior=prior)
    D = sigmoid((P - 0.5) / max(tau, 1e-8))
    np.fill_diagonal(D, 0.0)
    return D


def hard_majority_edge_from_counts(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    A = adjacency_from_counts(wins, comps)
    return A.astype(float)


def edge_matrix_from_counts(wins: np.ndarray, comps: np.ndarray, estimator: str, tau: float = 0.035) -> np.ndarray:
    estimator = estimator.lower().strip()
    if estimator == "posterior_edge":
        return posterior_edge_from_counts(wins, comps)
    if estimator == "weighted_mean":
        return weighted_mean_edge_from_counts(wins, comps, tau=tau)
    if estimator in {"missing_as_half", "mean", "smoothed_mean"}:
        return missing_as_half_edge_from_counts(wins, comps, tau=tau)
    if estimator in {"hard_majority", "empirical_majority"}:
        return hard_majority_edge_from_counts(wins, comps)
    raise ValueError(f"Unknown edge estimator: {estimator}")


def soft_majority_edge_from_P(P: np.ndarray, tau: float = 0.035, zero_diag: bool = True) -> np.ndarray:
    D = sigmoid((P - 0.5) / max(tau, 1e-8))
    if zero_diag:
        np.fill_diagonal(D, 0.0)
    return D


def max_min_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Result[a,b] = max_c min(X[a,c], Y[c,b])
    return np.max(np.minimum(X[:, :, None], Y[None, :, :]), axis=1)


def reachability_max_min(D: np.ndarray, K: int) -> np.ndarray:
    n = D.shape[0]
    Q = D.copy()
    R = D.copy()
    for _ in range(2, K + 1):
        Q = max_min_product(Q, D)
        R = np.maximum(R, Q)
    np.fill_diagonal(R, 0.0)
    return R


def soft_boolean_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # 1 - prod_c(1 - X_ac * Y_cb), numerically clipped.
    vals = 1.0 - X[:, :, None] * Y[None, :, :]
    vals = np.clip(vals, 1e-12, 1.0)
    return 1.0 - np.prod(vals, axis=1)


def reachability_prob_or(D: np.ndarray, K: int) -> np.ndarray:
    Q = D.copy()
    R = D.copy()
    for _ in range(2, K + 1):
        Q = soft_boolean_product(Q, D)
        R = 1.0 - (1.0 - R) * (1.0 - Q)
        R = np.clip(R, 0.0, 1.0)
    np.fill_diagonal(R, 0.0)
    return R


def reachability_walk_count(D: np.ndarray, K: int, damping: float = 1.0, normalize: bool = True) -> np.ndarray:
    Q = D.copy()
    R = np.zeros_like(D, dtype=float)
    for k in range(1, K + 1):
        if k == 1:
            Q = D.copy()
        elif k > 1:
            Q = Q @ D
        R += (damping ** (k - 1)) * Q
    np.fill_diagonal(R, 0.0)
    if normalize:
        # For ranking metrics, normalize to avoid overflow while preserving order.
        R = normalize_scores(R)
        np.fill_diagonal(R, 0.0)
    return R


def reachability_from_edge(D: np.ndarray, K: int, mode: str = "max_min") -> np.ndarray:
    mode = mode.lower().strip()
    if mode == "max_min":
        return reachability_max_min(D, K)
    if mode in {"prob_or", "soft_boolean"}:
        return reachability_prob_or(D, K)
    if mode in {"walk_count", "old_walk_count", "matrix_power"}:
        return reachability_walk_count(D, K)
    raise ValueError(f"Unknown reachability mode: {mode}")


def ste_scores_from_edge(D: np.ndarray, K: Optional[int] = None, reachability: str = "max_min", gamma: float = 0.035) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    if K is None or K <= 0:
        K = n - 1
    R = reachability_from_edge(D, K=min(K, n - 1), mode=reachability)
    mask = ~np.eye(n, dtype=bool)
    vals = R[mask].reshape(n, n - 1)
    tc_scores = normalized_smin(vals, gamma=gamma, axis=1)

    cover = np.zeros_like(D, dtype=float)
    idx_all = np.arange(n)
    for c in range(n):
        for a in range(n):
            if c == a:
                continue
            mask_ca = (idx_all != a) & (idx_all != c)
            if np.any(mask_ca):
                witnesses = D[a, mask_ca] * (1.0 - D[c, mask_ca])
                violation = float(smooth_max_bounded(witnesses, gamma=gamma, axis=0))
            else:
                violation = 0.0
            cover[c, a] = D[c, a] * (1.0 - violation)
    q = np.zeros(n, dtype=float)
    for a in range(n):
        idx = [c for c in range(n) if c != a]
        q[a] = float(smooth_max_bounded(cover[idx, a], gamma=gamma, axis=0)) if idx else 0.0
    uc_scores = 1.0 - np.clip(q, 0.0, 1.0)
    return np.clip(tc_scores, 0.0, 1.0), np.clip(uc_scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Ranking/rating baselines
# ---------------------------------------------------------------------------


def win_rate_scores(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    # Row-average smoothed win probability, with missing pairs contributing 0.5.
    # This matches the natural ranking baseline for sparse round-robin data.
    P_hat = smoothed_mean_P(wins, comps, prior=0.5)
    n = wins.shape[0]
    return (P_hat * (~np.eye(n, dtype=bool))).sum(axis=1) / max(n - 1, 1)


def btl_scores(wins: np.ndarray, comps: np.ndarray, l2: float = 1e-3, maxiter: int = 200) -> np.ndarray:
    if minimize is None:
        return win_rate_scores(wins, comps)
    n = wins.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] > 0:
                pairs.append((i, j, float(wins[i, j]), float(wins[j, i])))
    if not pairs:
        return np.zeros(n, dtype=float)

    def obj(z):
        # Fix identifiability by centering inside objective.
        s = z - z.mean()
        loss = 0.5 * l2 * float(np.dot(s, s))
        grad = l2 * s
        for i, j, wij, wji in pairs:
            d = s[i] - s[j]
            p = 1.0 / (1.0 + math.exp(-float(np.clip(d, -50, 50))))
            # Negative log-likelihood for aggregate Bernoulli counts.
            loss -= wij * math.log(max(p, 1e-12)) + wji * math.log(max(1.0 - p, 1e-12))
            g = (wij + wji) * p - wij
            grad[i] += g
            grad[j] -= g
        # Project gradient to centered subspace.
        grad -= grad.mean()
        return loss, grad

    res = minimize(lambda z: obj(z), np.zeros(n), jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-8})
    s = res.x - res.x.mean()
    return s


def elo_scores(wins: np.ndarray, comps: np.ndarray, seed: int = 0, k_factor: float = 24.0, passes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = wins.shape[0]
    ratings = np.full(n, 1500.0, dtype=float)
    games = []
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            games.extend([(i, j, 1.0)] * int(wins[i, j]))
            games.extend([(i, j, 0.0)] * int(wins[j, i]))
    if not games:
        return ratings
    for _ in range(passes):
        rng.shuffle(games)
        for i, j, y in games:
            exp_i = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
            change = k_factor * (y - exp_i)
            ratings[i] += change
            ratings[j] -= change
    return ratings


def trueskill_scores(wins: np.ndarray, comps: np.ndarray, seed: int = 0) -> np.ndarray:
    """Optional TrueSkill baseline. Falls back to Elo if trueskill is unavailable."""
    if _trueskill is None:
        return elo_scores(wins, comps, seed=seed)
    rng = np.random.default_rng(seed)
    n = wins.shape[0]
    env = _trueskill.TrueSkill(draw_probability=0.0)
    ratings = [env.create_rating() for _ in range(n)]
    games = []
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            games.extend([(i, j, 0)] * int(wins[i, j]))  # i wins
            games.extend([(i, j, 1)] * int(wins[j, i]))  # j wins
    if not games:
        return np.zeros(n)
    rng.shuffle(games)
    for i, j, winner in games:
        if winner == 0:
            (ratings[i],), (ratings[j],) = env.rate([(ratings[i],), (ratings[j],)], ranks=[0, 1])
        else:
            (ratings[j],), (ratings[i],) = env.rate([(ratings[j],), (ratings[i],)], ranks=[0, 1])
    return np.asarray([r.mu - 3.0 * r.sigma for r in ratings], dtype=float)


def rank_centrality_scores(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Simple Rank-Centrality-style stationary scores from pairwise preferences."""
    n = wins.shape[0]
    # Pij = prob j beats i transition from i to j. Stronger nodes receive incoming mass.
    obs_degree = np.maximum((comps > 0).sum(axis=1).max(), 1)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or comps[i, j] <= 0:
                continue
            p_j_beats_i = wins[j, i] / max(comps[i, j], 1)
            M[i, j] = p_j_beats_i / obs_degree
    row_sum = M.sum(axis=1)
    for i in range(n):
        M[i, i] = max(0.0, 1.0 - row_sum[i])
    # Power iteration on row-stochastic chain.
    pi = np.full(n, 1.0 / n)
    for _ in range(200):
        pi_new = pi @ M
        if np.linalg.norm(pi_new - pi, 1) < 1e-12:
            break
        pi = pi_new
    return pi


def hodge_rank_scores(wins: np.ndarray, comps: np.ndarray, prior: float = 0.5) -> np.ndarray:
    if sparse is None or lsqr is None:
        return btl_scores(wins, comps)
    n = wins.shape[0]
    rows = []
    bvec = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            p = (wins[i, j] + prior) / (comps[i, j] + 2 * prior)
            y = float(safe_logit(np.array([p]))[0])
            rows.append((i, j))
            bvec.append(y)
            weights.append(math.sqrt(comps[i, j]))
    if not rows:
        return np.zeros(n)
    m = len(rows)
    data = []
    row_ind = []
    col_ind = []
    for r, (i, j) in enumerate(rows):
        w = weights[r]
        row_ind.extend([r, r])
        col_ind.extend([i, j])
        data.extend([w, -w])
    A = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(m, n)).tocsr()
    b = np.asarray(bvec) * np.asarray(weights)
    # Add tiny centering regularization by subtracting mean after solve.
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=1000)[0]
    return sol - sol.mean()


def pagerank_scores(wins: np.ndarray, comps: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    n = wins.shape[0]
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or comps[i, j] <= 0:
                continue
            # Edge from loser to winner.
            G[j, i] += wins[i, j] / max(comps[i, j], 1)
    row_sum = G.sum(axis=1)
    M = np.divide(G, row_sum[:, None], out=np.full((n, n), 1.0 / n), where=row_sum[:, None] > 0)
    pi = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(200):
        pi_new = alpha * (pi @ M) + (1.0 - alpha) * teleport
        if np.linalg.norm(pi_new - pi, 1) < 1e-12:
            break
        pi = pi_new
    return pi




def schulze_scores(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Condorcet/Schulze baseline: score by strongest-path pairwise wins."""
    n = wins.shape[0]
    strength = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or comps[i, j] <= 0:
                continue
            margin = wins[i, j] - wins[j, i]
            if margin > 0:
                strength[i, j] = float(margin) + 1e-6 * float(comps[i, j])
    pmat = strength.copy()
    for k in range(n):
        for i in range(n):
            via_ik = pmat[i, k]
            if i == k or via_ik <= 0:
                continue
            for j in range(n):
                if j == i or j == k:
                    continue
                pmat[i, j] = max(pmat[i, j], min(via_ik, pmat[k, j]))
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pmat[i, j] > pmat[j, i]:
                scores[i] += 1.0
            elif pmat[i, j] == pmat[j, i] and pmat[i, j] > 0:
                scores[i] += 0.5
    return scores


def minimax_scores(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Simpson/minimax Condorcet baseline: minimize worst pairwise defeat."""
    n = wins.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        worst_defeat = 0.0
        avg_margin = 0.0
        denom = 0
        for j in range(n):
            if i == j or comps[i, j] <= 0:
                continue
            margin = (wins[i, j] - wins[j, i]) / max(comps[i, j], 1)
            avg_margin += margin
            denom += 1
            if margin < 0:
                worst_defeat = max(worst_defeat, -margin)
        scores[i] = -worst_defeat + 1e-3 * (avg_margin / max(denom, 1))
    return scores


def _would_create_cycle(locked: np.ndarray, src: int, dst: int) -> bool:
    """Return True if adding src -> dst creates a directed cycle."""
    if src == dst:
        return True
    n = locked.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [dst]
    seen[dst] = True
    while stack:
        u = stack.pop()
        if u == src:
            return True
        for v in np.nonzero(locked[u])[0]:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    return False


def ranked_pairs_scores(wins: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Ranked Pairs/Tideman-style Condorcet baseline, returned as graph scores."""
    n = wins.shape[0]
    victories = []
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            margin = wins[i, j] - wins[j, i]
            if margin > 0:
                victories.append((float(margin), float(wins[i, j]), i, j))
            elif margin < 0:
                victories.append((float(-margin), float(wins[j, i]), j, i))
    victories.sort(reverse=True)
    locked = np.zeros((n, n), dtype=bool)
    for margin, support, winner, loser in victories:
        if not _would_create_cycle(locked, winner, loser):
            locked[winner, loser] = True
    outdeg = locked.sum(axis=1).astype(float)
    indeg = locked.sum(axis=0).astype(float)
    return outdeg - indeg


def kemeny_local_scores(wins: np.ndarray, comps: np.ndarray, seed: int = 0, restarts: int = 16, max_passes: int = 100) -> np.ndarray:
    """Approximate Kemeny-Young ranking via adjacent-swap and insertion local search.

    Exact Kemeny optimization is NP-hard, so this baseline is marked in the
    report as an approximate Kemeny local-search baseline.
    """
    n = wins.shape[0]
    if n == 0:
        return np.array([])
    rng = np.random.default_rng(seed)

    def cost(order: np.ndarray) -> float:
        c = 0.0
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)
        for i in range(n):
            for j in range(i + 1, n):
                if comps[i, j] <= 0:
                    continue
                if pos[i] < pos[j]:
                    c += wins[j, i]
                else:
                    c += wins[i, j]
        return float(c)

    starts = [np.argsort(-copeland_scores_from_counts(wins, comps))]
    try:
        starts.append(np.argsort(-btl_scores(wins, comps)))
    except Exception:
        pass
    for _ in range(max(0, restarts - len(starts))):
        starts.append(rng.permutation(n))

    best_order = starts[0].copy()
    best_cost = cost(best_order)
    for start in starts:
        order = start.copy()
        cur = cost(order)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for idx in range(n - 1):
                cand = order.copy()
                cand[idx], cand[idx + 1] = cand[idx + 1], cand[idx]
                cc = cost(cand)
                if cc + 1e-12 < cur:
                    order, cur = cand, cc
                    improved = True
            for _ in range(min(n, 20)):
                i, j = sorted(rng.choice(n, size=2, replace=False))
                if i == j:
                    continue
                cand = order.copy()
                item = cand[i]
                cand = np.delete(cand, i)
                cand = np.insert(cand, j, item)
                cc = cost(cand)
                if cc + 1e-12 < cur:
                    order, cur = cand, cc
                    improved = True
        if cur < best_cost:
            best_order, best_cost = order, cur
    scores = np.zeros(n, dtype=float)
    for r, agent in enumerate(best_order):
        scores[int(agent)] = float(n - r)
    return scores

def method_scores(
    method: str,
    wins: np.ndarray,
    comps: np.ndarray,
    tau: float = 0.035,
    K: Optional[int] = None,
    reachability: str = "max_min",
    seed: int = 0,
    posterior_samples: int = 200,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return a one-dimensional score vector for TC/core recovery."""
    n = wins.shape[0]
    if K is None or K <= 0:
        K = n - 1
    method = method.lower().strip()
    meta: Dict[str, float] = {}

    estimator, solution, posterior_mode = parse_ste_method(method)
    if estimator is not None:
        if posterior_mode:
            scores = posterior_membership_scores(wins, comps, solution=solution or "uc", samples=posterior_samples, seed=seed)
            meta["posterior_samples"] = float(posterior_samples)
            meta["solution"] = 1.0 if solution == "uc" else 0.0
            return scores, meta
        D = edge_matrix_from_counts(wins, comps, estimator=estimator, tau=tau)
        tc, uc = ste_scores_from_edge(D, K=K, reachability=reachability, gamma=tau)
        meta["mean_tc_score"] = float(np.mean(tc))
        meta["mean_uc_score"] = float(np.mean(uc))
        meta["solution"] = 1.0 if solution == "uc" else 0.0
        return (uc if solution == "uc" else tc), meta

    if method == "hard_tc":
        A = adjacency_from_counts(wins, comps)
        return top_cycle_from_adj(A).astype(float), meta
    if method == "hard_uc":
        A = adjacency_from_counts(wins, comps)
        return uncovered_from_adj(A).astype(float), meta
    if method == "copeland":
        return copeland_scores_from_counts(wins, comps), meta
    if method == "winrate" or method == "win_rate":
        return win_rate_scores(wins, comps), meta
    if method == "btl":
        return btl_scores(wins, comps), meta
    if method == "elo":
        return elo_scores(wins, comps, seed=seed), meta
    if method == "trueskill":
        return trueskill_scores(wins, comps, seed=seed), meta
    if method == "rank_centrality":
        return rank_centrality_scores(wins, comps), meta
    if method == "hodge":
        return hodge_rank_scores(wins, comps), meta
    if method == "pagerank":
        return pagerank_scores(wins, comps), meta
    if method == "schulze":
        return schulze_scores(wins, comps), meta
    if method == "minimax" or method == "simpson":
        return minimax_scores(wins, comps), meta
    if method == "ranked_pairs" or method == "tideman":
        return ranked_pairs_scores(wins, comps), meta
    if method == "kemeny" or method == "kemeny_local":
        return kemeny_local_scores(wins, comps, seed=seed), meta

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def select_topk(scores: np.ndarray, k: int, rng: Optional[np.random.Generator] = None, jitter: float = 1e-10) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    k = int(max(0, min(k, n)))
    if k == 0:
        return np.zeros(n, dtype=np.int32)
    if rng is None:
        rng = np.random.default_rng(0)
    s = scores + rng.normal(scale=jitter, size=n)
    idx = np.argpartition(-s, k - 1)[:k]
    pred = np.zeros(n, dtype=np.int32)
    pred[idx] = 1
    return pred


def f1_jaccard(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float, float]:
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    tp = float(np.sum(pred & true))
    fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true))
    tn = float(np.sum(~pred & ~true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return float(f1), float(jacc), float(fpr), float(fnr), float(tp)


def tie_randomized_metrics(scores: np.ndarray, true: np.ndarray, k: Optional[int] = None, seed: int = 0, repeats: int = 25) -> Dict[str, float]:
    true = np.asarray(true).astype(int)
    if k is None:
        k = int(true.sum())
    rng = np.random.default_rng(seed)
    f1s, jaccs, fprs, fnrs, tps = [], [], [], [], []
    for _ in range(max(1, repeats)):
        pred = select_topk(scores, k=k, rng=rng)
        f1, j, fpr, fnr, tp = f1_jaccard(pred, true)
        f1s.append(f1); jaccs.append(j); fprs.append(fpr); fnrs.append(fnr); tps.append(tp)
    out = {
        "topk_f1": float(np.mean(f1s)),
        "topk_jaccard": float(np.mean(jaccs)),
        "topk_fpr": float(np.mean(fprs)),
        "topk_fnr": float(np.mean(fnrs)),
        "topk_tp": float(np.mean(tps)),
    }
    # Ranking metrics.
    if len(np.unique(true)) == 2:
        try:
            out["auroc"] = float(roc_auc_score(true, scores)) if roc_auc_score else float("nan")
        except Exception:
            out["auroc"] = float("nan")
        try:
            out["auprc"] = float(average_precision_score(true, scores)) if average_precision_score else float("nan")
        except Exception:
            out["auprc"] = float("nan")
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    norm = normalize_scores(scores)
    try:
        out["brier"] = float(brier_score_loss(true, norm)) if brier_score_loss else float("nan")
    except Exception:
        out["brier"] = float("nan")
    out["ece"] = expected_calibration_error(norm, true, bins=10)
    out["score_gap"] = float(np.mean(scores[true == 1]) - np.mean(scores[true == 0])) if np.any(true == 1) and np.any(true == 0) else float("nan")
    out["pred_core_size_topk"] = float(k)
    return out


def expected_calibration_error(scores: np.ndarray, true: np.ndarray, bins: int = 10) -> float:
    scores = np.asarray(scores, dtype=float)
    true = np.asarray(true, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(scores)
    if n == 0:
        return float("nan")
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (scores >= lo) & (scores < hi if b < bins - 1 else scores <= hi)
        if not np.any(mask):
            continue
        conf = float(scores[mask].mean())
        acc = float(true[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def edge_orientation_error(D: np.ndarray, true_A: np.ndarray, observed: Optional[np.ndarray] = None) -> float:
    n = D.shape[0]
    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            if observed is not None and not observed[i, j]:
                continue
            pred_i = D[i, j] > D[j, i]
            true_i = bool(true_A[i, j])
            errors.append(float(pred_i != true_i))
    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Synthetic experiment suite
# ---------------------------------------------------------------------------


def run_oracle_sanity(out_dir: Path) -> pd.DataFrame:
    rows = []
    cases = [
        {"case": "transitive_singleton", "n": 30, "core_size": 1, "mode": "transitive"},
        {"case": "planted_3_core", "n": 30, "core_size": 3, "mode": "cyclic_core"},
        {"case": "planted_5_core", "n": 50, "core_size": 5, "mode": "cyclic_core"},
    ]
    for ix, c in enumerate(cases):
        T = make_planted_core_tournament(c["n"], c["core_size"], seed=1000 + ix, mode=c["mode"])
        D = majority_adjacency(T.P).astype(float)
        tc, uc = ste_scores_from_edge(D, K=T.P.shape[0] - 1, reachability="max_min", gamma=0.01)
        for target_name, target, scores in [("TC", T.true_tc, tc), ("UC", T.true_uc, uc)]:
            m = tie_randomized_metrics(scores, target, seed=ix, repeats=50)
            rows.append({"case": c["case"], "n": c["n"], "core_size": int(target.sum()), "target": target_name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "oracle_sanity.csv", index=False)
    return df


def run_synthetic_grid(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    n_values = parse_csv_ints(get_cfg(cfg, "n_values", "30,50,100"))
    core_sizes = parse_csv_ints(get_cfg(cfg, "core_sizes", "3,5,7"))
    m_values = parse_csv_ints(get_cfg(cfg, "m_values", "1,2,5,10,20,50"))
    missing_values = parse_csv_floats(get_cfg(cfg, "missing_values", "0,0.1,0.3,0.5"))
    label_noise_values = parse_csv_floats(get_cfg(cfg, "label_noise_values", str(get_cfg(cfg, "label_noise", 0.02))))
    margin_values = parse_csv_floats(get_cfg(cfg, "margin_out_values", str(get_cfg(cfg, "margin_out", 0.26))))
    seeds = int(get_cfg(cfg, "seeds", 40))
    tau = float(get_cfg(cfg, "tau", 0.035))
    K_policy = str(get_cfg(cfg, "K", "n-1"))
    reachability = str(get_cfg(cfg, "reachability", "max_min"))
    posterior_samples = int(get_cfg(cfg, "posterior_samples", 200))
    methods = get_cfg(cfg, "methods", ["ste_posterior_edge_uc", "ste_plugin_uc", "hard_uc", "hard_tc", "winrate", "btl", "elo", "rank_centrality", "hodge", "pagerank", "copeland"])
    if isinstance(methods, str):
        methods = [x.strip() for x in methods.split(",") if x.strip()]

    rows = []
    total = len(n_values) * len(core_sizes) * len(m_values) * len(missing_values) * len(label_noise_values) * len(margin_values) * seeds
    print(f"[synthetic] Running {total} trials x {len(methods)} methods")
    t0 = time.time()
    trial_ix = 0
    for n in n_values:
        for s in core_sizes:
            if s >= n or (s > 1 and s % 2 == 0):
                continue
            for margin_out in margin_values:
                for m in m_values:
                    for missing in missing_values:
                        for noise in label_noise_values:
                            for seed in range(seeds):
                                trial_seed = 17_000_000 + 100000 * n + 1000 * s + 100 * seed + int(1000 * missing) + int(100 * m) + int(1000 * noise)
                                T = make_planted_core_tournament(n, s, seed=trial_seed, margin_out=margin_out)
                                wins, comps = sample_counts(T.P, m_per_pair=m, missing_rate=missing, label_noise=noise, seed=trial_seed + 1)
                                K = n - 1 if K_policy == "n-1" else int(K_policy)
                                observed = comps > 0
                                for method in methods:
                                    start = time.perf_counter()
                                    try:
                                        scores, meta = method_scores(method, wins, comps, tau=tau, K=K, reachability=reachability, seed=trial_seed, posterior_samples=posterior_samples)
                                    except Exception as exc:
                                        print(f"[warning] method {method} failed: {exc}", file=sys.stderr)
                                        continue
                                    runtime = time.perf_counter() - start
                                    target = target_for_method(method, T.true_tc, T.true_uc)
                                    metrics = tie_randomized_metrics(scores, target, k=int(target.sum()), seed=trial_seed, repeats=25)
                                    est, _sol, _post = parse_ste_method(method)
                                    if est is not None and not _post:
                                        est = est
                                        D = edge_matrix_from_counts(wins, comps, estimator=est, tau=tau)
                                        metrics["edge_orientation_error_observed"] = edge_orientation_error(D, T.A, observed=observed)
                                    else:
                                        metrics["edge_orientation_error_observed"] = float("nan")
                                    rows.append({
                                        "trial": trial_ix,
                                        "seed": seed,
                                        "trial_seed": trial_seed,
                                        "n": n,
                                        "core_size": int(target.sum()),
                                        "target_solution": "UC" if target is T.true_uc else "TC",
                                        "planted_core_size": s,
                                        "m_per_pair": m,
                                        "missing_rate": missing,
                                        "label_noise": noise,
                                        "margin_out": margin_out,
                                        "method": method,
                                        "reachability": reachability if method.startswith("ste") else "na",
                                        "K": K,
                                        "tau": tau,
                                        "runtime_sec": runtime,
                                        **metrics,
                                        **meta,
                                    })
                                trial_ix += 1
                                if trial_ix % 200 == 0:
                                    elapsed = time.time() - t0
                                    print(f"[synthetic] {trial_ix}/{total} trials, elapsed {elapsed:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic_recovery.csv", index=False)
    return df


def run_ablation_grid(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    n_values = parse_csv_ints(get_cfg(cfg, "ablation_n_values", get_cfg(cfg, "n_values", "50")))
    core_sizes = parse_csv_ints(get_cfg(cfg, "ablation_core_sizes", get_cfg(cfg, "core_sizes", "5")))
    m_values = parse_csv_ints(get_cfg(cfg, "ablation_m_values", "5,10,20"))
    missing_values = parse_csv_floats(get_cfg(cfg, "ablation_missing_values", "0,0.3,0.5"))
    seeds = int(get_cfg(cfg, "ablation_seeds", min(30, int(get_cfg(cfg, "seeds", 40)))))
    edge_estimators = [x.strip() for x in str(get_cfg(cfg, "ablation_edge_estimators", "posterior_edge,weighted_mean,missing_as_half,hard_majority")).split(",") if x.strip()]
    reach_modes = [x.strip() for x in str(get_cfg(cfg, "ablation_reachability", "max_min,prob_or,walk_count")).split(",") if x.strip()]
    K_values_raw = [x.strip() for x in str(get_cfg(cfg, "ablation_K_values", "2,3,5,n-1")).split(",") if x.strip()]
    tau_values = parse_csv_floats(get_cfg(cfg, "ablation_tau_values", "0.01,0.035,0.1"))
    label_noise = float(get_cfg(cfg, "label_noise", 0.02))
    rows = []
    t0 = time.time()
    trial_ix = 0
    print("[ablation] Starting operator/estimator ablations")
    for n in n_values:
        for s in core_sizes:
            if s >= n or (s > 1 and s % 2 == 0):
                continue
            for m in m_values:
                for missing in missing_values:
                    for seed in range(seeds):
                        trial_seed = 91_000_000 + 100000 * n + 1000 * s + 100 * seed + int(1000 * missing) + m
                        T = make_planted_core_tournament(n, s, seed=trial_seed)
                        wins, comps = sample_counts(T.P, m_per_pair=m, missing_rate=missing, label_noise=label_noise, seed=trial_seed + 2)
                        for edge_est in edge_estimators:
                            for reach in reach_modes:
                                for Kraw in K_values_raw:
                                    K = n - 1 if Kraw == "n-1" else int(Kraw)
                                    for tau in tau_values:
                                        start = time.perf_counter()
                                        try:
                                            D = edge_matrix_from_counts(wins, comps, estimator=edge_est, tau=tau)
                                            tc_scores, uc_scores = ste_scores_from_edge(D, K=K, reachability=reach, gamma=tau)
                                            # Ablations focus on the refined Uncovered-Set core used in the main table.
                                            metrics = tie_randomized_metrics(uc_scores, T.true_uc, k=int(T.true_uc.sum()), seed=trial_seed, repeats=25)
                                        except Exception as exc:
                                            print(f"[warning] ablation failed: edge={edge_est}, reach={reach}, K={K}, tau={tau}: {exc}", file=sys.stderr)
                                            continue
                                        runtime = time.perf_counter() - start
                                        rows.append({
                                            "trial": trial_ix,
                                            "seed": seed,
                                            "n": n,
                                            "core_size": int(T.true_uc.sum()),
                                            "target_solution": "UC",
                                            "m_per_pair": m,
                                            "missing_rate": missing,
                                            "label_noise": label_noise,
                                            "edge_estimator": edge_est,
                                            "reachability": reach,
                                            "K": K,
                                            "tau": tau,
                                            "runtime_sec": runtime,
                                            **metrics,
                                        })
                        trial_ix += 1
                        if trial_ix % 100 == 0:
                            print(f"[ablation] {trial_ix} base trials, elapsed {time.time() - t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic_ablation.csv", index=False)
    return df


def bootstrap_counts_from_empirical(wins: np.ndarray, comps: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = wins.shape[0]
    bw = np.zeros_like(wins)
    bc = comps.copy()
    for i in range(n):
        for j in range(i + 1, n):
            N = int(comps[i, j])
            if N <= 0:
                continue
            p = wins[i, j] / N
            w = int(rng.binomial(N, p))
            bw[i, j] = w
            bw[j, i] = N - w
    return bw, bc


def run_bootstrap_stability(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    n = int(get_cfg(cfg, "bootstrap_n", 40))
    s = int(get_cfg(cfg, "bootstrap_core_size", 5))
    m = int(get_cfg(cfg, "bootstrap_m", 20))
    missing = float(get_cfg(cfg, "bootstrap_missing", 0.1))
    label_noise = float(get_cfg(cfg, "label_noise", 0.02))
    bootstraps = int(get_cfg(cfg, "bootstraps", 400))
    stability_seeds = int(get_cfg(cfg, "stability_seeds", 8))
    tau = float(get_cfg(cfg, "tau", 0.035))
    reachability = str(get_cfg(cfg, "reachability", "max_min"))
    posterior_samples = int(get_cfg(cfg, "posterior_samples", 200))
    methods = get_cfg(cfg, "bootstrap_methods", ["ste_posterior_edge_uc", "ste_plugin_uc", "winrate", "btl", "rank_centrality", "hodge"])
    if isinstance(methods, str):
        methods = [x.strip() for x in methods.split(",") if x.strip()]
    K = n - 1
    rows = []
    pred_sets_by_method: Dict[Tuple[int, str], List[np.ndarray]] = {}
    top1_by_method: Dict[Tuple[int, str], List[int]] = {}

    print(f"[bootstrap] n={n}, s={s}, m={m}, missing={missing}, B={bootstraps}, seeds={stability_seeds}")
    for seed in range(stability_seeds):
        trial_seed = 123_000_000 + 1000 * seed
        T = make_planted_core_tournament(n, s, seed=trial_seed)
        wins, comps = sample_counts(T.P, m_per_pair=m, missing_rate=missing, label_noise=label_noise, seed=trial_seed + 5)
        for b in range(bootstraps):
            bw, bc = bootstrap_counts_from_empirical(wins, comps, seed=trial_seed + 10_000 + b)
            for method in methods:
                try:
                    scores, meta = method_scores(method, bw, bc, tau=tau, K=K, reachability=reachability, seed=trial_seed + b, posterior_samples=posterior_samples)
                except Exception as exc:
                    print(f"[warning] bootstrap method {method} failed: {exc}", file=sys.stderr)
                    continue
                target = target_for_method(method, T.true_tc, T.true_uc)
                pred = select_topk(scores, k=int(target.sum()), rng=np.random.default_rng(trial_seed + b))
                f1, j, fpr, fnr, tp = f1_jaccard(pred, target)
                if len(np.unique(target)) == 2:
                    try:
                        auroc = float(roc_auc_score(target, scores)) if roc_auc_score else float("nan")
                    except Exception:
                        auroc = float("nan")
                else:
                    auroc = float("nan")
                rows.append({
                    "seed": seed,
                    "bootstrap": b,
                    "n": n,
                    "core_size": int(target.sum()),
                    "target_solution": "UC" if target is T.true_uc else "TC",
                    "m_per_pair": m,
                    "missing_rate": missing,
                    "label_noise": label_noise,
                    "method": method,
                    "bootstrap_f1": f1,
                    "bootstrap_jaccard_to_truth": j,
                    "bootstrap_fpr": fpr,
                    "bootstrap_fnr": fnr,
                    "bootstrap_auroc": auroc,
                    "top1": int(np.argmax(scores)),
                })
                pred_sets_by_method.setdefault((seed, method), []).append(pred)
                top1_by_method.setdefault((seed, method), []).append(int(np.argmax(scores)))

    df = pd.DataFrame(rows)
    # Pairwise Jaccard within each seed/method.
    pair_rows = []
    for (seed, method), sets in pred_sets_by_method.items():
        vals = []
        # sample pairs if too many
        rng = np.random.default_rng(seed + 999)
        B = len(sets)
        max_pairs = min(10000, B * (B - 1) // 2)
        for _ in range(max_pairs):
            i, j = rng.integers(0, B, size=2)
            if i == j:
                continue
            _, jac, _, _, _ = f1_jaccard(sets[i], sets[j])
            vals.append(jac)
        top_entropy = entropy_from_counts(top1_by_method[(seed, method)], n=n)
        pair_rows.append({"seed": seed, "method": method, "pairwise_jaccard": float(np.mean(vals)) if vals else float("nan"), "top1_entropy": top_entropy})
    pair_df = pd.DataFrame(pair_rows)
    df = df.merge(pair_df, on=["seed", "method"], how="left")
    df.to_csv(out_dir / "bootstrap_stability.csv", index=False)
    return df


def run_scaling(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    n_values = parse_csv_ints(get_cfg(cfg, "scaling_n_values", "50,100,200,500"))
    seeds = int(get_cfg(cfg, "scaling_seeds", 5))
    K_values = [x.strip() for x in str(get_cfg(cfg, "scaling_K_values", "2,3,5,n-1")).split(",") if x.strip()]
    reach_modes = [x.strip() for x in str(get_cfg(cfg, "scaling_reachability", "max_min,prob_or")).split(",") if x.strip()]
    rows = []
    print("[scaling] Starting operator runtime scaling")
    for n in n_values:
        s = 5 if n >= 10 else 3
        for seed in range(seeds):
            T = make_planted_core_tournament(n, s, seed=777_000 + n * 100 + seed)
            D = majority_adjacency(T.P).astype(float)
            for reach in reach_modes:
                for Kraw in K_values:
                    K = n - 1 if Kraw == "n-1" else int(Kraw)
                    start = time.perf_counter()
                    tc, uc = ste_scores_from_edge(D, K=K, reachability=reach)
                    runtime = time.perf_counter() - start
                    rows.append({"n": n, "seed": seed, "reachability": reach, "K": K, "runtime_sec": runtime, "tc_mean": float(tc.mean()), "uc_mean": float(uc.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runtime_scaling.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Real human-preference / arena analysis
# ---------------------------------------------------------------------------


def load_pairwise_csv(
    path: str,
    agent_a_col: str = "model_a",
    agent_b_col: str = "model_b",
    winner_col: str = "winner",
    category_col: Optional[str] = "category",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [agent_a_col, agent_b_col, winner_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Available columns: {list(df.columns)}")
    out = pd.DataFrame({
        "agent_a": df[agent_a_col].astype(str),
        "agent_b": df[agent_b_col].astype(str),
        "winner": df[winner_col].astype(str),
    })
    if category_col and category_col in df.columns:
        out["category"] = df[category_col].fillna("global").astype(str)
    else:
        out["category"] = "global"
    return out


def outcome_to_y(row) -> Optional[float]:
    a = str(row["agent_a"])
    b = str(row["agent_b"])
    w = str(row["winner"]).strip()
    wl = w.lower()
    if wl in {"tie", "draw", "both", "both_bad", "no_preference", "nan", "none"}:
        return None
    if w == a or wl in {"model_a", "a", "left", "winner_a", "1"}:
        return 1.0
    if w == b or wl in {"model_b", "b", "right", "winner_b", "0"}:
        return 0.0
    # Some Chatbot Arena exports use winner as 'model_a'/'model_b'; if exact names are not available, handle above.
    return None


def counts_from_pairwise_df(df: pd.DataFrame, min_agents: int = 2) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
    agents = sorted(set(df["agent_a"].astype(str)).union(set(df["agent_b"].astype(str))))
    if len(agents) < min_agents:
        return agents, np.zeros((len(agents), len(agents)), dtype=np.int32), np.zeros((len(agents), len(agents)), dtype=np.int32), 0
    idx = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    wins = np.zeros((n, n), dtype=np.int32)
    comps = np.zeros((n, n), dtype=np.int32)
    ties = 0
    for _, row in df.iterrows():
        a, b = str(row["agent_a"]), str(row["agent_b"])
        if a == b:
            continue
        y = outcome_to_y(row)
        ia, ib = idx[a], idx[b]
        if y is None:
            ties += 1
            continue
        if y == 1.0:
            wins[ia, ib] += 1
        else:
            wins[ib, ia] += 1
        comps[ia, ib] += 1
        comps[ib, ia] += 1
    return agents, wins, comps, ties


def cycle_audit(agents: List[str], wins: np.ndarray, comps: np.ndarray, min_count: int = 20, confidence: float = 0.95, max_cycles: int = 100) -> pd.DataFrame:
    if betainc is None:
        raise RuntimeError("scipy is required for cycle audit")
    n = len(agents)
    pr = np.full((n, n), 0.5, dtype=float)
    margin = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            N = comps[i, j]
            if N <= 0:
                continue
            a = 0.5 + wins[i, j]
            b = 0.5 + wins[j, i]
            p = 1.0 - float(betainc(a, b, 0.5))
            pr[i, j] = p
            pr[j, i] = 1.0 - p
            margin[i, j] = (wins[i, j] - wins[j, i]) / max(N, 1)
            margin[j, i] = -margin[i, j]
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j or comps[i, j] < min_count or pr[i, j] < confidence:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                # i beats j, j beats k, k beats i
                if comps[j, k] >= min_count and comps[k, i] >= min_count and pr[j, k] >= confidence and pr[k, i] >= confidence:
                    score = min(pr[i, j], pr[j, k], pr[k, i])
                    rows.append({
                        "agent_1": agents[i], "agent_2": agents[j], "agent_3": agents[k],
                        "edge_12_pr": pr[i, j], "edge_23_pr": pr[j, k], "edge_31_pr": pr[k, i],
                        "edge_12_count": int(comps[i, j]), "edge_23_count": int(comps[j, k]), "edge_31_count": int(comps[k, i]),
                        "cycle_confidence_min": score,
                        "cycle_margin_min_abs": float(min(abs(margin[i, j]), abs(margin[j, k]), abs(margin[k, i]))),
                    })
    cyc = pd.DataFrame(rows)
    if not cyc.empty:
        cyc = cyc.sort_values(["cycle_confidence_min", "cycle_margin_min_abs"], ascending=False).head(max_cycles)
    return cyc


def selection_diagnostics_from_scores(
    group_name: str,
    agents: List[str],
    wins: np.ndarray,
    comps: np.ndarray,
    method: str,
    scores: np.ndarray,
    seed: int = 0,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """Real-data diagnostics for a method's selected set.

    Real datasets usually do not expose a ground-truth tournament core. This
    diagnostic therefore asks a weaker, observable question: if a method selects
    a common-size top set, how often do outside agents empirically beat members
    of that selected set? Lower external_attack_rate and higher dominance_gap
    indicate that the selected set is less vulnerable to observed outside attacks.
    """
    n = len(agents)
    if k is None:
        k = max(1, min(n, int(math.ceil(0.1 * n))))
    pred = select_topk(scores, k=k, rng=np.random.default_rng(seed))
    selected = np.nonzero(pred)[0]
    outside = np.nonzero(1 - pred)[0]
    cross_total = 0.0
    selected_beats_outside = 0.0
    outside_beats_selected = 0.0
    pair_coverage = 0
    for i in selected:
        for j in outside:
            if comps[i, j] <= 0:
                continue
            cross_total += comps[i, j]
            selected_beats_outside += wins[i, j]
            outside_beats_selected += wins[j, i]
            pair_coverage += 1
    if cross_total > 0:
        external_attack_rate = outside_beats_selected / cross_total
        selected_dominance_rate = selected_beats_outside / cross_total
    else:
        external_attack_rate = float("nan")
        selected_dominance_rate = float("nan")
    coverage_den = max(1, len(selected) * len(outside))
    return {
        "group": group_name,
        "method": method,
        "k_selected": int(k),
        "n_agents": int(n),
        "n_cross_pairs_observed": int(pair_coverage),
        "cross_pair_coverage": float(pair_coverage / coverage_den),
        "external_attack_rate": float(external_attack_rate),
        "selected_dominance_rate": float(selected_dominance_rate),
        "dominance_gap": float(selected_dominance_rate - external_attack_rate) if np.isfinite(selected_dominance_rate) and np.isfinite(external_attack_rate) else float("nan"),
        "selected_agents": ";".join([agents[ii] for ii in selected]),
    }


def run_real_group(
    group_name: str,
    agents: List[str],
    wins: np.ndarray,
    comps: np.ndarray,
    out_dir: Path,
    methods: Sequence[str],
    tau: float,
    reachability: str,
    bootstrap: int = 200,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(agents)
    K = max(1, n - 1)
    score_rows = []
    diag_rows = []
    for method in methods:
        try:
            scores, meta = method_scores(method, wins, comps, tau=tau, K=K, reachability=reachability, seed=seed, posterior_samples=200)
        except Exception as exc:
            print(f"[warning] real method {method} failed for {group_name}: {exc}", file=sys.stderr)
            continue
        for i, agent in enumerate(agents):
            score_rows.append({
                "group": group_name,
                "agent": agent,
                "method": method,
                "score": float(scores[i]),
                "rank": int(pd.Series(-scores).rank(method="first").iloc[i]),
                "n_agents": n,
                "n_comparisons": int(comps.sum() // 2),
            })
        diag_rows.append(selection_diagnostics_from_scores(group_name, agents, wins, comps, method, scores, seed=seed))
    score_df = pd.DataFrame(score_rows)
    diag_df = pd.DataFrame(diag_rows)

    boot_rows = []
    if bootstrap > 0 and n >= 2 and comps.sum() > 0:
        rng = np.random.default_rng(seed)
        for b in range(bootstrap):
            bw, bc = bootstrap_counts_from_empirical(wins, comps, seed=int(rng.integers(0, 2**31 - 1)))
            for method in methods:
                try:
                    scores, _ = method_scores(method, bw, bc, tau=tau, K=K, reachability=reachability, seed=seed + b, posterior_samples=100)
                except Exception:
                    continue
                k = max(1, min(n, int(math.ceil(0.1 * n))))
                pred = select_topk(scores, k=k, rng=np.random.default_rng(seed + b))
                boot_rows.append({"group": group_name, "bootstrap": b, "method": method, "top1": agents[int(np.argmax(scores))], "pred_set": ";".join([agents[ii] for ii in np.nonzero(pred)[0]])})
    boot_df = pd.DataFrame(boot_rows)
    return score_df, boot_df, diag_df

def run_real_arena(args) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    df = load_pairwise_csv(args.input, args.agent_a_col, args.agent_b_col, args.winner_col, args.category_col)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    all_scores = []
    all_boot = []
    all_diag = []
    all_cycles = []
    groups = [("global", df)]
    if args.by_category:
        for cat, sub in df.groupby("category"):
            if cat == "global":
                continue
            groups.append((f"category={cat}", sub))
    for gname, sub in groups:
        agents, wins, comps, ties = counts_from_pairwise_df(sub)
        if len(agents) < 2:
            continue
        print(f"[arena] {gname}: {len(agents)} agents, {int(comps.sum()//2)} decisive comparisons, {ties} ties")
        scores, boot, diag = run_real_group(gname, agents, wins, comps, out_dir, methods, tau=args.tau, reachability=args.reachability, bootstrap=args.bootstrap, seed=args.seed)
        all_scores.append(scores)
        all_boot.append(boot)
        all_diag.append(diag)
        cycles = cycle_audit(agents, wins, comps, min_count=args.cycle_min_count, confidence=args.cycle_confidence, max_cycles=args.max_cycles)
        if not cycles.empty:
            cycles.insert(0, "group", gname)
            all_cycles.append(cycles)
    score_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    boot_df = pd.concat(all_boot, ignore_index=True) if all_boot else pd.DataFrame()
    cycles_df = pd.concat(all_cycles, ignore_index=True) if all_cycles else pd.DataFrame()
    diag_df = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()
    score_df.to_csv(out_dir / "real_arena_scores.csv", index=False)
    boot_df.to_csv(out_dir / "real_arena_bootstrap.csv", index=False)
    cycles_df.to_csv(out_dir / "real_arena_high_confidence_cycles.csv", index=False)
    diag_df.to_csv(out_dir / "real_selection_diagnostics.csv", index=False)
    write_real_summary(score_df, boot_df, cycles_df, out_dir / "real_arena_report.md", title="Arena / human-preference diagnostics")


# ---------------------------------------------------------------------------
# AgentBench-style execution log analysis
# ---------------------------------------------------------------------------


def load_agentbench_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Canonical score-log schema is environment, agent, task_id, score/success/status.
    # Some benchmark exports use domain/site/repository for environment and
    # instance_id for task_id; accept these aliases so the direct `scorelog`
    # command works on the templates and common leaderboard CSVs.
    alias_groups = {
        "environment": ["environment", "env", "domain", "site", "benchmark", "repository", "repo"],
        "agent": ["agent", "model", "model_name", "system", "solver"],
        "task_id": ["task_id", "task", "instance_id", "example_id", "problem_id", "sample_id"],
        "score": ["score", "reward", "accuracy", "metric", "value"],
        "success": ["success", "resolved", "passed", "solved"],
        "status": ["status", "outcome", "result"],
    }
    rename = {}
    lower_to_col = {c.lower(): c for c in df.columns}
    for canonical, aliases in alias_groups.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_to_col:
                rename[lower_to_col[alias.lower()]] = canonical
                break
    if rename:
        df = df.rename(columns=rename)

    required = ["environment", "agent", "task_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Score-log CSV missing {missing}. Expected canonical columns "
            "environment, agent, task_id plus one of score/success/status. "
            f"Available columns: {list(pd.read_csv(path, nrows=0).columns)}"
        )
    if "score" not in df.columns:
        if "success" in df.columns:
            df["score"] = df["success"].astype(float)
        elif "status" in df.columns:
            df["score"] = df["status"].astype(str).str.lower().isin(["completed", "success", "succeeded", "solved", "resolved", "passed"]).astype(float)
        else:
            raise ValueError("Score-log CSV must contain score, success, or status column")
    return df


def agentbench_to_pairwise(df: pd.DataFrame, environment: str) -> pd.DataFrame:
    sub = df[df["environment"].astype(str) == str(environment)].copy()
    rows = []
    for task_id, group in sub.groupby("task_id"):
        records = group[["agent", "score"]].dropna().drop_duplicates("agent").to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = str(records[i]["agent"]), str(records[j]["agent"])
                sa, sb = float(records[i]["score"]), float(records[j]["score"])
                if sa > sb:
                    w = a
                elif sb > sa:
                    w = b
                else:
                    w = "tie"
                rows.append({"agent_a": a, "agent_b": b, "winner": w, "category": environment})
    return pd.DataFrame(rows)


def run_agentbench(args) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    df = load_agentbench_csv(args.input)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    all_pairwise = []
    status_rows = []
    for env, sub in df.groupby("environment"):
        p = agentbench_to_pairwise(df, str(env))
        all_pairwise.append(p)
        if "status" in sub.columns:
            for agent, g in sub.groupby("agent"):
                counts = g["status"].astype(str).value_counts().to_dict()
                row = {"environment": env, "agent": agent, "n_episodes": len(g)}
                row.update({f"status_{k}": v for k, v in counts.items()})
                status_rows.append(row)
    pair_df = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame(columns=["agent_a", "agent_b", "winner", "category"])
    pair_df.to_csv(out_dir / "agentbench_pairwise_comparisons.csv", index=False)
    if status_rows:
        pd.DataFrame(status_rows).to_csv(out_dir / "agentbench_error_rates.csv", index=False)

    # Reuse real-arena engine.
    tmp_csv = out_dir / "agentbench_pairwise_tmp.csv"
    pair_df.to_csv(tmp_csv, index=False)
    class TmpArgs:
        pass
    t = TmpArgs()
    t.input = str(tmp_csv); t.out = str(out_dir); t.agent_a_col = "agent_a"; t.agent_b_col = "agent_b"; t.winner_col = "winner"; t.category_col = "category"
    t.methods = args.methods; t.tau = args.tau; t.reachability = args.reachability; t.bootstrap = args.bootstrap; t.seed = args.seed
    t.by_category = True; t.cycle_min_count = args.cycle_min_count; t.cycle_confidence = args.cycle_confidence; t.max_cycles = args.max_cycles
    run_real_arena(t)
    (out_dir / "agentbench_report.md").write_text((out_dir / "real_arena_report.md").read_text(encoding="utf-8"), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summaries, figures, and LaTeX tables
# ---------------------------------------------------------------------------


def group_summary(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(group_cols)):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["n_rows"] = len(g)
        for m in metric_cols:
            if m in g.columns:
                mean, half, se = ci95(g[m].values)
                row[f"{m}_mean"] = mean
                row[f"{m}_ci95"] = half
                row[f"{m}_se"] = se
        rows.append(row)
    return pd.DataFrame(rows)


def latex_float(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def write_latex_tables(out_dir: Path) -> None:
    tex_path = out_dir / "paper_tables.tex"
    parts = []
    rec = out_dir / "synthetic_recovery.csv"
    if rec.exists():
        df = pd.read_csv(rec)
        if not df.empty:
            # Moderate evidence by m.
            main_methods_for_table = ["ste_posterior_edge_uc", "ste_plugin_uc", "hard_uc", "btl", "winrate", "rank_centrality", "hodge", "copeland"]
            mod = df[(df["m_per_pair"] >= 5) & (df["method"].isin(main_methods_for_table))]
            summ = group_summary(mod, ["m_per_pair", "method"], ["topk_f1", "auprc", "auroc", "topk_fpr", "topk_fnr"])
            summ.to_csv(out_dir / "summary_by_m_method.csv", index=False)
            methods = ["ste_posterior_edge_uc", "ste_plugin_uc", "btl", "winrate", "rank_centrality", "hodge"]
            parts.append("% Main finite-sample recovery table generated by neurips_suite.py\n")
            parts.append("\\begin{table}[t]\n\\centering\n\\caption{Finite-sample planted-core recovery by comparisons per observed pair. Entries are mean $\\pm$ 95\\% CI over seeds and grid settings.}\n")
            parts.append("\\begin{tabular}{lrrrr}\n\\toprule\nMethod & $m=5$ & $m=10$ & $m=20$ & $m=50$ \\\\ \n\\midrule\n")
            for method in methods:
                vals = []
                for m in [5, 10, 20, 50]:
                    row = summ[(summ["method"] == method) & (summ["m_per_pair"] == m)]
                    if row.empty:
                        vals.append("--")
                    else:
                        vals.append(f"{latex_float(row.iloc[0]['topk_f1_mean'])} $\\pm$ {latex_float(row.iloc[0]['topk_f1_ci95'])}")
                parts.append(f"{method.replace('_', '-')} & " + " & ".join(vals) + " \\\\ \n")
            parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

            # Missingness table for m>=5.
            summ_miss = group_summary(mod, ["missing_rate", "method"], ["topk_f1", "auprc"])
            summ_miss.to_csv(out_dir / "summary_by_missing_method.csv", index=False)
            parts.append("% Missingness robustness table\n")
            parts.append("\\begin{table}[t]\n\\centering\n\\caption{Robustness to missing comparisons for $m\\ge5$. Entries are mean top-$|C|$ F1 $\\pm$ 95\\% CI.}\n")
            parts.append("\\begin{tabular}{lrrrr}\n\\toprule\nMethod & $\\mu=0$ & $\\mu=0.1$ & $\\mu=0.3$ & $\\mu=0.5$ \\\\ \n\\midrule\n")
            for method in methods[:4]:
                vals = []
                for mu in [0.0, 0.1, 0.3, 0.5]:
                    row = summ_miss[(summ_miss["method"] == method) & (np.isclose(summ_miss["missing_rate"], mu))]
                    if row.empty:
                        vals.append("--")
                    else:
                        vals.append(f"{latex_float(row.iloc[0]['topk_f1_mean'])} $\\pm$ {latex_float(row.iloc[0]['topk_f1_ci95'])}")
                parts.append(f"{method.replace('_', '-')} & " + " & ".join(vals) + " \\\\ \n")
            parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    boot = out_dir / "bootstrap_stability.csv"
    if boot.exists():
        dfb = pd.read_csv(boot)
        if not dfb.empty:
            summ = group_summary(dfb, ["method"], ["bootstrap_f1", "pairwise_jaccard", "bootstrap_auroc", "top1_entropy"])
            summ.to_csv(out_dir / "summary_bootstrap.csv", index=False)
            parts.append("% Bootstrap stability table\n")
            parts.append("\\begin{table}[t]\n\\centering\n\\caption{Bootstrap recovery stability on planted cyclic cores. Entries are mean $\\pm$ 95\\% CI.}\n")
            parts.append("\\begin{tabular}{lrrrr}\n\\toprule\nMethod & F1 to core & Pairwise Jaccard & AUROC & Top-1 entropy \\\\ \n\\midrule\n")
            for _, row in summ.iterrows():
                parts.append(
                    f"{str(row['method']).replace('_','-')} & "
                    f"{latex_float(row['bootstrap_f1_mean'])} $\\pm$ {latex_float(row['bootstrap_f1_ci95'])} & "
                    f"{latex_float(row['pairwise_jaccard_mean'])} $\\pm$ {latex_float(row['pairwise_jaccard_ci95'])} & "
                    f"{latex_float(row['bootstrap_auroc_mean'])} $\\pm$ {latex_float(row['bootstrap_auroc_ci95'])} & "
                    f"{latex_float(row['top1_entropy_mean'])} $\\pm$ {latex_float(row['top1_entropy_ci95'])} \\\\ \n"
                )
            parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    ab = out_dir / "synthetic_ablation.csv"
    if ab.exists():
        dfa = pd.read_csv(ab)
        if not dfa.empty:
            # Primary ablation table: edge estimator x reachability, m>=10
            sub = dfa[dfa["m_per_pair"] >= 10]
            if sub.empty:
                sub = dfa
            summ = group_summary(sub, ["edge_estimator", "reachability"], ["topk_f1", "auprc", "runtime_sec"])
            summ.to_csv(out_dir / "summary_ablation_edge_reachability.csv", index=False)
            parts.append("% Edge/reachability ablation table\n")
            parts.append("\\begin{table}[t]\n\\centering\n\\caption{Ablation over edge estimator and reachability operator for $m\\ge10$.}\n")
            parts.append("\\begin{tabular}{llrr}\n\\toprule\nEdge estimator & Reachability & F1 & AUPRC \\\\ \n\\midrule\n")
            for _, row in summ.sort_values("topk_f1_mean", ascending=False).head(12).iterrows():
                parts.append(f"{row['edge_estimator']} & {row['reachability']} & {latex_float(row['topk_f1_mean'])} $\\pm$ {latex_float(row['topk_f1_ci95'])} & {latex_float(row['auprc_mean'])} $\\pm$ {latex_float(row['auprc_ci95'])} \\\\ \n")
            parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    tex_path.write_text("".join(parts), encoding="utf-8")


def write_figures(out_dir: Path) -> None:
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    rec = out_dir / "synthetic_recovery.csv"
    if rec.exists():
        df = pd.read_csv(rec)
        if not df.empty:
            main_methods = ["ste_posterior_edge_uc", "ste_plugin_uc", "btl", "winrate", "rank_centrality", "hodge"]
            sub = df[(df["method"].isin(main_methods))]
            summ = group_summary(sub, ["m_per_pair", "method"], ["topk_f1", "auprc"])
            plt.figure(figsize=(7, 4.5))
            for method in main_methods:
                g = summ[summ["method"] == method].sort_values("m_per_pair")
                if g.empty: continue
                plt.errorbar(g["m_per_pair"], g["topk_f1_mean"], yerr=g["topk_f1_ci95"], marker="o", label=method.replace("_", "-"))
            plt.xscale("log")
            plt.xlabel("comparisons per observed pair (m)")
            plt.ylabel("tie-safe top-|C| F1")
            plt.title("Planted-core recovery")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "recovery_by_m.png", dpi=200)
            plt.close()

            sub2 = df[(df["method"].isin(main_methods)) & (df["m_per_pair"] >= 5)]
            summ2 = group_summary(sub2, ["missing_rate", "method"], ["topk_f1"])
            plt.figure(figsize=(7, 4.5))
            for method in main_methods[:4]:
                g = summ2[summ2["method"] == method].sort_values("missing_rate")
                if g.empty: continue
                plt.errorbar(g["missing_rate"], g["topk_f1_mean"], yerr=g["topk_f1_ci95"], marker="o", label=method.replace("_", "-"))
            plt.xlabel("missing-pair rate")
            plt.ylabel("tie-safe top-|C| F1, m>=5")
            plt.title("Robustness to missing comparisons")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "missingness_robustness.png", dpi=200)
            plt.close()

    ab = out_dir / "synthetic_ablation.csv"
    if ab.exists():
        dfa = pd.read_csv(ab)
        if not dfa.empty:
            sub = dfa[dfa["m_per_pair"] >= 10]
            if sub.empty:
                sub = dfa
            summ = group_summary(sub, ["edge_estimator", "reachability"], ["topk_f1"])
            if summ.empty or "topk_f1_mean" not in summ.columns:
                return
            top = summ.sort_values("topk_f1_mean", ascending=False).head(12)
            labels = [f"{r.edge_estimator}\n{r.reachability}" for r in top.itertuples()]
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(top)), top["topk_f1_mean"], yerr=top["topk_f1_ci95"])
            plt.xticks(range(len(top)), labels, rotation=45, ha="right", fontsize=8)
            plt.ylabel("top-|C| F1")
            plt.title("Estimator/reachability ablation")
            plt.tight_layout()
            plt.savefig(fig_dir / "ablation_edge_reachability.png", dpi=200)
            plt.close()

    boot = out_dir / "bootstrap_stability.csv"
    if boot.exists():
        dfb = pd.read_csv(boot)
        if not dfb.empty:
            summ = group_summary(dfb, ["method"], ["bootstrap_f1"])
            summ = summ.sort_values("bootstrap_f1_mean", ascending=False)
            plt.figure(figsize=(7, 4.5))
            plt.bar(range(len(summ)), summ["bootstrap_f1_mean"], yerr=summ["bootstrap_f1_ci95"])
            plt.xticks(range(len(summ)), [str(x).replace("_", "-") for x in summ["method"]], rotation=30, ha="right")
            plt.ylabel("bootstrap F1 to true core")
            plt.title("Bootstrap recovery stability")
            plt.tight_layout()
            plt.savefig(fig_dir / "bootstrap_f1.png", dpi=200)
            plt.close()


def _mean_pairwise_jaccard_from_sets(set_strings: Sequence[str], max_sets: int = 300) -> float:
    sets = []
    for s in list(set_strings)[:max_sets]:
        if not isinstance(s, str) or not s:
            continue
        sets.append(set([x for x in s.split(";") if x]))
    if len(sets) < 2:
        return float("nan")
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            vals.append(len(sets[i] & sets[j]) / len(union))
    return float(np.mean(vals)) if vals else float("nan")


def write_real_summary(score_df: pd.DataFrame, boot_df: pd.DataFrame, cycles_df: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}\n\n"]
    if not score_df.empty:
        lines.append("## Top scores by group\n\n")
        for group, g in score_df.groupby("group"):
            lines.append(f"### {group}\n\n")
            pivot = g.pivot_table(index="agent", columns="method", values="score", aggfunc="first")
            sort_col = "ste_posterior_edge_uc" if "ste_posterior_edge_uc" in pivot.columns else ("ste_posterior_edge" if "ste_posterior_edge" in pivot.columns else None)
            if sort_col is not None:
                pivot = pivot.sort_values(sort_col, ascending=False)
            lines.append(df_to_markdown(pivot.head(20)))
            lines.append("\n\n")
    diag_path = path.parent / "real_selection_diagnostics.csv"
    if diag_path.exists():
        diag_df = pd.read_csv(diag_path)
        if not diag_df.empty:
            lines.append("## Selected-set dominance/error diagnostics\n\n")
            cols = [c for c in ["group", "method", "k_selected", "cross_pair_coverage", "external_attack_rate", "selected_dominance_rate", "dominance_gap"] if c in diag_df.columns]
            lines.append(df_to_markdown(diag_df[cols].sort_values(["group", "dominance_gap"], ascending=[True, False]).head(60), index=False))
            lines.append("\n\n")
    if not cycles_df.empty:
        lines.append("## High-confidence 3-cycles\n\n")
        lines.append(df_to_markdown(cycles_df.head(30), index=False))
        lines.append("\n\n")
    else:
        lines.append("No high-confidence 3-cycles found under the configured count/confidence thresholds.\n\n")
    if not boot_df.empty:
        lines.append("## Bootstrap top-1 and selected-set stability\n\n")
        rows = []
        for (group, method), g in boot_df.groupby(["group", "method"]):
            top_counts = g["top1"].value_counts(normalize=True)
            rows.append({
                "group": group,
                "method": method,
                "top1_entropy": float(-(top_counts * np.log2(top_counts)).sum()),
                "modal_top1": top_counts.index[0],
                "modal_freq": float(top_counts.iloc[0]),
                "mean_pairwise_set_jaccard": _mean_pairwise_jaccard_from_sets(g.get("pred_set", [])),
            })
        lines.append(df_to_markdown(pd.DataFrame(rows).sort_values(["group", "mean_pairwise_set_jaccard"], ascending=[True, False]), index=False))
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_summary_report(out_dir: Path) -> None:
    lines = ["# STE NeurIPS Experiment Suite Report\n\n"]
    lines.append("This report is generated automatically from the run artifacts. The final manuscript should quote only completed runs and should not report unrun planned experiments as results.\n\n")
    oracle = out_dir / "oracle_sanity.csv"
    if oracle.exists():
        df = pd.read_csv(oracle)
        lines.append("## Oracle sanity checks\n\n")
        lines.append(df_to_markdown(df, index=False))
        lines.append("\n\n")
    rec = out_dir / "synthetic_recovery.csv"
    if rec.exists():
        df = pd.read_csv(rec)
        lines.append("## Synthetic recovery summary\n\n")
        sub = df[df["m_per_pair"] >= 5]
        summ = group_summary(sub, ["method"], ["topk_f1", "auprc", "auroc", "topk_fpr", "topk_fnr", "runtime_sec"])
        lines.append(df_to_markdown(summ.sort_values("topk_f1_mean", ascending=False), index=False))
        lines.append("\n\n")
    ab = out_dir / "synthetic_ablation.csv"
    if ab.exists():
        dfa = pd.read_csv(ab)
        lines.append("## Top ablations\n\n")
        sub_ab = dfa[dfa["m_per_pair"] >= 10]
        if sub_ab.empty:
            sub_ab = dfa
        summ = group_summary(sub_ab, ["edge_estimator", "reachability"], ["topk_f1", "auprc", "runtime_sec"])
        lines.append(df_to_markdown(summ.sort_values("topk_f1_mean", ascending=False).head(20), index=False))
        lines.append("\n\n")
    boot = out_dir / "bootstrap_stability.csv"
    if boot.exists():
        dfb = pd.read_csv(boot)
        lines.append("## Bootstrap recovery/stability\n\n")
        summ = group_summary(dfb, ["method"], ["bootstrap_f1", "pairwise_jaccard", "bootstrap_auroc", "top1_entropy"])
        lines.append(df_to_markdown(summ.sort_values("bootstrap_f1_mean", ascending=False), index=False))
        lines.append("\n\n")
    scaling = out_dir / "runtime_scaling.csv"
    if scaling.exists():
        dfs = pd.read_csv(scaling)
        lines.append("## Runtime scaling\n\n")
        summ = group_summary(dfs, ["n", "reachability", "K"], ["runtime_sec"])
        lines.append(df_to_markdown(summ.head(100), index=False))
        lines.append("\n\n")
    ctrl = out_dir / "negative_controls.csv"
    if ctrl.exists():
        dfc = pd.read_csv(ctrl)
        lines.append("## Negative and sanity controls\n\n")
        if not dfc.empty:
            summ = group_summary(dfc, ["control", "method"], ["topk_f1", "auroc", "ece"])
            lines.append(df_to_markdown(summ.sort_values(["control", "topk_f1_mean"], ascending=[True, False]), index=False))
            lines.append("\n\n")
    th = out_dir / "synthetic_threshold_sensitivity.csv"
    if th.exists():
        dft = pd.read_csv(th)
        lines.append("## Posterior-edge threshold sensitivity\n\n")
        if not dft.empty:
            summ = group_summary(dft, ["solution", "threshold"], ["threshold_f1", "reported_core_size"])
            lines.append(df_to_markdown(summ, index=False))
            lines.append("\n\n")
    pair_rel = out_dir / "synthetic_pairwise_reliability.csv"
    if pair_rel.exists():
        dfr = pd.read_csv(pair_rel)
        lines.append("## Calibration/reliability diagnostics\n\n")
        if not dfr.empty:
            lines.append("Pairwise probability reliability, averaged over diagnostic seeds:\n\n")
            lines.append(df_to_markdown(group_summary(dfr, ["bin"], ["mean_estimated_prob", "mean_true_prob", "abs_error"]), index=False))
            lines.append("\n\n")
    (out_dir / "summary_report.md").write_text("".join(lines), encoding="utf-8")



# ---------------------------------------------------------------------------
# Additional reviewer-facing diagnostics: reliability, thresholds, controls
# ---------------------------------------------------------------------------


def reliability_table(scores: np.ndarray, true: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Bin scores against binary labels for membership reliability diagnostics."""
    scores = normalize_scores(np.asarray(scores, dtype=float))
    true = np.asarray(true, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (scores >= lo) & (scores < hi if b < bins - 1 else scores <= hi)
        if not np.any(mask):
            continue
        rows.append({
            "bin": b,
            "lo": lo,
            "hi": hi,
            "count": int(mask.sum()),
            "mean_score": float(scores[mask].mean()),
            "empirical_membership": float(true[mask].mean()),
            "abs_error": float(abs(scores[mask].mean() - true[mask].mean())),
        })
    return pd.DataFrame(rows)


def pairwise_probability_reliability(P_true: np.ndarray, wins: np.ndarray, comps: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability of empirical/posterior mean pairwise probabilities vs simulator truth."""
    P_hat = smoothed_mean_P(wins, comps, prior=0.5)
    rows_raw = []
    n = P_true.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i, j] <= 0:
                continue
            rows_raw.append((float(P_hat[i, j]), float(P_true[i, j]), int(comps[i, j])))
            rows_raw.append((float(P_hat[j, i]), float(P_true[j, i]), int(comps[j, i])))
    if not rows_raw:
        return pd.DataFrame()
    arr = np.asarray(rows_raw, dtype=float)
    est, truep = arr[:, 0], arr[:, 1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (est >= lo) & (est < hi if b < bins - 1 else est <= hi)
        if not np.any(mask):
            continue
        rows.append({
            "bin": b,
            "lo": lo,
            "hi": hi,
            "count": int(mask.sum()),
            "mean_estimated_prob": float(est[mask].mean()),
            "mean_true_prob": float(truep[mask].mean()),
            "abs_error": float(abs(est[mask].mean() - truep[mask].mean())),
        })
    return pd.DataFrame(rows)


def edge_margin_records(P_true: np.ndarray, wins: np.ndarray, comps: np.ndarray, seed: int, n: int, core_size: int, m: int, missing: float) -> pd.DataFrame:
    P_hat = smoothed_mean_P(wins, comps, prior=0.5)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            observed = bool(comps[i, j] > 0)
            rows.append({
                "seed": seed,
                "n": n,
                "core_size": core_size,
                "m_per_pair": m,
                "missing_rate": missing,
                "observed": observed,
                "true_abs_margin": float(abs(P_true[i, j] - 0.5)),
                "empirical_abs_margin": float(abs(P_hat[i, j] - 0.5)) if observed else float("nan"),
                "comparison_count": int(comps[i, j]),
            })
    return pd.DataFrame(rows)


def run_threshold_sensitivity(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    """Threshold posterior-edge UC/TC scores and report core-size sensitivity."""
    n = int(get_cfg(cfg, "diagnostic_n", get_cfg(cfg, "bootstrap_n", 25)))
    s = int(get_cfg(cfg, "diagnostic_core_size", get_cfg(cfg, "bootstrap_core_size", 5)))
    m = int(get_cfg(cfg, "diagnostic_m", get_cfg(cfg, "bootstrap_m", 10)))
    missing = float(get_cfg(cfg, "diagnostic_missing", get_cfg(cfg, "bootstrap_missing", 0.1)))
    label_noise = float(get_cfg(cfg, "diagnostic_label_noise", get_cfg(cfg, "label_noise", 0.02)))
    seeds = int(get_cfg(cfg, "diagnostic_seeds", min(20, int(get_cfg(cfg, "seeds", 40)))))
    posterior_samples = int(get_cfg(cfg, "posterior_samples", 200))
    thresholds = parse_csv_floats(get_cfg(cfg, "threshold_values", "0.25,0.35,0.45,0.5,0.55,0.65,0.75"))
    rows = []
    for seed in range(seeds):
        trial_seed = 203_000_000 + seed
        T = make_planted_core_tournament(n, s, seed=trial_seed)
        wins, comps = sample_counts(T.P, m_per_pair=m, missing_rate=missing, label_noise=label_noise, seed=trial_seed + 7)
        for sol, target in [("TC", T.true_tc), ("UC", T.true_uc)]:
            scores = posterior_membership_scores(wins, comps, solution=sol.lower(), samples=posterior_samples, seed=trial_seed + 11)
            for th in thresholds:
                pred = (scores >= th).astype(int)
                f1, j, fpr, fnr, tp = f1_jaccard(pred, target)
                rows.append({
                    "seed": seed,
                    "n": n,
                    "core_size": int(target.sum()),
                    "m_per_pair": m,
                    "missing_rate": missing,
                    "solution": sol,
                    "threshold": th,
                    "reported_core_size": int(pred.sum()),
                    "threshold_f1": f1,
                    "threshold_jaccard": j,
                    "threshold_fpr": fpr,
                    "threshold_fnr": fnr,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic_threshold_sensitivity.csv", index=False)
    return df


def run_reliability_diagnostics(cfg: Dict, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise probability, membership calibration, and edge-margin diagnostics."""
    n = int(get_cfg(cfg, "diagnostic_n", get_cfg(cfg, "bootstrap_n", 25)))
    s = int(get_cfg(cfg, "diagnostic_core_size", get_cfg(cfg, "bootstrap_core_size", 5)))
    m = int(get_cfg(cfg, "diagnostic_m", get_cfg(cfg, "bootstrap_m", 10)))
    missing = float(get_cfg(cfg, "diagnostic_missing", get_cfg(cfg, "bootstrap_missing", 0.1)))
    label_noise = float(get_cfg(cfg, "diagnostic_label_noise", get_cfg(cfg, "label_noise", 0.02)))
    seeds = int(get_cfg(cfg, "diagnostic_seeds", min(20, int(get_cfg(cfg, "seeds", 40)))))
    bins = int(get_cfg(cfg, "reliability_bins", 10))
    posterior_samples = int(get_cfg(cfg, "posterior_samples", 200))
    pair_rows, mem_rows, margin_rows = [], [], []
    for seed in range(seeds):
        trial_seed = 207_000_000 + seed
        T = make_planted_core_tournament(n, s, seed=trial_seed)
        wins, comps = sample_counts(T.P, m_per_pair=m, missing_rate=missing, label_noise=label_noise, seed=trial_seed + 13)
        pr = pairwise_probability_reliability(T.P, wins, comps, bins=bins)
        if not pr.empty:
            pr.insert(0, "seed", seed)
            pr.insert(1, "m_per_pair", m)
            pair_rows.append(pr)
        mr = edge_margin_records(T.P, wins, comps, seed=seed, n=n, core_size=s, m=m, missing=missing)
        margin_rows.append(mr)
        for method in ["ste_posterior_edge_uc", "ste_plugin_uc", "winrate", "btl", "rank_centrality", "hodge"]:
            try:
                scores, _ = method_scores(method, wins, comps, tau=float(get_cfg(cfg, "tau", 0.035)), K=n-1, reachability=str(get_cfg(cfg, "reachability", "max_min")), seed=trial_seed, posterior_samples=posterior_samples)
            except Exception:
                continue
            target = target_for_method(method, T.true_tc, T.true_uc)
            rt = reliability_table(scores, target, bins=bins)
            if not rt.empty:
                rt.insert(0, "seed", seed)
                rt.insert(1, "method", method)
                rt.insert(2, "solution", "UC" if target is T.true_uc else "TC")
                mem_rows.append(rt)
    pair_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    mem_df = pd.concat(mem_rows, ignore_index=True) if mem_rows else pd.DataFrame()
    margin_df = pd.concat(margin_rows, ignore_index=True) if margin_rows else pd.DataFrame()
    pair_df.to_csv(out_dir / "synthetic_pairwise_reliability.csv", index=False)
    mem_df.to_csv(out_dir / "synthetic_membership_reliability.csv", index=False)
    margin_df.to_csv(out_dir / "synthetic_edge_margins.csv", index=False)
    return pair_df, mem_df, margin_df


def run_negative_controls(cfg: Dict, out_dir: Path) -> pd.DataFrame:
    """Negative and sanity-control experiments for reviewer concerns."""
    n = int(get_cfg(cfg, "control_n", 25))
    s = int(get_cfg(cfg, "control_core_size", 5))
    m = int(get_cfg(cfg, "control_m", 10))
    missing = float(get_cfg(cfg, "control_missing", 0.1))
    label_noise = float(get_cfg(cfg, "control_label_noise", get_cfg(cfg, "label_noise", 0.02)))
    seeds = int(get_cfg(cfg, "control_seeds", min(30, int(get_cfg(cfg, "seeds", 40)))))
    posterior_samples = int(get_cfg(cfg, "posterior_samples", 200))
    tau = float(get_cfg(cfg, "tau", 0.035))
    methods = get_cfg(cfg, "control_methods", ["ste_posterior_edge_uc", "ste_plugin_uc", "hard_uc", "winrate", "btl", "rank_centrality", "hodge", "copeland"])
    if isinstance(methods, str):
        methods = [x.strip() for x in methods.split(",") if x.strip()]
    rows = []
    for seed in range(seeds):
        # Positive sanity: transitive singleton Condorcet case.
        for control_name, mode in [("transitive_condorcet", "transitive"), ("cyclic_core", "cyclic_core")]:
            trial_seed = 211_000_000 + 1000 * seed + (0 if mode == "transitive" else 1)
            T = make_planted_core_tournament(n, 1 if mode == "transitive" else s, seed=trial_seed, mode=mode)
            wins, comps = sample_counts(T.P, m_per_pair=m, missing_rate=missing, label_noise=label_noise, seed=trial_seed + 17)
            for method in methods:
                try:
                    scores, meta = method_scores(method, wins, comps, tau=tau, K=n-1, seed=trial_seed, posterior_samples=posterior_samples)
                except Exception:
                    continue
                target = target_for_method(method, T.true_tc, T.true_uc)
                metrics = tie_randomized_metrics(scores, target, k=int(target.sum()), seed=trial_seed, repeats=25)
                rows.append({"control": control_name, "seed": seed, "n": n, "core_size": int(target.sum()), "method": method, **metrics})
        # Negative control: pairwise labels contain no relation to the planted core.
        trial_seed = 211_000_000 + 1000 * seed + 2
        T = make_planted_core_tournament(n, s, seed=trial_seed, mode="cyclic_core")
        P_null = np.full((n, n), 0.5, dtype=float); np.fill_diagonal(P_null, 0.5)
        wins, comps = sample_counts(P_null, m_per_pair=m, missing_rate=missing, label_noise=0.0, seed=trial_seed + 19)
        for method in methods:
            try:
                scores, meta = method_scores(method, wins, comps, tau=tau, K=n-1, seed=trial_seed, posterior_samples=posterior_samples)
            except Exception:
                continue
            target = target_for_method(method, T.true_tc, T.true_uc)
            metrics = tie_randomized_metrics(scores, target, k=int(target.sum()), seed=trial_seed, repeats=25)
            rows.append({"control": "random_labels_against_planted_core", "seed": seed, "n": n, "core_size": int(target.sum()), "method": method, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "negative_controls.csv", index=False)
    return df


def write_extra_figures(out_dir: Path) -> None:
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    th = out_dir / "synthetic_threshold_sensitivity.csv"
    if th.exists():
        df = pd.read_csv(th)
        if not df.empty:
            summ = group_summary(df, ["solution", "threshold"], ["threshold_f1", "reported_core_size"])
            plt.figure(figsize=(7, 4.5))
            for sol in sorted(summ["solution"].unique()):
                g = summ[summ["solution"] == sol].sort_values("threshold")
                plt.errorbar(g["threshold"], g["threshold_f1_mean"], yerr=g["threshold_f1_ci95"], marker="o", label=sol)
            plt.xlabel("membership reporting threshold")
            plt.ylabel("F1 to planted core")
            plt.title("Threshold sensitivity of posterior-edge STE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "threshold_sensitivity.png", dpi=200)
            plt.close()
    pr = out_dir / "synthetic_pairwise_reliability.csv"
    if pr.exists():
        df = pd.read_csv(pr)
        if not df.empty:
            summ = group_summary(df, ["bin"], ["mean_estimated_prob", "mean_true_prob"])
            plt.figure(figsize=(5.5, 5))
            plt.plot([0,1],[0,1], linestyle="--")
            plt.scatter(summ["mean_estimated_prob_mean"], summ["mean_true_prob_mean"])
            plt.xlabel("binned posterior-mean pairwise estimate")
            plt.ylabel("mean true simulator probability")
            plt.title("Pairwise reliability diagnostic")
            plt.tight_layout()
            plt.savefig(fig_dir / "pairwise_reliability.png", dpi=200)
            plt.close()
    mem = out_dir / "synthetic_membership_reliability.csv"
    if mem.exists():
        df = pd.read_csv(mem)
        if not df.empty:
            focus = ["ste_posterior_edge_uc", "ste_plugin_uc", "btl", "winrate"]
            sub = df[df["method"].isin(focus)]
            summ = group_summary(sub, ["method", "bin"], ["mean_score", "empirical_membership"])
            plt.figure(figsize=(6.5, 5))
            plt.plot([0,1],[0,1], linestyle="--")
            for method in focus:
                g = summ[summ["method"] == method].sort_values("mean_score_mean")
                if g.empty: continue
                plt.plot(g["mean_score_mean"], g["empirical_membership_mean"], marker="o", label=method.replace("_", "-"))
            plt.xlabel("binned normalized score")
            plt.ylabel("empirical core membership")
            plt.title("Membership reliability diagnostic")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "membership_reliability.png", dpi=200)
            plt.close()
    margins = out_dir / "synthetic_edge_margins.csv"
    if margins.exists():
        df = pd.read_csv(margins)
        if not df.empty:
            plt.figure(figsize=(7, 4.5))
            vals = df[df["observed"] == True]["empirical_abs_margin"].dropna()
            plt.hist(vals, bins=20)
            plt.xlabel("empirical absolute edge margin |P_hat - 0.5|")
            plt.ylabel("pair count")
            plt.title("Synthetic edge-margin distribution")
            plt.tight_layout()
            plt.savefig(fig_dir / "edge_margin_histogram.png", dpi=200)
            plt.close()
    ctrl = out_dir / "negative_controls.csv"
    if ctrl.exists():
        df = pd.read_csv(ctrl)
        if not df.empty:
            methods = ["ste_posterior_edge_uc", "ste_plugin_uc", "btl", "winrate", "copeland"]
            sub = df[df["method"].isin(methods)]
            summ = group_summary(sub, ["control", "method"], ["topk_f1"])
            controls = list(summ["control"].unique())
            for control in controls:
                g = summ[summ["control"] == control].sort_values("topk_f1_mean", ascending=False)
                plt.figure(figsize=(7, 4.5))
                plt.bar(range(len(g)), g["topk_f1_mean"], yerr=g["topk_f1_ci95"])
                plt.xticks(range(len(g)), [m.replace("_", "-") for m in g["method"]], rotation=30, ha="right")
                plt.ylabel("top-|C| F1")
                plt.title(f"Control: {control}")
                plt.tight_layout()
                plt.savefig(fig_dir / f"control_{control}.png", dpi=200)
                plt.close()


def append_extra_latex_tables(out_dir: Path) -> None:
    tex_path = out_dir / "paper_tables.tex"
    parts = []
    th = out_dir / "synthetic_threshold_sensitivity.csv"
    if th.exists():
        df = pd.read_csv(th)
        if not df.empty:
            summ = group_summary(df[df["solution"] == "UC"], ["threshold"], ["threshold_f1", "reported_core_size"])
            parts.append("% Threshold sensitivity table\n")
            parts.append("\\begin{table}[t]\n\\centering\n\\caption{Posterior-edge UC threshold sensitivity. The main synthetic comparison avoids thresholding by selecting top-$|C|$ sets; this table reports the effect of a discrete reporting threshold.}\n")
            parts.append("\\begin{tabular}{lrr}\n\\toprule\nThreshold & F1 & Reported size \\\\ \n\\midrule\n")
            for _, row in summ.sort_values("threshold").iterrows():
                parts.append(f"{latex_float(row['threshold'],2)} & {latex_float(row['threshold_f1_mean'])} $\\pm$ {latex_float(row['threshold_f1_ci95'])} & {latex_float(row['reported_core_size_mean'])} $\\pm$ {latex_float(row['reported_core_size_ci95'])} \\\\ \n")
            parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    ctrl = out_dir / "negative_controls.csv"
    if ctrl.exists():
        df = pd.read_csv(ctrl)
        if not df.empty:
            methods = ["ste_posterior_edge_uc", "ste_plugin_uc", "btl", "winrate", "copeland"]
            sub = df[df["method"].isin(methods)]
            summ = group_summary(sub, ["control", "method"], ["topk_f1", "auroc"])
            parts.append("% Negative/sanity controls table\n")
            parts.append("\\begin{table}[t]\n\\centering\n\\caption{Sanity and negative controls. The random-label control should not support strong planted-core recovery.}\n")
            parts.append("\\begin{tabular}{llrr}\n\\toprule\nControl & Method & F1 & AUROC \\\\ \n\\midrule\n")
            for _, row in summ.sort_values(["control", "topk_f1_mean"], ascending=[True, False]).iterrows():
                parts.append(f"{row['control'].replace('_','-')} & {str(row['method']).replace('_','-')} & {latex_float(row['topk_f1_mean'])} $\\pm$ {latex_float(row['topk_f1_ci95'])} & {latex_float(row['auroc_mean'])} $\\pm$ {latex_float(row['auroc_ci95'])} \\\\ \n")
            parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    if parts:
        with open(tex_path, "a", encoding="utf-8") as f:
            f.write("".join(parts))

# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------


def run_synthetic(args) -> None:
    cfg = read_yaml(args.config)
    if args.out:
        cfg["out"] = args.out
    out_dir = Path(get_cfg(cfg, "out", "outputs/neurips_synthetic"))
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    run_oracle_sanity(out_dir)
    if not args.only_summary:
        run_synthetic_grid(cfg, out_dir)
        if bool(get_cfg(cfg, "run_ablation", True)):
            run_ablation_grid(cfg, out_dir)
        if bool(get_cfg(cfg, "run_bootstrap", True)):
            run_bootstrap_stability(cfg, out_dir)
        if bool(get_cfg(cfg, "run_scaling", True)):
            run_scaling(cfg, out_dir)
        if bool(get_cfg(cfg, "run_extra_diagnostics", True)):
            run_threshold_sensitivity(cfg, out_dir)
            run_reliability_diagnostics(cfg, out_dir)
            run_negative_controls(cfg, out_dir)
    write_latex_tables(out_dir)
    append_extra_latex_tables(out_dir)
    write_figures(out_dir)
    write_extra_figures(out_dir)
    write_summary_report(out_dir)
    print(f"[done] Synthetic suite wrote {out_dir}")



# ---------------------------------------------------------------------------
# Multi-dataset real benchmark manifest runner
# ---------------------------------------------------------------------------


def _copy_and_rename_csv(input_path: str, output_path: Path, rename_map: Mapping[str, str]) -> str:
    df = pd.read_csv(input_path)
    # rename_map maps standard_col -> source_col
    actual = {source: standard for standard, source in rename_map.items() if source and source in df.columns}
    df = df.rename(columns=actual)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return str(output_path)


def run_scorelog(args) -> None:
    """Generic score-log runner: environment, agent, task_id, score/success/status."""
    run_agentbench(args)


def run_real_suite(args) -> None:
    """Run many real datasets from a YAML manifest.

    Manifest format:
        methods: [ste_posterior_edge_uc, btl, ...]
        output_root: outputs/real_suite
        datasets:
          - name: chatbot_arena_33k
            type: pairwise
            path: data/chatbot_arena_33k.csv
            columns: {agent_a: model_a, agent_b: model_b, winner: winner, category: category}
          - name: osworld
            type: scorelog
            path: data/osworld_scores.csv
            columns: {environment: environment, agent: agent, task_id: task_id, score: score}
    """
    manifest = read_yaml(args.manifest)
    out_root = Path(args.out or manifest.get("output_root", "outputs/real_suite"))
    ensure_dir(out_root)
    methods = manifest.get("methods", args.methods)
    if isinstance(methods, list):
        methods = ",".join(methods)
    datasets = manifest.get("datasets", [])
    enabled_datasets = [ds for ds in datasets if ds and ds.get("enabled", True)]
    if not enabled_datasets:
        raise RuntimeError(
            "Real-suite manifest contains no enabled datasets. Run scripts/download_real_datasets.py successfully, "
            "or edit the manifest and set enabled: true for at least one dataset with an existing CSV path."
        )
    summary_rows = []
    for ds in enabled_datasets:
        name = str(ds.get("name", Path(str(ds.get("path", "dataset"))).stem)).replace("/", "_")
        dtype = str(ds.get("type", "pairwise")).lower()
        path_in = ds.get("path")
        if not path_in:
            print(f"[real-suite] skipping {name}: missing path", file=sys.stderr)
            continue
        out_dir = out_root / name
        ensure_dir(out_dir)
        cols = ds.get("columns", {}) or {}
        print(f"[real-suite] running {name} ({dtype}) -> {out_dir}")
        try:
            class TmpArgs:
                pass
            t = TmpArgs()
            t.out = str(out_dir)
            t.methods = methods
            t.tau = float(ds.get("tau", manifest.get("tau", 0.035)))
            t.reachability = str(ds.get("reachability", manifest.get("reachability", "max_min")))
            t.bootstrap = int(ds.get("bootstrap", manifest.get("bootstrap", args.bootstrap)))
            t.seed = int(ds.get("seed", manifest.get("seed", args.seed)))
            t.cycle_min_count = int(ds.get("cycle_min_count", manifest.get("cycle_min_count", 20 if dtype == "pairwise" else 5)))
            t.cycle_confidence = float(ds.get("cycle_confidence", manifest.get("cycle_confidence", 0.95 if dtype == "pairwise" else 0.90)))
            t.max_cycles = int(ds.get("max_cycles", manifest.get("max_cycles", 100)))
            if dtype in {"pairwise", "arena", "human_preference", "llm_judge_pairwise"}:
                rename_map = {
                    "model_a": cols.get("agent_a", cols.get("model_a", "model_a")),
                    "model_b": cols.get("agent_b", cols.get("model_b", "model_b")),
                    "winner": cols.get("winner", "winner"),
                    "category": cols.get("category", "category"),
                }
                std_csv = out_dir / "standard_pairwise.csv"
                t.input = _copy_and_rename_csv(str(path_in), std_csv, rename_map)
                t.agent_a_col = "model_a"
                t.agent_b_col = "model_b"
                t.winner_col = "winner"
                t.category_col = "category"
                t.by_category = bool(ds.get("by_category", True))
                run_real_arena(t)
                produced = out_dir / "real_arena_scores.csv"
            elif dtype in {"scorelog", "task_scores", "agentbench", "webarena", "osworld", "swebench", "gaia"}:
                rename_map = {
                    "environment": cols.get("environment", "environment"),
                    "agent": cols.get("agent", "agent"),
                    "task_id": cols.get("task_id", "task_id"),
                    "score": cols.get("score", "score"),
                    "success": cols.get("success", "success"),
                    "status": cols.get("status", "status"),
                }
                std_csv = out_dir / "standard_scorelog.csv"
                t.input = _copy_and_rename_csv(str(path_in), std_csv, rename_map)
                run_agentbench(t)
                produced = out_dir / "real_arena_scores.csv"
            else:
                raise ValueError(f"Unknown dataset type {dtype!r}")
            summary_rows.append({"dataset": name, "type": dtype, "status": "ok", "out_dir": str(out_dir), "scores_file": str(produced)})
        except Exception as exc:
            print(f"[real-suite] {name} failed: {exc}", file=sys.stderr)
            summary_rows.append({"dataset": name, "type": dtype, "status": f"failed: {exc}", "out_dir": str(out_dir)})
    pd.DataFrame(summary_rows).to_csv(out_root / "real_suite_manifest_results.csv", index=False)
    # Merge key outputs across datasets for paper-level tables.
    merged_scores = []
    merged_diag = []
    merged_cycles = []
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        dname = row["dataset"]
        od = Path(row["out_dir"])
        for fname, acc in [("real_arena_scores.csv", merged_scores), ("real_selection_diagnostics.csv", merged_diag), ("real_arena_high_confidence_cycles.csv", merged_cycles)]:
            fp = od / fname
            if fp.exists() and fp.stat().st_size > 0:
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue
                if not df.empty:
                    df.insert(0, "dataset", dname)
                    acc.append(df)
    if merged_scores:
        pd.concat(merged_scores, ignore_index=True).to_csv(out_root / "all_real_scores.csv", index=False)
    if merged_diag:
        pd.concat(merged_diag, ignore_index=True).to_csv(out_root / "all_real_selection_diagnostics.csv", index=False)
    if merged_cycles:
        pd.concat(merged_cycles, ignore_index=True).to_csv(out_root / "all_real_high_confidence_cycles.csv", index=False)
    print(f"[real-suite] wrote {out_root}")

def run_summarize(args) -> None:
    out_dir = Path(args.out)
    write_latex_tables(out_dir)
    append_extra_latex_tables(out_dir)
    write_figures(out_dir)
    write_extra_figures(out_dir)
    write_summary_report(out_dir)
    print(f"[done] Summary files updated in {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NeurIPS-grade STE experiment suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synthetic", help="Run synthetic planted-core + ablations + bootstrap + scaling")
    ps.add_argument("--config", type=str, default=None)
    ps.add_argument("--out", type=str, default=None)
    ps.add_argument("--only-summary", action="store_true", help="Regenerate summaries/figures without rerunning experiments")
    ps.set_defaults(func=run_synthetic)

    pr = sub.add_parser("real-arena", help="Analyze Chatbot-Arena-style human pairwise preference CSV")
    pr.add_argument("--input", required=True, help="CSV with model_a, model_b, winner columns")
    pr.add_argument("--out", required=True)
    pr.add_argument("--agent-a-col", default="model_a")
    pr.add_argument("--agent-b-col", default="model_b")
    pr.add_argument("--winner-col", default="winner")
    pr.add_argument("--category-col", default="category")
    pr.add_argument("--by-category", action="store_true")
    pr.add_argument("--methods", default="ste_posterior_edge_uc,ste_posterior_edge_tc,ste_plugin_uc,ste_plugin_tc,hard_uc,hard_tc,winrate,btl,elo,trueskill,rank_centrality,hodge,pagerank,copeland,schulze,minimax,ranked_pairs,kemeny_local")
    pr.add_argument("--tau", type=float, default=0.035)
    pr.add_argument("--reachability", default="max_min")
    pr.add_argument("--bootstrap", type=int, default=300)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--cycle-min-count", type=int, default=20)
    pr.add_argument("--cycle-confidence", type=float, default=0.95)
    pr.add_argument("--max-cycles", type=int, default=100)
    pr.set_defaults(func=run_real_arena)

    pa = sub.add_parser("agentbench", help="Analyze AgentBench-style execution CSV")
    pa.add_argument("--input", required=True, help="CSV with environment,agent,task_id and score/success/status")
    pa.add_argument("--out", required=True)
    pa.add_argument("--methods", default="ste_posterior_edge_uc,ste_posterior_edge_tc,ste_plugin_uc,ste_plugin_tc,hard_uc,hard_tc,winrate,btl,elo,trueskill,rank_centrality,hodge,pagerank,copeland,schulze,minimax,ranked_pairs,kemeny_local")
    pa.add_argument("--tau", type=float, default=0.035)
    pa.add_argument("--reachability", default="max_min")
    pa.add_argument("--bootstrap", type=int, default=300)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--cycle-min-count", type=int, default=5)
    pa.add_argument("--cycle-confidence", type=float, default=0.90)
    pa.add_argument("--max-cycles", type=int, default=100)
    pa.set_defaults(func=run_agentbench)


    pg = sub.add_parser("scorelog", help="Analyze generic task score logs by converting same-task scores into pairwise comparisons")
    pg.add_argument("--input", required=True, help="CSV with environment,agent,task_id and score/success/status")
    pg.add_argument("--out", required=True)
    pg.add_argument("--methods", default="ste_posterior_edge_uc,ste_posterior_edge_tc,ste_plugin_uc,ste_plugin_tc,hard_uc,hard_tc,winrate,btl,elo,trueskill,rank_centrality,hodge,pagerank,copeland,schulze,minimax,ranked_pairs,kemeny_local")
    pg.add_argument("--tau", type=float, default=0.035)
    pg.add_argument("--reachability", default="max_min")
    pg.add_argument("--bootstrap", type=int, default=300)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--cycle-min-count", type=int, default=5)
    pg.add_argument("--cycle-confidence", type=float, default=0.90)
    pg.add_argument("--max-cycles", type=int, default=100)
    pg.set_defaults(func=run_scorelog)

    prs = sub.add_parser("real-suite", help="Run a multi-dataset real benchmark suite from a YAML manifest")
    prs.add_argument("--manifest", required=True)
    prs.add_argument("--out", default=None)
    prs.add_argument("--methods", default="ste_posterior_edge_uc,ste_posterior_edge_tc,ste_plugin_uc,ste_plugin_tc,hard_uc,hard_tc,winrate,btl,elo,trueskill,rank_centrality,hodge,pagerank,copeland,schulze,minimax,ranked_pairs,kemeny_local")
    prs.add_argument("--bootstrap", type=int, default=300)
    prs.add_argument("--seed", type=int, default=0)
    prs.set_defaults(func=run_real_suite)

    pm = sub.add_parser("summarize", help="Regenerate tables, figures, and report from existing outputs")
    pm.add_argument("--out", required=True)
    pm.set_defaults(func=run_summarize)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
