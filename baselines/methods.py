"""Baseline methods for tournament aggregation.

These baselines are used to contextualize STE results on both synthetic and
real-world pairwise comparison datasets.

Design goals:
- Deterministic and reproducible given (P_hat, comparisons).
- Reasonably faithful implementations (no placeholder outputs).
- Efficient enough for n up to a few hundred on CPU.

Included methods (returned by BaselineEvaluator.evaluate_all):
- win_rate        : Laplace-smoothed empirical win rate from comparisons
- copeland        : Copeland score on majority edges (P_hat > 0.5)
- elo             : Elo ratings (multi-epoch over comparisons)
- btl             : Bradley-Terry-Luce MLE via vectorized MM updates
- rank_centrality : Stationary distribution of Rank Centrality Markov chain
- hodgerank       : Least-squares HodgeRank score (Laplacian solve)

Optional (only if dependency is available):
- trueskill       : TrueSkill ratings via the `trueskill` package

All methods return a length-n score vector (higher = better).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def scores_to_ranking(scores: np.ndarray) -> np.ndarray:
    """Convert scores (higher is better) to rank positions (0 best)."""
    s = np.asarray(scores)
    order = np.argsort(-s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(s.shape[0])
    return ranks


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _scores_from_winrate(comparisons: np.ndarray, n: int, alpha: float = 1.0) -> np.ndarray:
    wins = np.zeros(n, dtype=np.float64)
    total = np.zeros(n, dtype=np.float64)
    comps = np.asarray(comparisons, dtype=np.int64)

    for a, b, y in comps:
        a = int(a); b = int(b); y = int(y)
        if y == 1:
            wins[a] += 1
        else:
            wins[b] += 1
        total[a] += 1
        total[b] += 1

    return (wins + float(alpha)) / (total + 2.0 * float(alpha))


def _copeland_from_phat(P_hat: np.ndarray) -> np.ndarray:
    A = (np.asarray(P_hat) > 0.5).astype(np.int8)
    np.fill_diagonal(A, 0)
    wins = A.sum(axis=1)
    losses = A.sum(axis=0)
    return (wins - losses).astype(np.float64)


def _elo_from_comparisons(
    comparisons: np.ndarray,
    n: int,
    init_rating: float = 1500.0,
    k_factor: float = 32.0,
    logistic_scale: float = 400.0,
    epochs: int = 1,
) -> np.ndarray:
    comps = np.asarray(comparisons, dtype=np.int64)
    r = np.full(n, float(init_rating), dtype=np.float64)

    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / float(logistic_scale)))

    for _ in range(int(epochs)):
        for a, b, y in comps:
            a = int(a); b = int(b); y = int(y)
            ea = expected(r[a], r[b])
            sa = float(y)
            r[a] += float(k_factor) * (sa - ea)
            r[b] += float(k_factor) * ((1.0 - sa) - (1.0 - ea))

    return r


def _rank_centrality_from_phat(
    P_hat: np.ndarray,
    damping: float = 0.85,
    max_iters: int = 2000,
    tol: float = 1e-12,
) -> np.ndarray:
    P = np.asarray(P_hat, dtype=np.float64)
    n = P.shape[0]

    # Transition probability i->j proportional to probability j beats i
    # Normalize by (n-1) to keep rows approximately stochastic.
    T = P.T / max(float(n - 1), 1.0)
    np.fill_diagonal(T, 0.0)
    T[np.arange(n), np.arange(n)] = 1.0 - T.sum(axis=1)

    pi = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(int(max_iters)):
        pi_new = pi @ T
        # Teleportation
        pi_new = float(damping) * pi_new + (1.0 - float(damping)) * (1.0 / n)
        if np.linalg.norm(pi_new - pi, ord=1) < float(tol):
            pi = pi_new
            break
        pi = pi_new

    return pi


def _btl_mle_from_comparisons(
    comparisons: np.ndarray,
    n: int,
    max_iters: int = 2000,
    tol: float = 1e-10,
    l2_reg: float = 1e-6,
) -> np.ndarray:
    """Bradley-Terry-Luce MLE using **vectorized** MM updates.

    Skills s_i > 0. P(i>j) = s_i / (s_i + s_j).

    Returns log(s) as score.
    """
    comps = np.asarray(comparisons, dtype=np.int64)
    w = np.zeros((n, n), dtype=np.float64)
    m = np.zeros((n, n), dtype=np.float64)

    # Count wins/total comparisons per ordered pair
    for a, b, y in comps:
        a = int(a); b = int(b); y = int(y)
        if y == 1:
            w[a, b] += 1.0
        else:
            w[b, a] += 1.0
        m[a, b] += 1.0
        m[b, a] += 1.0

    wins_i = w.sum(axis=1)

    # Initialize
    s = np.ones(n, dtype=np.float64)

    # Avoid divide-by-zero on diagonal
    np.fill_diagonal(m, 0.0)

    lam = float(l2_reg)
    for _ in range(int(max_iters)):
        s_old = s.copy()

        denom_mat = s[:, None] + s[None, :]
        denom_mat = np.maximum(denom_mat, 1e-12)

        # denom_i = sum_j m_ij / (s_i + s_j)
        denom_i = (m / denom_mat).sum(axis=1)

        s = (wins_i + lam) / (denom_i + lam)
        s = np.maximum(s, 1e-12)
        s = s / (np.mean(s) + 1e-12)

        if np.linalg.norm(s - s_old, ord=1) < float(tol):
            break

    return np.log(s + 1e-12)


def _hodgerank_from_comparisons(comparisons: np.ndarray, n: int, l2_reg: float = 1e-6) -> np.ndarray:
    """HodgeRank least squares score on signed edge outcomes."""
    comps = np.asarray(comparisons, dtype=np.int64)

    L = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for a, c, y in comps:
        a = int(a); c = int(c); y = int(y)
        y_ls = 1.0 if y == 1 else -1.0
        L[a, a] += 1.0
        L[c, c] += 1.0
        L[a, c] -= 1.0
        L[c, a] -= 1.0
        b[a] += y_ls
        b[c] -= y_ls

    L = L + float(l2_reg) * np.eye(n)

    try:
        x = np.linalg.solve(L, b)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(L) @ b

    x = x - np.mean(x)
    return x


def _trueskill_from_comparisons(comparisons: np.ndarray, n: int) -> np.ndarray:
    """TrueSkill rating baseline (optional dependency)."""
    try:
        import trueskill  # type: ignore
    except Exception:
        raise ImportError("trueskill package is not installed")

    env = trueskill.TrueSkill(draw_probability=0.0)
    ratings = [env.create_rating() for _ in range(n)]

    comps = np.asarray(comparisons, dtype=np.int64)
    for a, b, y in comps:
        a = int(a); b = int(b); y = int(y)
        if y == 1:
            (ra,), (rb,) = env.rate([(ratings[a],), (ratings[b],)], ranks=[0, 1])
        else:
            (rb,), (ra,) = env.rate([(ratings[b],), (ratings[a],)], ranks=[0, 1])
        ratings[a] = ra
        ratings[b] = rb

    # Use mu as score
    return np.array([r.mu for r in ratings], dtype=np.float64)


@dataclass
class BaselineEvaluator:
    """Compute baseline score vectors from a tournament."""

    elo_epochs: int = 1

    def evaluate_all(self, P_hat: np.ndarray, comparisons: np.ndarray) -> Dict[str, np.ndarray]:
        n = np.asarray(P_hat).shape[0]
        out: Dict[str, np.ndarray] = {}

        out['win_rate'] = _scores_from_winrate(comparisons, n=n, alpha=1.0)
        out['copeland'] = _copeland_from_phat(P_hat)
        out['elo'] = _elo_from_comparisons(comparisons, n=n, epochs=int(self.elo_epochs))
        out['rank_centrality'] = _rank_centrality_from_phat(P_hat)

        if np.asarray(comparisons).shape[0] > 0:
            out['btl'] = _btl_mle_from_comparisons(comparisons, n=n)
            out['hodgerank'] = _hodgerank_from_comparisons(comparisons, n=n)

        # Optional TrueSkill
        try:
            out['trueskill'] = _trueskill_from_comparisons(comparisons, n=n)
        except Exception:
            pass

        return out
