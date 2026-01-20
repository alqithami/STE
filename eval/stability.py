"""Bootstrap stability evaluation for STE.

The paper's stability story typically needs:
- inclusion rates across bootstrap resamples
- a stability statistic (e.g., average Jaccard vs a reference core)

This implementation is designed to be fast enough for n<=50 and
n_bootstrap up to a few hundred. For larger settings, reduce
n_bootstrap or use sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import torch

from ste_ops.ste import compute_ste_scores, top_cycle_membership_prob
from eval.core_metrics import jaccard_index


def _phat_from_comparisons(comparisons: np.ndarray, n: int, laplace_alpha: float = 1.0) -> np.ndarray:
    """Estimate dense P_hat from comparisons (a,b,y)."""
    w = np.zeros((n, n), dtype=np.float64)
    m = np.zeros((n, n), dtype=np.float64)

    for a, b, y in comparisons:
        a = int(a); b = int(b); y = int(y)
        if y == 1:
            w[a, b] += 1.0
        else:
            w[b, a] += 1.0
        m[a, b] += 1.0
        m[b, a] += 1.0

    P_hat = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if m[i, j] > 0:
                p_ij = (w[i, j] + laplace_alpha) / (m[i, j] + 2.0 * laplace_alpha)
                P_hat[i, j] = p_ij
                P_hat[j, i] = 1.0 - p_ij
            else:
                P_hat[i, j] = 0.5
                P_hat[j, i] = 0.5

    np.fill_diagonal(P_hat, 0.5)
    return P_hat


@dataclass
class BootstrapResult:
    tc_probs: np.ndarray  # (B,n)
    uc_probs: np.ndarray  # (B,n)
    tc_cores: np.ndarray  # (B,n) bool
    uc_cores: np.ndarray  # (B,n) bool


def bootstrap_ste(
    comparisons: np.ndarray,
    n: int,
    n_bootstrap: int,
    tau: float,
    K: int,
    seed: int,
    threshold: float = 0.5,
    ste_kwargs: Optional[Dict[str, Any]] = None,
) -> BootstrapResult:
    """Bootstrap STE over resampled comparison rows."""
    rng = np.random.default_rng(seed)
    m = comparisons.shape[0]

    tc_probs = np.zeros((n_bootstrap, n), dtype=np.float64)
    uc_probs = np.zeros((n_bootstrap, n), dtype=np.float64)
    tc_cores = np.zeros((n_bootstrap, n), dtype=bool)
    uc_cores = np.zeros((n_bootstrap, n), dtype=bool)


    # Optional: pass full STE hyperparameters (reachability/uncovered modes, etc.).
    # If omitted, we fall back to tau and K only.
    ste_kwargs_eff: Dict[str, Any] = {} if ste_kwargs is None else dict(ste_kwargs)
    ste_kwargs_eff.setdefault('tau', tau)
    ste_kwargs_eff.setdefault('K', K)

    for b in range(n_bootstrap):
        idx = rng.integers(low=0, high=m, size=(m,))
        samp = comparisons[idx]
        P_hat = _phat_from_comparisons(samp, n=n)
        P_t = torch.from_numpy(P_hat).float()
        t_tau, u_tau = compute_ste_scores(P_t, **ste_kwargs_eff)

        p_tc = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
        p_uc = u_tau.detach().cpu().numpy()

        tc_probs[b] = p_tc
        uc_probs[b] = p_uc
        tc_cores[b] = (p_tc > threshold)
        uc_cores[b] = (p_uc > threshold)

    return BootstrapResult(tc_probs=tc_probs, uc_probs=uc_probs, tc_cores=tc_cores, uc_cores=uc_cores)


def compute_stability_metrics(bootstrap: BootstrapResult, true_core: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute stability metrics from bootstrap outputs.

    Returned metrics:
      - tc_inclusion_mean, uc_inclusion_mean
      - tc_stability_jaccard, uc_stability_jaccard (vs modal core)
      - tc_modal_jaccard_true, uc_modal_jaccard_true (optional)
    """
    tc_inc = bootstrap.tc_cores.mean(axis=0)  # per-agent inclusion
    uc_inc = bootstrap.uc_cores.mean(axis=0)

    tc_modal = (tc_inc > 0.5).astype(float)
    uc_modal = (uc_inc > 0.5).astype(float)

    tc_j = np.mean([jaccard_index(bootstrap.tc_cores[i].astype(float), tc_modal) for i in range(bootstrap.tc_cores.shape[0])])
    uc_j = np.mean([jaccard_index(bootstrap.uc_cores[i].astype(float), uc_modal) for i in range(bootstrap.uc_cores.shape[0])])

    out: Dict[str, float] = {
        'tc_inclusion_mean': float(tc_inc.mean()),
        'uc_inclusion_mean': float(uc_inc.mean()),
        'tc_stability_jaccard': float(tc_j),
        'uc_stability_jaccard': float(uc_j),
        'tc_modal_size': float(tc_modal.sum()),
        'uc_modal_size': float(uc_modal.sum()),
    }

    if true_core is not None:
        out['tc_modal_jaccard_true'] = float(jaccard_index(tc_modal, true_core))
        out['uc_modal_jaccard_true'] = float(jaccard_index(uc_modal, true_core))

    return out
