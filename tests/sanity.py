"""Sanity checks for STE operators.

Run from the `ste/` directory:
  python -m tests.sanity

These checks are designed to catch degenerate operator behavior that can
make experimental outputs appear "fake" (e.g., predicting the full set
for a clearly transitive tournament).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ste_ops.ste import compute_ste_scores, top_cycle_membership_prob


def make_transitive(n: int, p: float = 0.9) -> np.ndarray:
    """Total order tournament: i<j => i beats j with probability p."""
    P = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = float(p)
            P[j, i] = 1.0 - float(p)
    np.fill_diagonal(P, 0.5)
    return P


def make_cycle(n: int = 3, p: float = 0.9) -> np.ndarray:
    """Directed n-cycle: i -> (i+1 mod n) with probability p."""
    if n < 3:
        raise ValueError('cycle requires n>=3')
    P = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        P[i, j] = float(p)
        P[j, i] = 1.0 - float(p)
    np.fill_diagonal(P, 0.5)
    return P


def check_case(
    name: str,
    P: np.ndarray,
    expected_tc_size: int,
    expected_uc_size: int,
    tau: float = 0.05,
    K: Optional[int] = None,
) -> None:
    n = int(P.shape[0])
    if K is None:
        K = min(3, max(1, n - 1))

    P_t = torch.tensor(P, dtype=torch.float32)
    t, u = compute_ste_scores(
        P_t,
        tau=float(tau),
        K=int(K),
        beta=5.0,
        reachability_mode='max_product',
    )

    tc_prob = top_cycle_membership_prob(t).detach().cpu().numpy()
    uc_prob = u.detach().cpu().numpy()

    tc_pred = (tc_prob > 0.5).astype(int)
    uc_pred = (uc_prob > 0.5).astype(int)

    print(f"\n[{name}] n={n}, tau={tau}, K={K}")
    print('  tc_prob:', np.round(tc_prob, 3))
    print('  tc_pred:', tc_pred, 'size=', int(tc_pred.sum()))
    print('  uc_prob:', np.round(uc_prob, 3))
    print('  uc_pred:', uc_pred, 'size=', int(uc_pred.sum()))

    assert np.all(np.isfinite(tc_prob)), 'Non-finite TC probabilities'
    assert np.all(np.isfinite(uc_prob)), 'Non-finite UC probabilities'

    if int(tc_pred.sum()) != int(expected_tc_size):
        raise AssertionError(
            f"TC size mismatch for {name}: got {int(tc_pred.sum())}, expected {expected_tc_size}"
        )
    if int(uc_pred.sum()) != int(expected_uc_size):
        raise AssertionError(
            f"UC size mismatch for {name}: got {int(uc_pred.sum())}, expected {expected_uc_size}"
        )


def main() -> None:
    # Transitive: unique Condorcet winner -> TC size 1, UC size 1
    check_case('transitive', make_transitive(8), expected_tc_size=1, expected_uc_size=1)

    # 3-cycle: strongly connected -> TC size 3, UC size 3
    check_case('3-cycle', make_cycle(3), expected_tc_size=3, expected_uc_size=3, K=3)

    print('\nAll sanity checks passed.')


if __name__ == '__main__':
    main()
