"""Probability estimation utilities for STE experiments.

This module centralizes how we convert observed pairwise comparisons
into a dense probabilistic tournament matrix \hat{P} \in [0,1]^{n\times n}
with \hat{P}[i,i]=0.5 and \hat{P}[j,i]=1-\hat{P}[i,j].

Why this exists:
- Avoids accidental leakage: you can choose whether \hat{P} is fit on all
  comparisons or only the training split.
- Enables a more "IJCAI-grade" setup with held-out predictive evaluation.

Estimators:
- empirical: Laplace-smoothed win-rate counts.
- learned_btl: fit a Bradley--Terry logistic model s_i so that
      P(i>j) = sigmoid((s_i - s_j)/T)
  with optional temperature scaling on the validation split.

No placeholder outputs: all matrices are computed from provided data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------

def split_comparisons(
    comparisons: np.ndarray,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split comparison rows into train/val/test.

    Args:
        comparisons: (m,3) int array (a,b,y) where y=1 means a beats b.
        seed: RNG seed.
        train_frac/val_frac/test_frac: must sum to 1.

    Returns:
        (train, val, test) arrays.
    """
    comps = np.asarray(comparisons, dtype=np.int64)
    m = comps.shape[0]

    fr_sum = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(fr_sum - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {fr_sum}")

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(m)

    n_train = int(round(train_frac * m))
    n_val = int(round(val_frac * m))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return comps[train_idx], comps[val_idx], comps[test_idx]


# -----------------------------------------------------------------------------
# Empirical P_hat
# -----------------------------------------------------------------------------

def phat_from_empirical(
    comparisons: np.ndarray,
    n: int,
    laplace_alpha: float = 1.0,
    fill_value: float = 0.5,
) -> np.ndarray:
    """Estimate a dense P_hat from observed comparisons.

    If an unordered pair (i,j) is unobserved, fill \hat{P}[i,j]=fill_value.

    Laplace smoothing:
      p_ij = (wins_ij + alpha) / (m_ij + 2*alpha)

    Args:
        comparisons: (m,3) rows (a,b,y).
        n: number of agents.
        laplace_alpha: smoothing strength.
        fill_value: probability for unobserved pairs.

    Returns:
        P_hat: (n,n) float64.
    """
    comps = np.asarray(comparisons, dtype=np.int64)
    w = np.zeros((n, n), dtype=np.float64)
    m = np.zeros((n, n), dtype=np.float64)

    for a, b, y in comps:
        a = int(a); b = int(b); y = int(y)
        if y == 1:
            w[a, b] += 1.0
        else:
            w[b, a] += 1.0
        m[a, b] += 1.0
        m[b, a] += 1.0

    P_hat = np.full((n, n), float(fill_value), dtype=np.float64)
    np.fill_diagonal(P_hat, 0.5)

    alpha = float(laplace_alpha)
    for i in range(n):
        for j in range(i + 1, n):
            if m[i, j] > 0:
                p_ij = (w[i, j] + alpha) / (m[i, j] + 2.0 * alpha)
                P_hat[i, j] = p_ij
                P_hat[j, i] = 1.0 - p_ij
            else:
                P_hat[i, j] = float(fill_value)
                P_hat[j, i] = 1.0 - float(fill_value)

    np.fill_diagonal(P_hat, 0.5)
    return P_hat


# -----------------------------------------------------------------------------
# Learned BTL (logistic) + temperature scaling
# -----------------------------------------------------------------------------

@dataclass
class LearnedBTLResult:
    scores: np.ndarray           # (n,) real scores
    temperature: float           # scalar > 0
    train_nll: float
    val_nll: float


def _nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # BCEWithLogits is numerically stable; returns mean NLL
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _eval_nll(scores: torch.Tensor, comps: torch.Tensor, temperature: float) -> float:
    if comps.numel() == 0:
        return float('nan')
    a = comps[:, 0].long()
    b = comps[:, 1].long()
    y = comps[:, 2].float()
    t = max(float(temperature), 1e-8)
    logits = (scores[a] - scores[b]) / t
    loss = _nll_from_logits(logits, y)
    return float(loss.detach().cpu().item())


def fit_learned_btl(
    train_comparisons: np.ndarray,
    val_comparisons: np.ndarray,
    n: int,
    seed: int,
    max_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    early_stop_patience: int = 20,
    calibrate_temperature: bool = True,
    temp_max_iters: int = 200,
    temp_lr: float = 0.1,
    device: Optional[str] = None,
) -> LearnedBTLResult:
    """Fit a logistic BTL model and (optionally) temperature-calibrate on val.

    Args:
        train_comparisons, val_comparisons: arrays (a,b,y).
        n: number of agents.
        seed: RNG seed.
        max_epochs: training epochs.
        batch_size: minibatch size.
        lr: Adam learning rate.
        weight_decay: AdamW/L2.
        early_stop_patience: epochs without improvement before stop.
        calibrate_temperature: if True, fit scalar temperature on val.
        temp_max_iters: iterations for temperature scaling.
        temp_lr: lr for temperature optimizer.
        device: 'cpu' or 'cuda'. If None, auto.

    Returns:
        LearnedBTLResult.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train = torch.tensor(np.asarray(train_comparisons, dtype=np.int64), device=device)
    val = torch.tensor(np.asarray(val_comparisons, dtype=np.int64), device=device)

    # Scores parameter
    scores = torch.nn.Parameter(torch.zeros((n,), device=device))

    opt = torch.optim.AdamW([scores], lr=float(lr), weight_decay=float(weight_decay))

    best_val = float('inf')
    best_scores = None
    bad_epochs = 0

    m = train.shape[0]
    for epoch in range(int(max_epochs)):
        # Shuffle each epoch
        perm = torch.randperm(m, device=device)
        train_shuf = train[perm]

        for start in range(0, m, int(batch_size)):
            batch = train_shuf[start:start + int(batch_size)]
            a = batch[:, 0].long()
            b = batch[:, 1].long()
            y = batch[:, 2].float()
            logits = scores[a] - scores[b]
            loss = _nll_from_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Validation NLL
        val_nll = _eval_nll(scores, val, temperature=1.0)
        if np.isnan(val_nll):
            # No validation data; stop early
            best_scores = scores.detach().clone()
            best_val = val_nll
            break

        if val_nll + 1e-12 < best_val:
            best_val = val_nll
            best_scores = scores.detach().clone()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(early_stop_patience):
                break

    if best_scores is None:
        best_scores = scores.detach().clone()

    # Temperature scaling
    temperature = 1.0
    if calibrate_temperature and val.numel() > 0:
        log_t = torch.nn.Parameter(torch.zeros((), device=device))
        opt_t = torch.optim.Adam([log_t], lr=float(temp_lr))

        for _ in range(int(temp_max_iters)):
            t = torch.exp(log_t)
            nll = _eval_nll(best_scores, val, temperature=float(t.detach().cpu().item()))
            # Backprop through explicit logits
            a = val[:, 0].long()
            b = val[:, 1].long()
            y = val[:, 2].float()
            logits = (best_scores[a] - best_scores[b]) / torch.clamp(t, min=1e-6)
            loss_t = _nll_from_logits(logits, y)
            opt_t.zero_grad(set_to_none=True)
            loss_t.backward()
            opt_t.step()

        temperature = float(torch.exp(log_t).detach().cpu().item())

    train_nll = _eval_nll(best_scores, train, temperature=temperature)
    val_nll = _eval_nll(best_scores, val, temperature=temperature)

    return LearnedBTLResult(
        scores=best_scores.detach().cpu().numpy().astype(np.float64),
        temperature=float(temperature),
        train_nll=float(train_nll),
        val_nll=float(val_nll),
    )


def phat_from_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert learned scores into a dense probabilistic tournament matrix."""
    s = np.asarray(scores, dtype=np.float64)
    n = s.shape[0]
    t = max(float(temperature), 1e-8)
    diff = (s[:, None] - s[None, :]) / t
    P = 1.0 / (1.0 + np.exp(-diff))
    np.fill_diagonal(P, 0.5)
    # Enforce antisymmetry
    for i in range(n):
        for j in range(i + 1, n):
            P[j, i] = 1.0 - P[i, j]
    np.fill_diagonal(P, 0.5)
    return P


# -----------------------------------------------------------------------------
# Predictive metrics
# -----------------------------------------------------------------------------

def predictive_logloss(P_hat: np.ndarray, comparisons: np.ndarray) -> float:
    """Compute predictive log-loss on a comparison set.

    Uses p = P_hat[a,b] as predicted probability that a beats b.
    """
    comps = np.asarray(comparisons, dtype=np.int64)
    if comps.shape[0] == 0:
        return float('nan')

    eps = 1e-12
    a = comps[:, 0]
    b = comps[:, 1]
    y = comps[:, 2].astype(np.float64)
    p = np.clip(P_hat[a, b].astype(np.float64), eps, 1.0 - eps)
    # y=1 -> -log p; y=0 -> -log(1-p)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


def predictive_accuracy(P_hat: np.ndarray, comparisons: np.ndarray, threshold: float = 0.5) -> float:
    comps = np.asarray(comparisons, dtype=np.int64)
    if comps.shape[0] == 0:
        return float('nan')
    a = comps[:, 0]
    b = comps[:, 1]
    y = comps[:, 2].astype(np.int64)
    pred = (P_hat[a, b] >= float(threshold)).astype(np.int64)
    return float(np.mean(pred == y))
