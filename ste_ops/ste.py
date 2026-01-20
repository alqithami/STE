"""STE operators (Top Cycle + Uncovered Set).

This module is the *core* of the experiment pipeline: it takes an estimated
probabilistic tournament \hat{P} and deterministically computes
probability-like membership scores for tournament solution concepts.

Outputs
-------
- Top Cycle membership score: ``t_tc[a] in [0,1]``
- Uncovered Set membership score: ``u_uc[a] in (0,1)``

Auditability
------------
Everything here is deterministic given inputs. If you enable the pipeline's
"debug/audit" mode, you can serialize the intermediate matrices (D, R, cover)
and reproduce every reported number.

Reachability modes
------------------
The Top Cycle requires a reachability notion. The paper presents a reachability
construction based on powers of a softened adjacency. In practice, summing
path-mass over *all* paths can saturate (and make every vertex look reachable).
To avoid degenerate results ("everyone in the core"), this implementation
supports multiple reachability modes and defaults to an existence-style mode.

Uncovered-set modes
-------------------
The Uncovered Set is defined via the covering relation:
    c covers a  iff  c beats a  AND  \forall b: (a beats b => c beats b)

A common pitfall is using a *strength comparison* witness D[a,b]-D[c,b], which
incorrectly penalizes cases where both a and c beat b but with different
margins. The default implementation uses an implication-style relaxation that
matches the logical condition above.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _tau(x: float, eps: float = 1e-8) -> float:
    return max(float(x), float(eps))


def softmin(values: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    """Differentiable soft-min using logsumexp."""
    t = _tau(tau)
    return -t * torch.logsumexp(-values / t, dim=dim)


def smoothmax(values: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    """Bounded smooth approximation to max.

    Returns a convex combination of the entries (softmax-weighted average),
    therefore always lies in [min(values), max(values)].

    This is preferable to the logsumexp max proxy for values that are meant to
    stay within [0,1] (e.g., cover scores), because logsumexp can exceed the
    true max by ~tau*log(n).
    """
    t = _tau(tau)
    w = torch.softmax(values / t, dim=dim)
    return torch.sum(w * values, dim=dim)


# -----------------------------------------------------------------------------
# Stage 1: Soft majority edges
# -----------------------------------------------------------------------------

def soft_majority_edge(P_hat: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft majority edge matrix D_tau = sigmoid((P_hat - 0.5)/tau).

    Args:
        P_hat: (n,n) matrix with entries in [0,1]. For pairwise probabilities
               we expect P_hat[b,a] = 1 - P_hat[a,b] and diagonal ~0.5.
        tau: temperature. Smaller => sharper majority edges.

    Returns:
        D: (n,n) in (0,1) with diagonal zeroed.
    """
    D = torch.sigmoid((P_hat - 0.5) / _tau(tau))
    D = D.clone()
    D.fill_diagonal_(0.0)
    return D


# -----------------------------------------------------------------------------
# Stage 2: Reachability
# -----------------------------------------------------------------------------

def reachability_sum_mass(D: torch.Tensor, K: int, alpha: float = 1.0) -> torch.Tensor:
    """Legacy reachability mass R_mass = sum_{k=1..K} alpha^{k-1} D^k.

    Warning:
        This can saturate because it aggregates over many paths.

    Returns:
        R_mass: (n,n) nonnegative (not a probability).
    """
    K = int(K)
    if K <= 0:
        raise ValueError("K must be >= 1")
    R = torch.zeros_like(D)
    power = D
    for k in range(1, K + 1):
        R = R + (float(alpha) ** (k - 1)) * power
        power = power @ D
    return R


def _max_plus(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Max-plus matrix product: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})."""
    tmp = A.unsqueeze(2) + B.unsqueeze(0)  # (i,k,j)
    return tmp.max(dim=1).values


def reachability_max_product(D: torch.Tensor, K: int, eps: float = 1e-12) -> torch.Tensor:
    """Existence-style reachability via best-path (max-product) semantics.

    Treat D[i,j] in (0,1) as edge strength; path strength is product of edges;
    reachability is the maximum path strength across paths of length 1..K.

    Returns:
        R: (n,n) in [0,1], diagonal 0.
    """
    K = int(K)
    if K <= 0:
        raise ValueError("K must be >= 1")

    logD = torch.log(torch.clamp(D, min=float(eps), max=1.0))
    log_power = logD
    log_best = log_power.clone()

    for _ in range(2, K + 1):
        log_power = _max_plus(log_power, logD)
        log_best = torch.maximum(log_best, log_power)

    R = torch.exp(log_best)
    R = torch.clamp(R, 0.0, 1.0)
    R = R.clone()
    R.fill_diagonal_(0.0)
    return R


def _max_min_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Max-min matrix product: (A ⊗ B)_{ij} = max_k min(A_{ik}, B_{kj})."""
    tmp = torch.minimum(A.unsqueeze(2), B.unsqueeze(0))
    return tmp.max(dim=1).values


def reachability_max_min(D: torch.Tensor, K: int) -> torch.Tensor:
    """Fuzzy reachability using max-min transitive closure up to length K."""
    K = int(K)
    if K <= 0:
        raise ValueError("K must be >= 1")

    R = D.clone()
    for _ in range(2, K + 1):
        R = torch.maximum(R, _max_min_product(R, D))

    R = torch.clamp(R, 0.0, 1.0)
    R = R.clone()
    R.fill_diagonal_(0.0)
    return R


# -----------------------------------------------------------------------------
# Stage 3: Top Cycle membership
# -----------------------------------------------------------------------------

def top_cycle_membership(R: torch.Tensor, tau_softmin: float) -> torch.Tensor:
    """Top Cycle membership score t(a)=softmin_{b!=a} R[a,b]."""
    Rm = R.clone()
    Rm.fill_diagonal_(1e6)
    t = softmin(Rm, tau=float(tau_softmin), dim=1)
    return torch.clamp(t, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Stage 4: Uncovered Set membership
# -----------------------------------------------------------------------------

def cover_lukasiewicz(D: torch.Tensor, tau_imp: float) -> torch.Tensor:
    """Soft covering score using Łukasiewicz implication.

    Hard condition:
        c covers a iff D[c,a]=1 and for all b: (D[a,b]=1 => D[c,b]=1).

    Łukasiewicz implication on [0,1]:
        I(u,v) = clamp(1 - u + v, 0, 1)

    Soft cover score:
        cover(c,a) = D[c,a] * softmin_b I(D[a,b], D[c,b])

    Returns:
        cover: (n,n) in [0,1], diagonal 0.
    """
    t = _tau(tau_imp)

    # Broadcast to shape (c,a,b)
    D_a_b = D.unsqueeze(0)         # (1,a,b)
    D_c_b = D.unsqueeze(1)         # (c,1,b)
    I = torch.clamp(1.0 - D_a_b + D_c_b, 0.0, 1.0)  # (c,a,b)

    min_I = softmin(I, tau=t, dim=2)
    min_I = torch.clamp(min_I, 0.0, 1.0)

    cover = D * min_I
    cover = cover.clone()
    cover.fill_diagonal_(0.0)
    return cover


def cover_violation(D: torch.Tensor, tau_violation: float) -> torch.Tensor:
    """Soft covering score using an implication violation witness.

    Violation for a triple (c,a,b):
        v(c,a,b) = D[a,b] * (1 - D[c,b])

    If a strongly beats b but c does not, v is high.

    We aggregate with a bounded smoothmax over b:
        witness(c,a) = smoothmax_b v(c,a,b)

    Then:
        cover(c,a) = D[c,a] * (1 - witness(c,a))

    Returns:
        cover: (n,n) in [0,1], diagonal 0.
    """
    t = _tau(tau_violation)

    # (c,a,b)
    v = D.unsqueeze(0) * (1.0 - D.unsqueeze(1))
    witness = smoothmax(v, tau=t, dim=2)

    cover = D * (1.0 - witness)
    cover = cover.clone()
    cover.fill_diagonal_(0.0)
    return torch.clamp(cover, 0.0, 1.0)


def cover_witness_exp(D: torch.Tensor, tau_witness: float) -> torch.Tensor:
    """Legacy covering score based on strength comparison witness.

    This mode is kept for compatibility/ablations.

    witness(c,a) = max_b (D[a,b] - D[c,b])
    cover(c,a)   = D[c,a] * exp(-relu(witness)/tau_witness)

    Warning:
        This can be overly strict and can collapse to "nobody covers anyone"
        for small tau_witness.
    """
    t = _tau(tau_witness)

    diff = D.unsqueeze(0) - D.unsqueeze(1)  # (c,a,b)
    witness = diff.max(dim=2).values

    cover = D * torch.exp(-torch.relu(witness) / t)
    cover = cover.clone()
    cover.fill_diagonal_(0.0)
    return torch.clamp(cover, 0.0, 1.0)


def uncovered_membership_from_cover(
    cover: torch.Tensor,
    tau_coverer: float,
    beta: float,
) -> torch.Tensor:
    """Convert cover(c,a) scores into uncovered membership u(a).

    Hard uncovered set:
        a is uncovered iff max_c cover(c,a) = 0.

    Soft:
        max_coverer(a) = smoothmax_c cover(c,a)
        u(a) = sigmoid(beta * (0.5 - max_coverer(a)))

    With beta large, u>0.5 roughly corresponds to "no strong coverer".
    """
    t = _tau(tau_coverer)

    max_coverer = smoothmax(cover, tau=t, dim=0)
    max_coverer = torch.clamp(max_coverer, 0.0, 1.0)

    u = torch.sigmoid(float(beta) * (0.5 - max_coverer))
    return u


# -----------------------------------------------------------------------------
# End-to-end score computation
# -----------------------------------------------------------------------------

def compute_ste_scores(
    P_hat: torch.Tensor,
    tau: float = 0.05,
    K: int = 3,
    alpha: float = 1.0,
    reachability_mode: str = "max_product",
    tau_softmin: Optional[float] = None,
    uncovered_mode: str = "lukasiewicz",
    tau_imp: Optional[float] = None,
    tau_violation: Optional[float] = None,
    tau_witness: Optional[float] = None,
    tau_coverer: Optional[float] = None,
    beta_uncovered: float = 5.0,
    beta: Optional[float] = None,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute STE membership scores.

    Args:
        P_hat: (n,n) estimated probability matrix.
        tau: temperature for soft majority edges.
        K: max path length.
        alpha: damping for sum-mass reachability.
        reachability_mode:
            - "max_product" (default): best-path reachability (existence-style)
            - "max_min"              : fuzzy max-min reachability
            - "sum_mass"             : legacy sum of path-masses (can saturate)
        tau_softmin: temperature for Top Cycle softmin (defaults to tau).

        uncovered_mode:
            - "lukasiewicz" (default): implication-based cover relaxation
            - "violation"            : implication violation witness
            - "witness_exp"          : legacy strength-witness exponential

        tau_imp: softness for implication min (lukasiewicz). Defaults to tau.
        tau_violation: softness for violation witness smoothmax (violation). Defaults to tau.
        tau_witness: softness for legacy witness_exp. Defaults to tau.
        tau_coverer: softness for aggregating max coverer. Defaults to tau.
        beta_uncovered: sigmoid sharpness.
        return_aux: if True, also return intermediate matrices for auditing.

    Returns:
        t_tc: (n,) Top Cycle membership scores in [0,1]
        u_uc: (n,) Uncovered Set membership probabilities in (0,1)
        aux (optional): dict with D, R, cover
    """
    if tau_softmin is None:
        tau_softmin = tau
    if tau_imp is None:
        tau_imp = tau
    if tau_violation is None:
        tau_violation = tau
    if tau_witness is None:
        tau_witness = tau
    if tau_coverer is None:
        tau_coverer = tau

    # Backward-compatible alias: older call sites pass beta=...
    if beta is not None:
        beta_uncovered = float(beta)

    D = soft_majority_edge(P_hat, tau=float(tau))

    rm = str(reachability_mode).lower().strip()
    if rm in {"max_product", "viterbi", "max"}:
        R = reachability_max_product(D, K=int(K))
    elif rm in {"max_min", "maxmin"}:
        R = reachability_max_min(D, K=int(K))
    elif rm in {"sum_mass", "sum", "mass"}:
        R_mass = reachability_sum_mass(D, K=int(K), alpha=float(alpha))
        # Map mass to [0,1] proxy (monotone). Still can saturate.
        R = 1.0 - torch.exp(-torch.clamp(R_mass, min=0.0))
        R = torch.clamp(R, 0.0, 1.0)
        R = R.clone()
        R.fill_diagonal_(0.0)
    else:
        raise ValueError(f"Unknown reachability_mode: {reachability_mode}")

    t_tc = top_cycle_membership(R, tau_softmin=float(tau_softmin))

    um = str(uncovered_mode).lower().strip()
    if um in {"lukasiewicz", "luk"}:
        cover = cover_lukasiewicz(D, tau_imp=float(tau_imp))
    elif um in {"violation", "implication"}:
        cover = cover_violation(D, tau_violation=float(tau_violation))
    elif um in {"witness_exp", "witness", "exp"}:
        cover = cover_witness_exp(D, tau_witness=float(tau_witness))
    else:
        raise ValueError(f"Unknown uncovered_mode: {uncovered_mode}")

    u_uc = uncovered_membership_from_cover(
        cover,
        tau_coverer=float(tau_coverer),
        beta=float(beta_uncovered),
    )

    if return_aux:
        return t_tc, u_uc, {"D": D, "R": R, "cover": cover}
    return t_tc, u_uc


def top_cycle_membership_prob(t_tc: torch.Tensor) -> torch.Tensor:
    """Backward-compatible helper: clamp TC score into [0,1]."""
    return torch.clamp(t_tc, 0.0, 1.0)
