"""Uncovered-set inference for independent random tournament orientations.

``q[..., a, b]`` is the probability that a beats b. Off-diagonal entries
are reciprocal, and different unordered edges are independent. It is an
orientation probability, not a posterior mean of a latent pair probability.
The diagonal is unused; constructors set it to .5. Torch is optional and
imported lazily. No GPU work is launched by importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import numpy as np
from scipy.special import betaincc, expit, logsumexp


@dataclass
class MarginalEstimate:
    """Monte Carlo estimates; variance is per draw, SE is for the mean."""

    marginals: np.ndarray
    variance: np.ndarray
    standard_errors: np.ndarray
    draws: int
    samples: np.ndarray | None = None


@dataclass
class JointUCResult(MarginalEstimate):
    """Joint UC membership/cardinality statistics needed by expected-F1 GFM."""

    membership_size_joint: np.ndarray | None = None
    cardinality_probabilities: np.ndarray | None = None
    half_decision: np.ndarray | None = None
    gfm_decision: np.ndarray | None = None
    gfm_expected_f1: float = 0.0
    memberships: np.ndarray | None = None


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("This operation requires the optional torch dependency.") from exc
    return torch


def _is_torch(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


def _square_shape(shape):
    if len(shape) < 2 or shape[-1] != shape[-2] or shape[-1] < 1:
        raise ValueError("Expected a square (..., n, n) matrix with n >= 1.")
    return shape[-1]


def validate_q(q, *, atol: float | None = None):
    """Validate finite [0,1] reciprocal orientations without clipping inputs.

    Supports NumPy and Torch, including leading batch dimensions. The diagonal
    does not enter tournament calculations but must be finite and in [0,1].
    Tolerance only accommodates floating-point reciprocity checks.
    """
    if atol is not None and (not np.isfinite(atol) or atol < 0):
        raise ValueError("Reciprocity tolerance must be finite and nonnegative.")
    if _is_torch(q):
        torch = _torch()
        n = _square_shape(q.shape)
        if not q.is_floating_point():
            raise ValueError("Torch orientation probabilities must be floating point.")
        tolerance = (1e-6 if q.dtype in (torch.float16, torch.bfloat16, torch.float32)
                     else 1e-10) if atol is None else atol
        if not bool(torch.isfinite(q).all()) or bool(((q < 0) | (q > 1)).any()):
            raise ValueError("Orientation probabilities must be finite and in [0,1].")
        mask = ~torch.eye(n, dtype=torch.bool, device=q.device)
        if bool((torch.abs((q + q.transpose(-1, -2))[..., mask] - 1) > tolerance).any()):
            raise ValueError("Off-diagonal orientation probabilities must be reciprocal.")
        return q
    original = np.asarray(q)
    arr = original.astype(np.float64)
    n = _square_shape(arr.shape)
    if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError("Orientation probabilities must be finite and in [0,1].")
    mask = ~np.eye(n, dtype=bool)
    low_precision = original.dtype.kind == "f" and original.dtype.itemsize <= 4
    tolerance = (1e-6 if low_precision else 1e-10) if atol is None else atol
    if np.any(np.abs((arr + arr.swapaxes(-1, -2))[..., mask] - 1) > tolerance):
        raise ValueError("Off-diagonal orientation probabilities must be reciprocal.")
    return arr


def _validate_wins(wins):
    arr = np.asarray(wins, dtype=np.float64)
    n = _square_shape(arr.shape)
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("Decisive win counts must be finite and nonnegative.")
    if np.any(np.diagonal(arr, axis1=-2, axis2=-1) != 0):
        raise ValueError("The diagonal of a decisive win-count matrix must be zero.")
    return arr, n


def _numpy_q(q):
    """Validated CPU probabilities for deliberately nondifferentiable routines."""
    validated = validate_q(q)
    if _is_torch(validated):
        return validated.detach().to(dtype=_torch().float64).cpu().numpy()
    return validated


def mixture_orientation_probabilities(components, weights):
    """Marginal of the actual normalized, upper-edge tournament sampling law.

    Samplers draw one global component and use its upper-triangle edge
    probabilities. Compute that marginal in float64, set reverse entries as
    complements, and set the diagonal exactly to .5. A final clip guards only
    convex-sum roundoff; invalid component probabilities are rejected first.
    This construction does not assert independence of the marginal edges.
    """
    arrays = np.asarray(components, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if arrays.ndim != 3 or len(arrays) < 1:
        raise ValueError("Components must have shape components by n by n")
    n = _square_shape(arrays.shape)
    if (not np.isfinite(arrays).all() or np.any((arrays < 0) | (arrays > 1))
            or not np.allclose(arrays + arrays.swapaxes(-1, -2), 1., atol=1e-6, rtol=0)):
        raise ValueError("Components must be finite reciprocal probabilities in [0,1]")
    with np.errstate(over="ignore", invalid="ignore"):
        total = weights.sum()
    if (weights.shape != (len(arrays),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(total) or total <= 0):
        raise ValueError("Mixture weights must be finite, nonnegative and have positive sum")
    normalized = weights / total
    i, j = np.triu_indices(n, 1)
    upper = np.clip((normalized[:, None] * arrays[:, i, j]).sum(0), 0., 1.)
    result = np.full((n, n), .5, dtype=np.float64)
    result[i, j], result[j, i] = upper, 1. - upper
    return validate_q(result, atol=0.)


def jeffreys_orientation_probabilities(wins):
    """Return q_ab = Pr(theta_ab > .5 | wins), with a Jeffreys Beta prior.

    ``wins[a,b]`` counts decisive wins of a over b; ties must not be inserted
    into these counts. Missing/tie-only pairs yield .5. Compute one orientation
    as ``1-betainc(w_ab+.5, w_ba+.5, .5)``. The smaller orientation tail is
    evaluated directly by SciPy's complementary betaincc function to avoid
    cancellation, and the larger tail is its complement.
    Clipping here is only a guard against special-function roundoff; arbitrary
    user q inputs are never repaired or clipped.
    """
    wins, n = _validate_wins(wins)
    i, j = np.triu_indices(n, 1)
    a, b = wins[..., i, j] + .5, wins[..., j, i] + .5
    forward_is_smaller = a <= b
    probability = betaincc(np.minimum(a, b), np.maximum(a, b), .5)
    if not np.isfinite(probability).all():
        raise FloatingPointError("The incomplete-beta backend returned nonfinite values.")
    # Identical beta shapes have exact symmetry around .5. Avoid letting
    # special-function roundoff turn an exact posterior tie into an ordering.
    probability = np.where(wins[..., i, j] == wins[..., j, i], .5, probability)
    probability = np.clip(probability, 0.0, 1.0)
    q = np.full(wins.shape, .5, dtype=np.float64)
    q[..., i, j] = np.where(forward_is_smaller, probability, 1.0 - probability)
    q[..., j, i] = np.where(forward_is_smaller, 1.0 - probability, probability)
    return q


def jeffreys_pair_means(wins):
    """Return posterior means of latent pair probabilities (not orientations)."""
    wins, _ = _validate_wins(wins)
    total = wins + wins.swapaxes(-1, -2)
    if not np.isfinite(total).all():
        raise FloatingPointError("Win-count totals overflowed floating point.")
    return (wins + .5) / (total + 1.0)


def exact_uc(adjacency, *, validate: bool = True):
    """Return UC membership of each strict tournament, optionally batched.

    A vertex is uncovered exactly when it reaches every other vertex in at
    most two directed steps. Input is (..., n, n) Boolean adjacency. Torch
    returns a Boolean tensor on the input device; NumPy returns an array.
    This identity requires a strict tournament, hence validation is on by
    default. It is not an oracle for arbitrary incomplete directed graphs.
    """
    if _is_torch(adjacency):
        torch = _torch()
        n = _square_shape(adjacency.shape)
        if adjacency.dtype != torch.bool:
            raise ValueError("Tournament adjacency must be Boolean.")
        eye = torch.eye(n, dtype=torch.bool, device=adjacency.device)
        if validate:
            if bool(adjacency[..., eye].any()):
                raise ValueError("Tournament adjacency diagonal must be false.")
            off = ~eye
            if not bool((adjacency ^ adjacency.transpose(-1, -2))[..., off].all()):
                raise ValueError("Exactly one direction is required on every unordered edge.")
        two = adjacency.to(torch.float32) @ adjacency.to(torch.float32) > 0
        return (adjacency | two | eye).all(dim=-1)
    arr = np.asarray(adjacency)
    n = _square_shape(arr.shape)
    if arr.dtype != np.bool_:
        raise ValueError("Tournament adjacency must be Boolean.")
    eye = np.eye(n, dtype=bool)
    if validate:
        if arr[..., eye].any():
            raise ValueError("Tournament adjacency diagonal must be false.")
        if not (arr ^ arr.swapaxes(-1, -2))[..., ~eye].all():
            raise ValueError("Exactly one direction is required on every unordered edge.")
    two = np.matmul(arr.astype(np.int64), arr.astype(np.int64)) > 0
    return (arr | two | eye).all(axis=-1)


def _subsets(n: int):
    return ((np.arange(1 << n, dtype=np.uint64)[:, None]
             >> np.arange(n, dtype=np.uint64)) & 1).astype(bool)


def _conditional_h(q, v: int, masks):
    """H_v(W) for outgoing-neighbor masks W; q is B,n,n, masks S,n-1."""
    others = np.delete(np.arange(q.shape[-1]), v)
    sub = q[:, others[:, None], others[None, :]]
    covered = np.where(masks[None, :, None, :], sub[:, None, :, :], 1.0).prod(-1)
    return np.where(masks[None, :, :], 1.0, 1.0 - covered).prod(-1)


def exact_uc_moments(q, *, max_n: int = 16, subset_batch_size: int = 2048):
    """Enumerate outgoing sets to obtain exact UC marginals and RB variance.

    Given W=N+(v), independent outside-edge blocks give
    H_v(W)=prod_{u outside W union {v}} (1-prod_{w in W} q_uw).
    Empty products use 1. The subset sum costs O(n^3 2^(n-1)), not a
    full-tournament enumeration. ``max_n`` is a resource guard, not a model
    restriction. Returned variances compare individual estimator draws;
    they make no claim about equal wall-clock effort or runtime.
    """
    arr = _numpy_q(q)
    n = arr.shape[-1]
    if n > max_n:
        raise ValueError(f"Subset enumeration limited to n <= {max_n}; got n={n}.")
    if subset_batch_size < 1:
        raise ValueError("subset_batch_size must be positive.")
    leading = arr.shape[:-2]
    flat = arr.reshape((-1, n, n))
    masks = _subsets(n - 1)
    mean = np.zeros((len(flat), n))
    for v in range(n):
        others = np.delete(np.arange(n), v)
        outgoing = flat[:, v, others]
        for start in range(0, len(masks), subset_batch_size):
            current = masks[start:start + subset_batch_size]
            weights = np.where(current[None], outgoing[:, None], 1 - outgoing[:, None]).prod(-1)
            h = _conditional_h(flat, v, current)
            mean[:, v] += (weights * h).sum(-1)
    # A centered second pass avoids cancellation near deterministic outcomes.
    rb_var = np.zeros_like(mean)
    for v in range(n):
        others = np.delete(np.arange(n), v)
        outgoing = flat[:, v, others]
        for start in range(0, len(masks), subset_batch_size):
            current = masks[start:start + subset_batch_size]
            weights = np.where(current[None], outgoing[:, None], 1 - outgoing[:, None]).prod(-1)
            h = _conditional_h(flat, v, current)
            rb_var[:, v] += (weights * (h - mean[:, v, None]) ** 2).sum(-1)
    shape = leading + (n,)
    return {"mean": mean.reshape(shape), "rb_variance": rb_var.reshape(shape),
            "bernoulli_variance": (mean * (1 - mean)).reshape(shape)}


def exact_uc_marginals(q, *, max_n: int = 16, subset_batch_size: int = 2048):
    """Exact marginal Pr(v in UC) from reciprocal independent-edge q."""
    if _is_torch(q):
        return exact_uc_marginals_torch(q, max_n=min(max_n, 10))
    return exact_uc_moments(q, max_n=max_n, subset_batch_size=subset_batch_size)["mean"]


def exact_uc_marginals_torch(q, *, max_n: int = 10):
    """Differentiable exact subset formula on Torch CPU or CUDA, n <= 10.

    Gradients flow through the probability polynomial, including at q=0/1;
    no logarithms, detached edge probabilities, or sampling are used. When
    parameterizing reciprocal edges, form q_ji=1-q_ij before calling.
    """
    torch = _torch()
    q = validate_q(q)
    if not _is_torch(q):
        raise TypeError("exact_uc_marginals_torch requires a Torch tensor.")
    n = q.shape[-1]
    if n > min(max_n, 10):
        raise ValueError(f"Differentiable subset enumeration limited to n <= {min(max_n, 10)}.")
    masks = torch.as_tensor(_subsets(n - 1), device=q.device)
    flat = q.reshape(-1, n, n)
    result = []
    for v in range(n):
        others = torch.cat((torch.arange(v, device=q.device), torch.arange(v + 1, n, device=q.device)))
        outgoing = flat[:, v, others]
        weights = torch.where(masks[None], outgoing[:, None], 1 - outgoing[:, None]).prod(-1)
        sub = flat[:, others[:, None], others[None, :]]
        covered = torch.where(masks[None, :, None, :], sub[:, None], 1.0).prod(-1)
        h = torch.where(masks[None], 1.0, 1 - covered).prod(-1)
        result.append((weights * h).sum(-1))
    return torch.stack(result, dim=-1).reshape(q.shape[:-2] + (n,))


def _validate_draws(draws, batch_size):
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 2:
        raise ValueError("At least two integer draws are required to estimate variance.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")


def rb_uc_marginals(q, draws: int, *, seed: int = 0, batch_size: int = 512,
                    keep_samples: bool = False):
    """Unbiased UC marginals by Rao-Blackwellizing all non-v incident edges.

    Independently sample each vertex's outgoing set W using q_vw, then compute
    H_v(W). Variance is an unbiased sample estimate of Var(H_v), and standard
    errors divide it by draws. This estimator produces no joint UC samples;
    use joint_uc_monte_carlo for expected-F1 decisions.
    """
    arr = _numpy_q(q)
    if arr.ndim != 2:
        raise ValueError("RB Monte Carlo accepts one n x n q matrix.")
    _validate_draws(draws, batch_size)
    n = len(arr)
    rng = np.random.default_rng(seed)
    mean = np.zeros(n)
    centered_sum = np.zeros(n)
    count = 0
    saved = [] if keep_samples else None
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        h = np.ones((size, n))
        for v in range(n):
            others = np.delete(np.arange(n), v)
            masks = rng.random((size, n - 1)) < arr[v, others]
            h[:, v] = _conditional_h(arr[None], v, masks)[0]
        batch_mean = h.mean(0)
        batch_centered_sum = ((h - batch_mean) ** 2).sum(0)
        difference = batch_mean - mean
        updated_count = count + size
        mean += difference * size / updated_count
        centered_sum += batch_centered_sum + difference * difference * count * size / updated_count
        count = updated_count
        if saved is not None:
            saved.append(h)
    variance = centered_sum / (draws - 1)
    return MarginalEstimate(mean, variance, np.sqrt(variance / draws), draws,
                            np.concatenate(saved) if saved is not None else None)


def sample_tournaments(q, draws: int, *, seed: int = 0, device: str | None = None):
    """Generate full strict tournaments, once per independent unordered edge.

    ``device=None`` uses NumPy; a Torch device string uses a generator on that
    device. This convenience function materializes all adjacency draws. The
    streaming joint estimator below is preferable for large draw counts.
    """
    arr = _numpy_q(q)
    if arr.ndim != 2:
        raise ValueError("Sampling accepts one n x n q matrix.")
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer.")
    n = len(arr)
    i, j = np.triu_indices(n, 1)
    if device is None:
        orientations = np.random.default_rng(seed).random((draws, len(i))) < arr[i, j]
        adjacency = np.zeros((draws, n, n), dtype=bool)
    else:
        torch = _torch()
        generator = torch.Generator(device=device).manual_seed(seed)
        probability = torch.as_tensor(arr[i, j], dtype=torch.float64, device=device)
        orientations = torch.rand((draws, len(i)), dtype=torch.float64,
                                   device=device, generator=generator) < probability
        adjacency = torch.zeros((draws, n, n), dtype=torch.bool, device=device)
    adjacency[:, i, j] = orientations
    adjacency[:, j, i] = ~orientations
    return adjacency


def marginal_half_decision(marginals):
    """Bayes action for additive symmetric membership error; include .5 ties."""
    arr = np.asarray(marginals, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError("Marginals must be a finite [0,1] vector.")
    return arr >= .5


def gfm_decision_from_joint(membership_size_joint, cardinality_probabilities=None):
    """Maximize expected set F1 using Pr(v in Y, |Y|=s), not marginals alone.

    For candidate size k>0, rank vertices by sum_s 2 p_vs/(k+s), then choose
    the top k. F1(empty,empty)=1. Exact ties prefer the smaller prediction
    size (including ties within floating-point roundoff); within a size they
    prefer lower vertex indices. Joint columns are
    s=0,...,n. Return (Boolean prediction, expected F1).
    """
    joint = np.asarray(membership_size_joint, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != joint.shape[0] + 1:
        raise ValueError("Joint statistics must have shape (n, n+1).")
    n = len(joint)
    if n < 1 or not np.isfinite(joint).all() or np.any(joint < 0) or np.any(joint > 1):
        raise ValueError("Joint probabilities must be finite and in [0,1].")
    if np.any(joint[:, 0] > 1e-10):
        raise ValueError("No vertex can belong to a set of cardinality zero.")
    sizes = np.arange(1, n + 1)
    inferred = joint[:, 1:].sum(0) / sizes
    if cardinality_probabilities is None:
        card = np.r_[max(0.0, 1 - inferred.sum()), inferred]
    else:
        card = np.asarray(cardinality_probabilities, dtype=float)
        if card.shape != (n + 1,) or not np.isfinite(card).all() or np.any(card < 0):
            raise ValueError("Cardinality probabilities must be a nonnegative n+1 vector.")
        if not np.isclose(card.sum(), 1, atol=1e-9, rtol=0) or not np.allclose(card[1:], inferred, atol=1e-9, rtol=0):
            raise ValueError("Cardinality probabilities disagree with joint statistics.")
    if inferred.sum() > 1 + 1e-9:
        raise ValueError("Joint statistics imply total cardinality probability greater than one.")
    if np.any(joint > card[None] + 1e-9):
        raise ValueError("Membership/size joint mass cannot exceed its cardinality mass.")
    best = np.zeros(n, dtype=bool)
    best_value = float(card[0])
    for k in range(1, n + 1):
        scores = (2 * joint[:, 1:] / (k + sizes)[None]).sum(1)
        order = np.argsort(-scores, kind="stable")
        value = float(scores[order[:k]].sum())
        roundoff = 8 * np.finfo(np.float64).eps * max(1.0, abs(value), abs(best_value))
        if value > best_value + roundoff:
            best_value = value
            best = np.zeros(n, dtype=bool)
            best[order[:k]] = True
    return best, best_value


def joint_uc_monte_carlo(q, draws: int, *, seed: int = 0, batch_size: int = 512,
                         device: str | None = None, keep_memberships: bool = False,
                         membership_path: str | Path | None = None):
    """Full-tournament joint UC Monte Carlo and native half/GFM decisions.

    Joint samples, rather than independently sampled UC labels, preserve
    membership/cardinality dependence. ``device`` optionally accelerates the
    adjacency oracle with Torch, but all returned statistics are NumPy. Raw
    membership rows can be retained or saved in a compressed NPZ containing
    memberships, q, seed, and draw count. Fixed seeds are reproducible within
    a backend; identical NumPy/Torch random streams are not promised.
    """
    arr = _numpy_q(q)
    if arr.ndim != 2:
        raise ValueError("Joint Monte Carlo accepts one n x n q matrix.")
    _validate_draws(draws, batch_size)
    n = len(arr)
    i, j = np.triu_indices(n, 1)
    joint = np.zeros((n, n + 1), dtype=np.int64)
    cardinality = np.zeros(n + 1, dtype=np.int64)
    saved = [] if keep_memberships or membership_path is not None else None
    if device is None:
        rng = np.random.default_rng(seed)
    else:
        torch = _torch()
        generator = torch.Generator(device=device).manual_seed(seed)
        probability = torch.as_tensor(arr[i, j], dtype=torch.float64, device=device)
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        if device is None:
            orientations = rng.random((size, len(i))) < arr[i, j]
            adjacency = np.zeros((size, n, n), dtype=bool)
        else:
            orientations = torch.rand((size, len(i)), dtype=torch.float64, device=device,
                                      generator=generator) < probability
            adjacency = torch.zeros((size, n, n), dtype=torch.bool, device=device)
        adjacency[:, i, j] = orientations
        adjacency[:, j, i] = ~orientations
        memberships = exact_uc(adjacency, validate=False)
        if device is not None:
            memberships = memberships.cpu().numpy()
        sizes = memberships.sum(1)
        cardinality += np.bincount(sizes, minlength=n + 1)
        for s in np.unique(sizes):
            joint[:, s] += memberships[sizes == s].sum(0)
        if saved is not None:
            saved.append(memberships)
    joint_probability = joint / draws
    card_probability = cardinality / draws
    mean = joint_probability.sum(1)
    variance = mean * (1 - mean) * draws / (draws - 1)
    gfm, gfm_value = gfm_decision_from_joint(joint_probability, card_probability)
    raw = np.concatenate(saved) if saved is not None else None
    if membership_path is not None:
        path = Path(membership_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            # Preserve the source precision so reciprocity checks upon replay
            # can use the same dtype-aware floating-point tolerance.
            if _is_torch(q):
                cpu_q = q.detach().cpu()
                if cpu_q.dtype == _torch().bfloat16:
                    cpu_q = cpu_q.to(dtype=_torch().float32)
                saved_q = cpu_q.numpy()
            else:
                saved_q = np.asarray(q)
            np.savez_compressed(handle, memberships=raw, q=saved_q,
                                seed=np.asarray(seed), draws=np.asarray(draws))
    return JointUCResult(
        marginals=mean, variance=variance, standard_errors=np.sqrt(variance / draws),
        draws=draws, membership_size_joint=joint_probability,
        cardinality_probabilities=card_probability,
        half_decision=marginal_half_decision(mean), gfm_decision=gfm,
        gfm_expected_f1=gfm_value, memberships=raw if keep_memberships else None,
    )


def soft_uc_plugin(probabilities, *, tau: float = .035, gamma: float = .035):
    """Canonical STE normalized-LSE UC score, a plug-in score, not a posterior.

    D_ab=sigmoid((P_ab-.5)/tau), with D_aa=0. Witnesses use normalized
    log-sum-exp of D_vw(1-D_uw) over w excluding v,u; the outer normalized
    log-sum-exp combines D_uv(1-witness). n=2 uses empty witness=0, n=1
    returns 1. Positive tau/gamma are mandatory. Supports leading batches.
    """
    if not np.isfinite(tau) or not np.isfinite(gamma) or tau <= 0 or gamma <= 0:
        raise ValueError("tau and gamma must be finite and positive.")
    p = validate_q(probabilities)
    n = p.shape[-1]
    if _is_torch(p):
        torch = _torch()
        d = torch.sigmoid((p - .5) / tau) * ~torch.eye(n, dtype=torch.bool, device=p.device)
        if n == 1:
            return p[..., 0, :] * 0 + 1
        maximum = lambda x: gamma * (torch.logsumexp(x / gamma, dim=-1) - math.log(x.shape[-1]))
        stack = lambda items: torch.stack(items, dim=-1)
        zero = p[..., 0, 0] * 0
    else:
        d = expit((p - .5) / tau) * ~np.eye(n, dtype=bool)
        if n == 1:
            return np.ones(p.shape[:-2] + (1,))
        maximum = lambda x: gamma * (logsumexp(x / gamma, axis=-1) - math.log(x.shape[-1]))
        stack = lambda items: np.stack(items, axis=-1)
        zero = np.zeros(p.shape[:-2])
    output = []
    for v in range(n):
        covers = []
        for u in range(n):
            if u == v:
                continue
            witnesses = [d[..., v, w] * (1 - d[..., u, w])
                         for w in range(n) if w not in (v, u)]
            witness = maximum(stack(witnesses)) if witnesses else zero
            covers.append(d[..., u, v] * (1 - witness))
        output.append(1 - maximum(stack(covers)))
    return stack(output)
