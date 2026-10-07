"""Sound UC certificates, with explicit separation of logical and statistical claims.

``wins[i, j]`` is the number of observations in which i beats j.  Pair sample
sizes are ``wins + wins.T``.  The formal Hoeffding guarantee applies only to
fixed, nonadaptive sample sizes and i.i.d. Bernoulli outcomes within each pair.
Independence between different pairs is NOT needed for simultaneous coverage.
Finite-cohort and adaptive-count calls produce numerical diagnostics only.

All arrays use vertices 0, ..., n-1.  A partial tournament is represented by a
Boolean adjacency matrix: two false entries mean an unknown edge.  No ties are
resolved by fiat.  The frequentist UC target requires p[i,j] != 1/2 off diagonal.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

import numpy as np

Edge = tuple[int, int]


def _square(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.shape[0]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    return array


def _partial(known) -> np.ndarray:
    raw = _square(known, "known")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError("known must contain only Boolean/0/1 edge indicators")
    adjacency = raw.astype(bool, copy=True)
    if np.any(np.diag(adjacency)) or np.any(adjacency & adjacency.T):
        raise ValueError("known must have no self edges or bidirectional pairs")
    return adjacency


def _q_matrix(q) -> np.ndarray:
    array = _square(q, "q").astype(float)
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError("q must contain finite probabilities in [0, 1]")
    off = ~np.eye(len(array), dtype=bool)
    if not np.allclose((array + array.T)[off], 1.0, atol=1e-12, rtol=0):
        raise ValueError("q[i,j] + q[j,i] must equal 1 off diagonal")
    return array


@dataclass(frozen=True)
class PairwiseIntervals:
    lower: np.ndarray
    upper: np.ndarray
    known: np.ndarray
    sample_sizes: np.ndarray
    delta: float
    total_pairs: int
    observed_pairs: int
    sampling_model: str
    fixed_nonadaptive: bool
    coverage_guarantee: bool
    guarantee_label: str

    def asdict(self) -> dict:
        return {
            "lower": self.lower.tolist(), "upper": self.upper.tolist(),
            "known": self.known.astype(int).tolist(),
            "sample_sizes": self.sample_sizes.tolist(), "delta": self.delta,
            "total_pairs": self.total_pairs, "observed_pairs": self.observed_pairs,
            "sampling_model": self.sampling_model,
            "fixed_nonadaptive": self.fixed_nonadaptive,
            "coverage_guarantee": self.coverage_guarantee,
            "guarantee_label": self.guarantee_label,
        }


def simultaneous_hoeffding_intervals(
    wins, delta: float = 0.05, *, sampling_model: str = "iid_bernoulli",
    fixed_nonadaptive: bool = True,
) -> PairwiseIntervals:
    """Build a fixed-delta, simultaneous envelope over all predefined pairs.

    For each observed unordered pair, radius = sqrt(log(2*M/delta)/(2*m)),
    with M=n*(n-1)/2.  Unobserved pairs receive [0,1]; diagonals are [0.5,0.5].
    Known orientations use STRICT lower>0.5 (equivalently upper<0.5 in the
    reverse direction).  Counts are required to be finite nonnegative integers.

    ``sampling_model='finite_cohort'`` is accepted solely for a descriptive
    envelope. It does not certify a finite empirical cohort or a population.
    Likewise, adaptive or stopped sample sizes do not get fixed-time coverage.
    Set the assumptions honestly; this function cannot audit the data generator.
    The intervals cover latent Bernoulli probabilities, not observed vote shares.
    """
    counts = _square(wins, "wins").astype(float)
    if (not np.all(np.isfinite(counts)) or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or np.any(np.diag(counts) != 0)):
        raise ValueError("wins must be finite nonnegative integer counts with zero diagonal")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be strictly between 0 and 1")
    if sampling_model not in {"iid_bernoulli", "finite_cohort"}:
        raise ValueError("sampling_model must be 'iid_bernoulli' or 'finite_cohort'")
    if not isinstance(fixed_nonadaptive, (bool, np.bool_)):
        raise ValueError("fixed_nonadaptive must be Boolean")
    n = len(counts)
    total_pairs = n * (n - 1) // 2
    sizes = counts + counts.T
    lower, upper = np.zeros((n, n)), np.ones((n, n))
    np.fill_diagonal(lower, 0.5)
    np.fill_diagonal(upper, 0.5)
    observed_pairs = 0
    for i, j in combinations(range(n), 2):
        m = sizes[i, j]
        if m == 0:
            continue
        observed_pairs += 1
        radius = np.sqrt(np.log(2 * total_pairs / delta) / (2 * m))
        mean = counts[i, j] / m
        lo, hi = max(0.0, mean - radius), min(1.0, mean + radius)
        lower[i, j], upper[i, j] = lo, hi
        lower[j, i], upper[j, i] = 1.0 - hi, 1.0 - lo
    known = lower > 0.5
    np.fill_diagonal(known, False)
    formal = sampling_model == "iid_bernoulli" and bool(fixed_nonadaptive)
    label = (
        "fixed_nonadaptive_iid_bernoulli_latent_probability_simultaneous_coverage"
        if formal else "descriptive_only_no_frequentist_coverage_claim"
    )
    return PairwiseIntervals(lower, upper, known, sizes, float(delta), total_pairs,
                             observed_pairs, sampling_model, bool(fixed_nonadaptive),
                             formal, label)


def depth_two_certificate(known, vertex: int) -> tuple[Edge, ...] | None:
    """Return an n-1-edge spanning out-tree of depth <=2, or None.

    All required edges are known.  A returned tree guarantees UC membership in
    every tournament completion.  Choosing the smallest indexed parent makes
    this function deterministic; it does not minimize a posterior failure cost.
    """
    adjacency = _partial(known)
    n = len(adjacency)
    if not isinstance(vertex, (int, np.integer)) or not 0 <= vertex < n:
        raise ValueError("vertex index out of range")
    children = np.flatnonzero(adjacency[vertex])
    edges: list[Edge] = [(int(vertex), int(w)) for w in children]
    for u in range(n):
        if u == vertex or adjacency[vertex, u]:
            continue
        parents = [int(w) for w in children if adjacency[w, u]]
        if not parents:
            return None
        edges.append((parents[0], u))
    return tuple(sorted(edges))


def covering_witness(known, vertex: int) -> int | None:
    """Return u that covers vertex in EVERY completion, or None."""
    adjacency = _partial(known)
    n = len(adjacency)
    if not isinstance(vertex, (int, np.integer)) or not 0 <= vertex < n:
        raise ValueError("vertex index out of range")
    for u in range(n):
        if u == vertex or not adjacency[u, vertex]:
            continue
        if all(adjacency[w, vertex] or adjacency[u, w]
               for w in range(n) if w not in (u, vertex)):
            return u
    return None


@dataclass(frozen=True)
class UCBounds:
    inner: np.ndarray
    outer: np.ndarray
    member_certificates: Mapping[int, tuple[Edge, ...]]
    nonmember_witnesses: Mapping[int, int]
    method: str
    exact_logical: bool
    coverage_guarantee: bool | None = None
    guarantee_label: str = "structural_every_completion_bounds_only"

    @property
    def lower(self) -> np.ndarray:
        return self.inner

    @property
    def upper(self) -> np.ndarray:
        return self.outer

    @property
    def excluded(self) -> np.ndarray:
        return ~self.outer

    @property
    def is_exact(self) -> bool:
        """Whether both bounds identify the same UC set, on the stated event."""
        return bool(np.array_equal(self.inner, self.outer))

    def asdict(self) -> dict:
        return {
            "inner": np.flatnonzero(self.inner).tolist(),
            "outer": np.flatnonzero(self.outer).tolist(),
            "excluded": np.flatnonzero(self.excluded).tolist(),
            "member_certificates": {str(v): [list(e) for e in edges]
                                    for v, edges in self.member_certificates.items()},
            "nonmember_witnesses": {str(v): int(u)
                                    for v, u in self.nonmember_witnesses.items()},
            "method": self.method, "exact_logical": self.exact_logical,
            "is_exact": self.is_exact,
            "coverage_guarantee": self.coverage_guarantee,
            "guarantee_label": self.guarantee_label,
        }


def certificate_bounds(known, *, exact: bool = False, max_unknown_edges: int = 20) -> UCBounds:
    """Return sound inner/outer bounds, optionally exact over all completions.

    The scalable mode is intentionally described as sufficient certificates:
    it does not assert that every logical necessary/possible member is found.
    Its running time is O(n^3).  When ``known`` is PairwiseIntervals the formal
    label propagates, with the further strict true-tournament assumption.
    Bare adjacency input carries only a structural every-completion claim.
    """
    intervals = known if isinstance(known, PairwiseIntervals) else None
    adjacency = _partial(intervals.known if intervals is not None else known)
    n = len(adjacency)
    members = {}
    nonmembers = {}
    for v in range(n):
        tree = depth_two_certificate(adjacency, v)
        if tree is not None:
            members[v] = tree
        witness = covering_witness(adjacency, v)
        if witness is not None:
            nonmembers[v] = witness
    inner = np.array([v in members for v in range(n)], dtype=bool)
    outer = np.array([v not in nonmembers for v in range(n)], dtype=bool)
    method = "constructive_sufficient_certificates"
    if exact:
        enumerated = enumerate_completion_bounds(adjacency, max_unknown_edges=max_unknown_edges)
        inner, outer = enumerated.inner, enumerated.outer
        method = "exhaustive_all_tournament_completions"
    guarantee = None if intervals is None else intervals.coverage_guarantee
    label = "structural_every_completion_bounds_only"
    if intervals is not None:
        label = ("simultaneous_1_minus_delta_bounds_for_strict_latent_bernoulli_tournament"
                 if guarantee else "descriptive_finite_cohort_or_adaptive_diagnostic_only")
    return UCBounds(inner, outer, members, nonmembers, method, exact, guarantee, label)


def _complete_uc(adjacency: np.ndarray) -> np.ndarray:
    from .graphs import strict_tournament
    adjacency = strict_tournament(adjacency)
    reach = adjacency | ((adjacency.astype(np.int64) @ adjacency.astype(np.int64)) > 0)
    np.fill_diagonal(reach, True)
    return reach.all(axis=1)


def enumerate_completion_bounds(known, *, max_unknown_edges: int = 20) -> UCBounds:
    """Exact intersection/union of UC sets over completions; small graphs only.

    Time is exponential in the number of unknown edges.  The cap is checked
    before enumeration.  These are individual membership sets: a subset of the
    union need not be an attainable complete UC winning set.
    """
    adjacency = _partial(known)
    if not isinstance(max_unknown_edges, (int, np.integer)) or max_unknown_edges < 0:
        raise ValueError("max_unknown_edges must be a nonnegative integer")
    n = len(adjacency)
    unknown = [(i, j) for i, j in combinations(range(n), 2)
               if not adjacency[i, j] and not adjacency[j, i]]
    if len(unknown) > max_unknown_edges:
        raise ValueError(f"{len(unknown)} unknown edges exceed enumeration cap {max_unknown_edges}")
    inner, outer = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    for bits in product((False, True), repeat=len(unknown)):
        completion = adjacency.copy()
        for (i, j), bit in zip(unknown, bits):
            completion[i, j], completion[j, i] = bit, not bit
        uc = _complete_uc(completion)
        inner &= uc
        outer |= uc
    return UCBounds(inner, outer, {}, {}, "exhaustive_all_tournament_completions", True)


def _validate_tree(certificate: Sequence[Edge], n: int, vertex: int) -> tuple[Edge, ...]:
    edges = []
    for edge in certificate:
        if len(edge) != 2:
            raise ValueError("certificate edges must be ordered vertex pairs")
        a, b = edge
        if (not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer))
                or not 0 <= a < n or not 0 <= b < n or a == b):
            raise ValueError("certificate edge endpoint out of range or self edge")
        edges.append((int(a), int(b)))
    if len(set(edges)) != len(edges) or len(edges) != n - 1:
        raise ValueError("certificate must have n-1 distinct edges")
    adjacency = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        adjacency[a, b] = True
    if (np.any(adjacency & adjacency.T) or adjacency[:, vertex].any()
            or any(adjacency[:, u].sum() != 1 for u in range(n) if u != vertex)
            or depth_two_certificate(adjacency, vertex) is None):
        raise ValueError("certificate must be a vertex-rooted spanning out-tree of depth <=2")
    return tuple(sorted(edges))


@dataclass(frozen=True)
class PosteriorCertificate:
    vertex: int
    edges: tuple[Edge, ...]
    union_bound_lower: float
    product_lower: float | None
    independent_edges: bool

    @property
    def lower(self) -> float:
        return self.union_bound_lower if self.product_lower is None else self.product_lower

    def asdict(self) -> dict:
        return {
            "vertex": self.vertex, "edges": [list(e) for e in self.edges],
            "union_bound_lower": self.union_bound_lower,
            "product_lower": self.product_lower, "lower": self.lower,
            "independent_edges": self.independent_edges,
            "interpretation": "conditional_posterior_UC_membership_lower_bound",
        }


def posterior_certificate_bound(
    q, vertex: int, certificate: Sequence[Edge] | None = None, *,
    independent_edges: bool = False,
) -> PosteriorCertificate | None:
    """Bound posterior UC membership using required orientation probabilities q.

    q[i,j]=Pr(P[i,j]>1/2 | data), NOT the posterior mean of P[i,j].  The union
    bound needs no edge independence.  The product bound is returned only when
    the caller asserts joint posterior edge independence.  Data-dependent tree
    selection is allowed here: the bound is conditional on the fixed posterior.
    It does not imply frequentist coverage of a selected certificate.

    If no tree is supplied, build one from q>0.5, returning None if impossible.
    This deterministic choice is a convenience, not an optimal tree search.
    """
    array = _q_matrix(q)
    n = len(array)
    if not isinstance(vertex, (int, np.integer)) or not 0 <= vertex < n:
        raise ValueError("vertex index out of range")
    if not isinstance(independent_edges, (bool, np.bool_)):
        raise ValueError("independent_edges must be Boolean")
    if certificate is None:
        # Orient each unordered pair once: complement validation admits a tiny
        # floating-point tolerance, which must not create bidirectional edges.
        adjacency = np.triu(array > 0.5, 1) | np.triu(array < 0.5, 1).T
        certificate = depth_two_certificate(adjacency, vertex)
        if certificate is None:
            return None
    edges = _validate_tree(certificate, n, vertex)
    required_q = np.array([array[a, b] for a, b in edges], dtype=float)
    union_lower = float(max(0.0, 1.0 - np.sum(1.0 - required_q)))
    product_lower = float(np.prod(required_q)) if independent_edges else None
    return PosteriorCertificate(int(vertex), edges, union_lower, product_lower,
                                bool(independent_edges))
