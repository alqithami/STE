"""Synthetic tournament generation for STE experiments.

This module generates probabilistic tournaments with controllable:
- transitive base signal (BTL strengths)
- cyclicity injection strength (rho)
- observation noise (eta)
- missingness/sparsity (mu)

It also computes *ground-truth* Top Cycle and Uncovered Set on the
majority tournament induced by the underlying probability matrix P.

The goal is to provide a reproducible synthetic benchmark with known
core sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SyntheticTournament:
    n: int
    P: np.ndarray  # true probabilistic tournament, shape (n, n)
    P_hat: np.ndarray  # estimated from observations, shape (n, n)
    comparisons: np.ndarray  # rows: (a, b, y) where y=1 if a beats b
    true_top_cycle: np.ndarray  # binary vector length n
    true_uncovered: np.ndarray  # binary vector length n


def _build_cycle_matrix(
    n: int,
    cycle_nodes: List[int],
    cycle_edge_prob: float = 0.7,
    non_cycle_prob: float = 0.5,
) -> np.ndarray:
    """Construct a probabilistic tournament matrix representing a directed cycle
    on `cycle_nodes`.

    For edges on the directed cycle, set P(i>j)=cycle_edge_prob.
    For non-cycle pairs, set to non_cycle_prob (usually 0.5).

    Ensures P[j,i] = 1 - P[i,j] and P[i,i]=0.5.
    """
    P = np.full((n, n), non_cycle_prob, dtype=np.float64)
    np.fill_diagonal(P, 0.5)

    k = len(cycle_nodes)
    for idx in range(k):
        a = cycle_nodes[idx]
        b = cycle_nodes[(idx + 1) % k]
        P[a, b] = cycle_edge_prob
        P[b, a] = 1.0 - cycle_edge_prob

    # Enforce antisymmetry globally
    for i in range(n):
        for j in range(i + 1, n):
            P[j, i] = 1.0 - P[i, j]

    np.fill_diagonal(P, 0.5)
    return P


def _majority_adjacency(P: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return adjacency matrix A where A[i,j]=1 iff P[i,j] > threshold."""
    A = (P > threshold).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def _top_cycle(A: np.ndarray) -> np.ndarray:
    """Compute the Top Cycle (a.k.a. Smith set / top cycle) of a tournament.

    We use a practical equivalent characterization for tournaments:
    Top Cycle = { i : for all j, j is reachable from i in A }.

    This is consistent with the common "reachability" definition used in
    STE-style formulations.

    Returns a 0/1 vector of length n.
    """
    n = A.shape[0]

    # BFS reachability from each node
    reachable_all = np.zeros(n, dtype=np.int8)
    for start in range(n):
        seen = np.zeros(n, dtype=bool)
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            nbrs = np.where(A[u] == 1)[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        # Node is in top cycle if it can reach every node
        if seen.all():
            reachable_all[start] = 1

    # In some degenerate settings, this set can be empty due to numerical ties.
    # Fall back to the (unique) maximal SCC heuristic if needed.
    if reachable_all.sum() == 0:
        # Compute transitive closure and take nodes in largest SCC of closure.
        # For tournaments, there is always at least one maximal SCC; this is a
        # defensible fallback for near-tie graphs.
        closure = A.astype(bool)
        # Warshall
        for k in range(n):
            closure = closure | (closure[:, [k]] & closure[[k], :])
        # SCCs in closure using Kosaraju
        def _dfs_order(graph: np.ndarray):
            visited = np.zeros(n, dtype=bool)
            order: List[int] = []
            for i in range(n):
                if visited[i]:
                    continue
                st = [i]
                while st:
                    node = st[-1]
                    if visited[node]:
                        st.pop()
                        continue
                    visited[node] = True
                    # push neighbors
                    nbrs2 = np.where(graph[node])[0]
                    for nb in nbrs2:
                        if not visited[nb]:
                            st.append(nb)
                    # postorder
                    # We can't easily do postorder in iterative form without extra state;
                    # instead use recursion-less two-pass marker.
                    # Simpler: use recursion for n<=200.
                    break
            # Use recursion for correctness
            visited = np.zeros(n, dtype=bool)
            order = []
            import sys
            sys.setrecursionlimit(10000)

            def dfs(u: int):
                visited[u] = True
                for v in np.where(graph[u])[0]:
                    if not visited[v]:
                        dfs(v)
                order.append(u)

            for i in range(n):
                if not visited[i]:
                    dfs(i)
            return order

        order = _dfs_order(closure)
        rev = closure.T
        visited = np.zeros(n, dtype=bool)
        comps: List[List[int]] = []

        import sys
        sys.setrecursionlimit(10000)

        def dfs_rev(u: int, comp: List[int]):
            visited[u] = True
            comp.append(u)
            for v in np.where(rev[u])[0]:
                if not visited[v]:
                    dfs_rev(v, comp)

        for u in reversed(order):
            if not visited[u]:
                comp: List[int] = []
                dfs_rev(u, comp)
                comps.append(comp)

        largest = max(comps, key=len)
        reachable_all[largest] = 1

    return reachable_all.astype(float)


def _uncovered_set(A: np.ndarray) -> np.ndarray:
    """Compute Uncovered Set for a tournament adjacency matrix A.

    In a tournament, c covers a if:
      (1) c beats a
      (2) for all b, if a beats b then c beats b

    Uncovered set = nodes not covered by anyone.

    Returns a 0/1 vector of length n.
    """
    n = A.shape[0]
    uncovered = np.ones(n, dtype=bool)

    for a in range(n):
        for c in range(n):
            if c == a:
                continue
            if A[c, a] != 1:
                continue
            # Condition (2)
            a_beats = np.where(A[a] == 1)[0]
            if a_beats.size == 0:
                # If a beats nobody, then any c that beats a covers a (vacuously)
                uncovered[a] = False
                break
            if np.all(A[c, a_beats] == 1):
                uncovered[a] = False
                break

    return uncovered.astype(float)


def generate_synthetic_tournament(
    n: int,
    rho: float,
    eta: float,
    mu: float,
    m_per_pair: int,
    seed: int,
    cycle_size: int = 3,
    # How strongly transitive the base BTL signal is. Lower => closer to 0.5,
    # making cycle injection visible at smaller rho.
    btl_scale: float = 1.0,
    num_cycles: int = 1,
    cycle_edge_prob: float = 0.7,
    # How to inject cyclicity:
    # - 'edge_only' (recommended): only modify the directed cycle edges,
    #   leaving all other pairs at the base BTL probabilities.
    # - 'matrix_mix' (legacy): build a cycle matrix with non-cycle pairs set
    #   to 0.5 and mix P = (1-rho) P_base + rho P_cycle. This can flatten many
    #   edges toward 0.5 and make the benchmark degenerate.
    cycle_injection_mode: str = 'edge_only',
    laplace_alpha: float = 1.0,
) -> SyntheticTournament:
    """Generate a synthetic tournament and observed comparisons.

    Args:
        n: number of agents
        rho: cyclicity mixing coefficient in [0,1]
        eta: label flip probability in [0,1]
        mu: fraction of unordered pairs removed (sparsity) in [0,1)
        m_per_pair: number of comparisons per observed unordered pair
        seed: RNG seed
        cycle_size: size of injected cycle (default 3)
        btl_scale: scaling factor applied to Bradley-Terry logits; smaller values
            make the base tournament closer to random (0.5/0.5), which ensures
            cyclicity effects are visible at moderate rho.
        num_cycles: how many cycles to inject (default 1)
        cycle_edge_prob: probability on cycle edges in injected cycles (default 0.7)
        cycle_injection_mode: 'edge_only' (recommended) or 'matrix_mix' (legacy)
        laplace_alpha: Laplace smoothing parameter for P_hat

    Returns:
        SyntheticTournament
    """
    rng = np.random.default_rng(seed)

    # 1) latent strengths
    lam = rng.normal(loc=0.0, scale=1.0, size=(n,))

    # 2) base BTL probabilities
    # NOTE: scaling is important; if the base signal is too strong, injected
    # cycles rarely flip majority edges unless rho is extremely high, which
    # makes the "F1 vs rho" experiment appear flat.
    diff = float(btl_scale) * (lam[:, None] - lam[None, :])
    P_base = _sigmoid(diff)
    np.fill_diagonal(P_base, 0.5)

    # 3) cycle injection
    # IMPORTANT:
    #   Earlier versions used a full-matrix mix with non-cycle pairs set to 0.5.
    #   That tends to flatten large portions of P toward 0.5 for moderate rho,
    #   which makes the recovery benchmark (and STE scores) appear degenerate.
    #
    #   The default here is 'edge_only': keep P_base everywhere, and only blend
    #   the directed cycle edges toward cycle_edge_prob.
    mode = str(cycle_injection_mode).lower().strip()

    if mode == 'matrix_mix':
        P_cycle = np.full((n, n), 0.5, dtype=np.float64)
        np.fill_diagonal(P_cycle, 0.5)
        if cycle_size >= 2 and int(num_cycles) > 0:
            mats = []
            for _ in range(int(num_cycles)):
                cycle_nodes = rng.choice(n, size=int(cycle_size), replace=False).tolist()
                mats.append(
                    _build_cycle_matrix(
                        n=n,
                        cycle_nodes=cycle_nodes,
                        cycle_edge_prob=float(cycle_edge_prob),
                        non_cycle_prob=0.5,
                    )
                )
            P_cycle = np.mean(np.stack(mats, axis=0), axis=0)
        P = (1.0 - rho) * P_base + rho * P_cycle

    elif mode == 'edge_only':
        P = P_base.copy()
        if cycle_size >= 2 and int(num_cycles) > 0 and float(rho) > 0.0:
            for _ in range(int(num_cycles)):
                cycle_nodes = rng.choice(n, size=int(cycle_size), replace=False).tolist()
                k = len(cycle_nodes)
                for idx in range(k):
                    a = cycle_nodes[idx]
                    b = cycle_nodes[(idx + 1) % k]
                    # Blend only this directed edge toward cycle_edge_prob.
                    p_ab = (1.0 - float(rho)) * P_base[a, b] + float(rho) * float(cycle_edge_prob)
                    P[a, b] = p_ab
                    P[b, a] = 1.0 - p_ab
        np.fill_diagonal(P, 0.5)

    else:
        raise ValueError(f"Unknown cycle_injection_mode: {cycle_injection_mode}")

    # Enforce antisymmetry
    for i in range(n):
        for j in range(i + 1, n):
            P[j, i] = 1.0 - P[i, j]
    np.fill_diagonal(P, 0.5)

    # Ground truth cores from majority tournament
    A = _majority_adjacency(P, threshold=0.5)
    true_tc = _top_cycle(A)
    true_uc = _uncovered_set(A)

    # Determine which unordered pairs are observed
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    n_drop = int(round(mu * len(pairs)))
    dropped = set(pairs[:n_drop])

    # Sample comparisons
    comparisons: List[Tuple[int, int, int]] = []
    for (i, j) in pairs:
        if (i, j) in dropped:
            continue
        for _ in range(m_per_pair):
            y = 1 if rng.random() < P[i, j] else 0
            # Noise flip
            if rng.random() < eta:
                y = 1 - y
            comparisons.append((i, j, y))

    comps = np.array(comparisons, dtype=np.int64)

    # Build P_hat from counts
    w = np.zeros((n, n), dtype=np.float64)
    m = np.zeros((n, n), dtype=np.float64)

    for (i, j, y) in comparisons:
        if y == 1:
            w[i, j] += 1.0
        else:
            w[j, i] += 1.0
        m[i, j] += 1.0
        m[j, i] += 1.0

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

    return SyntheticTournament(
        n=n,
        P=P,
        P_hat=P_hat,
        comparisons=comps,
        true_top_cycle=true_tc,
        true_uncovered=true_uc,
    )
