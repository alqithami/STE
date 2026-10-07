"""Validated directed graph boundaries for corrected v2 software.

Strict partial relations permit missing edges (including exact probability
ties), but never self-loops or bidirectional pairs. A strict tournament requires
one direction for every unordered pair. These are distinct input contracts.
"""
from __future__ import annotations

import numpy as np


def strict_partial_relation(adjacency):
    """Validate Boolean (..., n, n) adjacency without silently repairing it."""
    array = np.asarray(adjacency)
    if array.ndim < 2 or array.shape[-1] != array.shape[-2] or not array.shape[-1]:
        raise ValueError("Relation must be a nonempty square (..., n, n) matrix")
    if array.dtype != np.bool_:
        raise ValueError("Relation adjacency must be Boolean")
    if np.diagonal(array, axis1=-2, axis2=-1).any():
        raise ValueError("Strict relation must not contain self-loops")
    if (array & array.swapaxes(-1, -2)).any():
        raise ValueError("Strict relation must not contain bidirectional pairs")
    return array


def strict_tournament(adjacency):
    """Validate a Boolean strict tournament, optionally batched."""
    array = strict_partial_relation(adjacency)
    off = ~np.eye(array.shape[-1], dtype=bool)
    if not (array ^ array.swapaxes(-1, -2))[..., off].all():
        raise ValueError("Strict tournament requires exactly one direction on every pair")
    return array


def strict_top_cycle(adjacency):
    """Top-cycle membership by reachability on validated strict tournaments.

    This guarded utility is not an endpoint of the posterior UC experiment.
    Historical root-repository graph implementations retain their own archived
    semantics; this function must not be substituted into a historical replay.
    """
    array = strict_tournament(adjacency)
    n = array.shape[-1]
    reach = array.copy() | np.eye(n, dtype=bool)
    for k in range(n):
        reach |= reach[..., :, k, None] & reach[..., None, k, :]
    return reach.all(-1)


def hard_orientation_relation(q):
    """Threshold reciprocal orientation probabilities, preserving exact ties.

    Clear the diagonal explicitly at the decoder boundary even though v2
    constructors set it to exactly .5. Validate the resulting strict partial
    relation. Nonreciprocal inputs must not manufacture two winning directions.
    """
    from .posterior import validate_q
    probabilities = validate_q(q)
    if type(probabilities).__module__.startswith("torch"):
        probabilities = probabilities.detach().cpu().numpy()
    array = np.asarray(probabilities)
    adjacency = array > .5
    diagonal = np.arange(array.shape[-1])
    adjacency[..., diagonal, diagonal] = False
    return strict_partial_relation(adjacency)
