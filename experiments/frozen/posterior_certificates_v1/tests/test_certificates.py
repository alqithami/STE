"""Exhaustive logical checks and finite-model assumption checks (unittest)."""
import json
import math
import unittest
from itertools import combinations, product

import numpy as np

from stepc.certificates import (
    certificate_bounds, covering_witness, depth_two_certificate,
    enumerate_completion_bounds, posterior_certificate_bound,
    simultaneous_hoeffding_intervals,
)


def independent_uc(adjacency):
    """Set-inclusion oracle, independent of the matrix two-step implementation."""
    n = len(adjacency)
    out = [set(np.flatnonzero(adjacency[v])) for v in range(n)]
    return np.array([not any(adjacency[u, v] and out[v].issubset(out[u])
                             for u in range(n) if u != v)
                     for v in range(n)], dtype=bool)


def completions(known):
    unknown = [(i, j) for i, j in combinations(range(len(known)), 2)
               if not known[i, j] and not known[j, i]]
    for bits in product((0, 1), repeat=len(unknown)):
        adjacency = known.copy()
        for (i, j), bit in zip(unknown, bits):
            adjacency[i, j], adjacency[j, i] = bit, not bit
        yield adjacency


class CertificateTests(unittest.TestCase):
    def check_partial(self, known):
        bounds = certificate_bounds(known)
        sets = [independent_uc(t) for t in completions(known)]
        intersection = np.all(sets, axis=0)
        union = np.any(sets, axis=0)
        self.assertTrue(np.all(~bounds.inner | intersection))
        self.assertTrue(np.all(~union | bounds.outer))
        exact = enumerate_completion_bounds(known)
        np.testing.assert_array_equal(exact.inner, intersection)
        np.testing.assert_array_equal(exact.outer, union)
        merged = certificate_bounds(known, exact=True)
        np.testing.assert_array_equal(merged.inner, intersection)
        np.testing.assert_array_equal(merged.outer, union)
        for v, edges in bounds.member_certificates.items():
            self.assertEqual(len(edges), len(known) - 1)
            self.assertTrue(all(known[a, b] for a, b in edges))
            sparse = np.zeros_like(known)
            for a, b in edges:
                sparse[a, b] = True
            for tournament in completions(sparse):
                self.assertTrue(independent_uc(tournament)[v])
        for v, u in bounds.nonmember_witnesses.items():
            self.assertTrue(known[u, v])
            for tournament in completions(known):
                self.assertTrue(tournament[u, v])
                out_v = set(np.flatnonzero(tournament[v]))
                out_u = set(np.flatnonzero(tournament[u]))
                self.assertTrue(out_v.issubset(out_u))

    def test_all_729_partial_tournaments_n4(self):
        pairs = list(combinations(range(4), 2))
        for states in product((0, 1, 2), repeat=len(pairs)):
            known = np.zeros((4, 4), dtype=bool)
            for (i, j), state in zip(pairs, states):
                if state == 1:
                    known[i, j] = True
                elif state == 2:
                    known[j, i] = True
            self.check_partial(known)

    def test_100_generated_partial_tournaments_n5_every_completion(self):
        rng = np.random.default_rng(90210)
        pairs = list(combinations(range(5), 2))
        for repetition in range(100):
            known = np.zeros((5, 5), dtype=bool)
            density = (repetition % 5) / 4
            for i, j in pairs:
                if rng.random() < density:
                    a, b = (i, j) if rng.integers(2) else (j, i)
                    known[a, b] = True
            self.check_partial(known)

    def test_all_1024_complete_tournaments_n5(self):
        empty = np.zeros((5, 5), dtype=bool)
        for tournament in completions(empty):
            bounds = certificate_bounds(tournament)
            truth = independent_uc(tournament)
            np.testing.assert_array_equal(bounds.inner, truth)
            np.testing.assert_array_equal(bounds.outer, truth)
            self.assertTrue(bounds.is_exact)

    def test_singleton_and_empty_information(self):
        single = certificate_bounds(np.zeros((1, 1), dtype=bool))
        self.assertTrue(single.is_exact)
        self.assertEqual(single.asdict()["inner"], [0])
        unknown = certificate_bounds(np.zeros((4, 4), dtype=bool))
        self.assertFalse(unknown.inner.any())
        self.assertTrue(unknown.outer.all())
        self.assertIsNone(unknown.coverage_guarantee)
        self.assertIsNone(depth_two_certificate(np.zeros((4, 4)), 0))
        self.assertIsNone(covering_witness(np.zeros((4, 4)), 0))

    def test_validation_and_enumeration_cap(self):
        for bad in (np.zeros((0, 0)), np.zeros((2, 3)), np.eye(3), np.ones((3, 3)),
                    np.full((3, 3), 0.2)):
            with self.assertRaises(ValueError):
                certificate_bounds(bad)
        with self.assertRaises(ValueError):
            enumerate_completion_bounds(np.zeros((7, 7)), max_unknown_edges=20)
        with self.assertRaises(ValueError):
            depth_two_certificate(np.zeros((3, 3)), -1)
        with self.assertRaises(ValueError):
            enumerate_completion_bounds(np.zeros((3, 3)), max_unknown_edges=-1)


class IntervalTests(unittest.TestCase):
    def test_fixed_delta_all_pairs_formula_and_unknowns(self):
        counts = np.zeros((4, 4))
        counts[0, 1], counts[1, 0] = 190, 10
        delta = 0.05
        intervals = simultaneous_hoeffding_intervals(counts, delta)
        radius = math.sqrt(math.log(2 * 6 / delta) / (2 * 200))
        self.assertAlmostEqual(intervals.lower[0, 1], 0.95 - radius)
        self.assertAlmostEqual(intervals.upper[0, 1], min(1.0, 0.95 + radius))
        self.assertAlmostEqual(intervals.lower[1, 0], 1 - intervals.upper[0, 1])
        self.assertAlmostEqual(intervals.upper[1, 0], 1 - intervals.lower[0, 1])
        self.assertEqual(intervals.lower[0, 2], 0)
        self.assertEqual(intervals.upper[0, 2], 1)
        np.testing.assert_array_equal(np.diag(intervals.lower), [0.5] * 4)
        self.assertEqual(intervals.total_pairs, 6)
        self.assertEqual(intervals.observed_pairs, 1)
        self.assertTrue(intervals.known[0, 1])
        self.assertFalse(intervals.known[1, 0])
        self.assertTrue(intervals.coverage_guarantee)
        self.assertEqual(intervals.delta, delta)
        json.dumps(intervals.asdict())

    def test_strict_threshold_and_small_sample_unknown(self):
        intervals = simultaneous_hoeffding_intervals([[0, 1], [0, 0]])
        self.assertFalse(intervals.known.any())
        tied_counts = simultaneous_hoeffding_intervals([[0, 1000], [1000, 0]])
        self.assertFalse(tied_counts.known.any())
        np.testing.assert_array_equal(intervals.known, intervals.lower > 0.5)

    def test_statistical_assumption_gate_and_propagation(self):
        counts = np.array([[0, 1000, 1000], [0, 0, 1000], [0, 0, 0]])
        iid = simultaneous_hoeffding_intervals(counts)
        finite = simultaneous_hoeffding_intervals(counts, sampling_model="finite_cohort")
        adaptive = simultaneous_hoeffding_intervals(counts, fixed_nonadaptive=False)
        self.assertTrue(iid.coverage_guarantee)
        for diagnostic in (finite, adaptive):
            self.assertFalse(diagnostic.coverage_guarantee)
            self.assertIn("no_frequentist_coverage", diagnostic.guarantee_label)
            bounds = certificate_bounds(diagnostic)
            self.assertFalse(bounds.coverage_guarantee)
            self.assertIn("diagnostic_only", bounds.guarantee_label)
        formal = certificate_bounds(iid)
        self.assertTrue(formal.coverage_guarantee)
        self.assertTrue(formal.is_exact)
        self.assertEqual(formal.asdict()["inner"], [0])
        json.dumps(formal.asdict())
        with self.assertRaises(ValueError):
            simultaneous_hoeffding_intervals(counts, sampling_model="unknown")

    def test_exact_scalar_binomial_coverage_checks(self):
        # Enumerate all possible counts instead of using stochastic test claims.
        for m in (3, 10, 40):
            for p in (0.01, 0.2, 0.5, 0.8, 0.99):
                for delta in (0.05, 0.2):
                    failure = 0.0
                    for k in range(m + 1):
                        intervals = simultaneous_hoeffding_intervals([[0, k], [m - k, 0]], delta)
                        if not intervals.lower[0, 1] <= p <= intervals.upper[0, 1]:
                            failure += math.comb(m, k) * p**k * (1 - p)**(m-k)
                    self.assertLessEqual(failure, delta + 1e-12)

    def test_singleton_and_input_validation(self):
        single = simultaneous_hoeffding_intervals([[0]])
        self.assertEqual(single.total_pairs, 0)
        self.assertFalse(single.known.any())
        self.assertTrue(certificate_bounds(single).is_exact)
        for counts in ([[0, -1], [0, 0]], [[0, 0.5], [0, 0]], [[1, 0], [0, 0]],
                       [[0, np.nan], [0, 0]]):
            with self.assertRaises(ValueError):
                simultaneous_hoeffding_intervals(counts)
        for delta in (0, 1, -1, np.nan):
            with self.assertRaises(ValueError):
                simultaneous_hoeffding_intervals([[0, 1], [0, 0]], delta)


class PosteriorCertificateTests(unittest.TestCase):
    def test_union_bound_with_dependent_edges_and_product_gate(self):
        # Three correlated orientations: required edges each have q=.75, while
        # both hold with mass .5.  Remaining edge is always 2->0. Actual pi_0=.5.
        q = np.array([[0.5, 0.75, 0], [0.25, 0.5, 0.75], [1, 0.25, 0.5]])
        tree = [(0, 1), (1, 2)]
        bound = posterior_certificate_bound(q, 0, tree)
        self.assertEqual(bound.union_bound_lower, 0.5)
        self.assertIsNone(bound.product_lower)
        self.assertEqual(bound.lower, 0.5)
        independent = posterior_certificate_bound(q, 0, tree, independent_edges=True)
        self.assertEqual(independent.product_lower, 0.5625)
        self.assertGreater(independent.product_lower, 0.5)  # invalid for correlated model

    def test_independent_bound_against_all_tournaments(self):
        q = np.array([[0.5, 0.8, 0.65, 0.6], [0.2, 0.5, 0.75, 0.7],
                      [0.35, 0.25, 0.5, 0.55], [0.4, 0.3, 0.45, 0.5]])
        bound = posterior_certificate_bound(q, 0, [(0, 1), (1, 2), (1, 3)],
                                            independent_edges=True)
        posterior_pi = 0.0
        for adjacency in completions(np.zeros((4, 4), dtype=bool)):
            probability = math.prod(q[i, j] if adjacency[i, j] else q[j, i]
                                    for i, j in combinations(range(4), 2))
            posterior_pi += probability * independent_uc(adjacency)[0]
        self.assertLessEqual(bound.union_bound_lower, posterior_pi + 1e-12)
        self.assertLessEqual(bound.product_lower, posterior_pi + 1e-12)
        self.assertLessEqual(bound.union_bound_lower, bound.product_lower + 1e-12)
        self.assertEqual(len(posterior_certificate_bound(q, 0).edges), 3)
        json.dumps(bound.asdict())

    def test_no_tree_singleton_and_tree_validation(self):
        self.assertIsNone(posterior_certificate_bound(np.full((3, 3), 0.5), 0))
        single = posterior_certificate_bound([[0.5]], 0, [], independent_edges=True)
        self.assertEqual(single.lower, 1)
        q = np.full((4, 4), 0.5)
        for bad in ([(0, 1)], [(0, 1), (0, 1), (0, 2)],
                    [(0, 1), (1, 2), (2, 3)], [(1, 0), (1, 2), (1, 3)]):
            with self.assertRaises(ValueError):
                posterior_certificate_bound(q, 0, bad)
        with self.assertRaises(ValueError):
            posterior_certificate_bound(np.full((3, 3), 0.8), 0)

    def test_complement_roundoff_does_not_create_bidirectional_graph(self):
        q = np.array([[0.5, 0.5000000000001], [0.5000000000001, 0.5]])
        certificate = posterior_certificate_bound(q, 0)
        self.assertEqual(certificate.edges, ((0, 1),))


if __name__ == "__main__":
    unittest.main()
