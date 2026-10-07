"""Independent enumeration and decision checks; run with unittest discovery.

Torch tests explicitly skip when the optional dependency is unavailable.
No test starts training, a GPU production run, or an external-data experiment.
"""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logsumexp

from stepc.posterior import (
    exact_uc, exact_uc_marginals, exact_uc_marginals_torch, exact_uc_moments,
    gfm_decision_from_joint, jeffreys_orientation_probabilities,
    jeffreys_pair_means, joint_uc_monte_carlo, marginal_half_decision,
    rb_uc_marginals, sample_tournaments, soft_uc_plugin, validate_q,
)


def independent_set_uc(adjacency):
    """Set-inclusion covering definition, independent of two-step matrix UC."""
    n = len(adjacency)
    wins = [{j for j in range(n) if adjacency[i, j]} for i in range(n)]
    return np.asarray([not any(v in wins[u] and wins[v].issubset(wins[u])
                              for u in range(n) if u != v) for v in range(n)])


def random_q(n, seed):
    rng = np.random.default_rng(seed)
    q = np.full((n, n), .5)
    i, j = np.triu_indices(n, 1)
    q[i, j] = rng.uniform(.03, .97, len(i))
    q[j, i] = 1 - q[i, j]
    return q


def enumerate_joint(q):
    """Full 2^(n choose 2) independent oracle used only at n <= 5."""
    n = len(q)
    i, j = np.triu_indices(n, 1)
    mean = np.zeros(n)
    joint = np.zeros((n, n + 1))
    cardinality = np.zeros(n + 1)
    conditional = [{} for _ in range(n)]
    for bits in itertools.product((False, True), repeat=len(i)):
        bits = np.asarray(bits, dtype=bool)
        mass = np.where(bits, q[i, j], 1 - q[i, j]).prod()
        adjacency = np.zeros((n, n), dtype=bool)
        adjacency[i, j] = bits
        adjacency[j, i] = ~bits
        labels = independent_set_uc(adjacency)
        assert_array_equal(exact_uc(adjacency), labels)
        mean += mass * labels
        size = int(labels.sum())
        joint[:, size] += mass * labels
        cardinality[size] += mass
        for v in range(n):
            key = tuple(np.flatnonzero(adjacency[v]))
            if key not in conditional[v]:
                conditional[v][key] = [0.0, 0.0]
            conditional[v][key][0] += mass
            conditional[v][key][1] += mass * labels[v]
    rb_variance = np.asarray([
        sum(num * num / den for den, num in conditions.values() if den > 0) - mean[v] ** 2
        for v, conditions in enumerate(conditional)
    ])
    return mean, joint, cardinality, rb_variance


def expected_f1(prediction, memberships, probabilities):
    numerator = 2 * (memberships & prediction).sum(1)
    denominator = memberships.sum(1) + prediction.sum()
    values = np.divide(numerator, denominator, out=np.ones(len(memberships), dtype=float),
                       where=denominator > 0)
    return float(probabilities @ values)


class PosteriorValidationTests(unittest.TestCase):
    def test_probabilities_reject_invalid_inputs_without_repair(self):
        for q in (np.zeros((2, 3)), np.array([[.5, .8], [.3, .5]]),
                  np.array([[.5, np.nan], [0., .5]]),
                  np.array([[.5, 1.01], [-.01, .5]])):
            with self.subTest(q=q), self.assertRaises(ValueError):
                validate_q(q)
        q = np.array([[.5, 1.], [0., .5]])
        assert_array_equal(validate_q(q), q)
        for tolerance in (np.nan, np.inf, -1.):
            with self.assertRaises(ValueError):
                validate_q(q, atol=tolerance)

    def test_float32_reciprocity_tolerates_backend_roundoff_only(self):
        q = np.array([[.5, .8], [.20000003, .5]], dtype=np.float32)
        self.assertGreater(abs(q.astype(float)[0, 1] + q.astype(float)[1, 0] - 1), 1e-10)
        assert_array_equal(validate_q(q), q.astype(float))
        bad = q.copy()
        bad[1, 0] = .201
        with self.assertRaises(ValueError):
            validate_q(bad)

    def test_jeffreys_orientation_differs_from_latent_mean(self):
        wins = np.array([[0., 1.], [0., 0.]])
        q = jeffreys_orientation_probabilities(wins)
        assert_allclose(q[0, 1], .5 + 1 / np.pi, atol=1e-14)
        self.assertEqual(jeffreys_pair_means(wins)[0, 1], .75)
        assert_allclose(q + q.T, np.ones((2, 2)))
        assert_array_equal(jeffreys_orientation_probabilities(np.zeros((3, 3))), np.full((3, 3), .5))
        for bad in (np.array([[1., 0.], [0., 0.]]),
                    np.array([[0., -1.], [0., 0.]]),
                    np.array([[0., np.inf], [0., 0.]])):
            with self.assertRaises(ValueError):
                jeffreys_orientation_probabilities(bad)
        batch = np.stack([wins, wins.T])
        assert_allclose(jeffreys_orientation_probabilities(batch)[1], q.T)

    def test_small_beta_orientation_tails_survive_relabeling(self):
        for opposing_wins, tail in ((50., 6.967610398722273e-17),
                                    (100., 4.412536991751157e-32)):
            wins = np.array([[0., 0.], [opposing_wins, 0.]])
            q = jeffreys_orientation_probabilities(wins)
            self.assertGreater(q[0, 1], 0)
            assert_allclose(q[0, 1], tail, rtol=2e-14, atol=0)
            assert_array_equal(q.T, jeffreys_orientation_probabilities(wins.T))

    def test_uc_validates_strict_tournaments(self):
        for bad in (np.zeros((3, 3), dtype=bool), np.eye(3, dtype=bool),
                    np.array([[False, True], [True, False]]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                exact_uc(bad)
        transitive = np.triu(np.ones((4, 4), dtype=bool), 1)
        assert_array_equal(exact_uc(transitive), [True, False, False, False])
        cycle = np.array([[False, True, False], [False, False, True], [True, False, False]])
        assert_array_equal(exact_uc(cycle), [True, True, True])
        assert_array_equal(exact_uc(np.stack([cycle, cycle.T])), np.ones((2, 3), dtype=bool))


class ExactSubsetTests(unittest.TestCase):
    def test_uniform_n3_is_posterior_half_versus_plugin_five_eighths(self):
        q = np.full((3, 3), .5)
        assert_allclose(exact_uc_marginals(q), .5, atol=1e-15)
        assert_allclose(soft_uc_plugin(q), 5 / 8, atol=1e-15)
        moments = exact_uc_moments(q)
        assert_allclose(moments["rb_variance"], 1 / 8)
        assert_allclose(moments["bernoulli_variance"], 1 / 4)

    def test_arbitrary_n3_through_n5_match_full_enumeration(self):
        for n in (3, 4, 5):
            for seed in (71, 143, 404):
                with self.subTest(n=n, seed=seed):
                    q = random_q(n, seed)
                    mean, _, _, variance = enumerate_joint(q)
                    moments = exact_uc_moments(q, subset_batch_size=3)
                    assert_allclose(moments["mean"], mean, atol=8e-15, rtol=0)
                    assert_allclose(moments["rb_variance"], variance, atol=8e-15, rtol=0)
                    self.assertTrue(np.all(moments["rb_variance"] <= moments["bernoulli_variance"] + 1e-14))

    def test_batch_singleton_and_deterministic_edges(self):
        assert_allclose(exact_uc_marginals(np.array([[.5]])), [1.])
        for n in (2, 3, 5):
            adjacency = np.triu(np.ones((n, n), dtype=bool), 1)
            q = adjacency.astype(float)
            np.fill_diagonal(q, .5)
            moments = exact_uc_moments(q)
            assert_allclose(moments["mean"], exact_uc(adjacency))
            assert_allclose(moments["rb_variance"], 0.)
        batch = np.stack([random_q(4, 1), random_q(4, 2)])
        assert_allclose(exact_uc_marginals(batch), np.stack([exact_uc_marginals(q) for q in batch]))
        with self.assertRaises(ValueError):
            exact_uc_marginals(np.full((17, 17), .5))

    def test_plugin_matches_independent_normalized_lse(self):
        q = random_q(5, 114)
        tau, gamma = .081, .057
        d = expit((q - .5) / tau)
        np.fill_diagonal(d, 0)
        maximum = lambda values: gamma * (logsumexp(np.asarray(values) / gamma) - np.log(len(values)))
        reference = []
        for v in range(5):
            covers = [d[u, v] * (1 - maximum([d[v, w] * (1 - d[u, w])
                                             for w in range(5) if w not in (v, u)]))
                      for u in range(5) if u != v]
            reference.append(1 - maximum(covers))
        assert_allclose(soft_uc_plugin(q, tau=tau, gamma=gamma), reference, atol=1e-15)
        assert_allclose(soft_uc_plugin(np.array([[.5]])), [1.])
        with self.assertRaises(ValueError):
            soft_uc_plugin(q, gamma=0)


class MonteCarloAndDecisionTests(unittest.TestCase):
    def test_rb_unbiased_with_correct_variance_and_empirical_moments(self):
        q = random_q(5, 441)
        exact = exact_uc_moments(q)
        result = rb_uc_marginals(q, 25000, seed=813, batch_size=377, keep_samples=True)
        assert_allclose(result.marginals, result.samples.mean(0), atol=2e-14)
        assert_allclose(result.variance, result.samples.var(0, ddof=1), atol=2e-14)
        assert_allclose(result.standard_errors ** 2, result.variance / result.draws, atol=1e-16)
        self.assertTrue(np.all(np.abs(result.marginals - exact["mean"]) < 6 * result.standard_errors + 1e-12))
        assert_allclose(result.variance, exact["rb_variance"], atol=.004, rtol=0)

    def test_centered_rb_variance_near_deterministic_outcomes(self):
        q = np.full((3, 3), .5)
        q[0, 1], q[1, 0] = 1., 0.
        epsilon = 1e-9
        q[2, 1], q[1, 2] = epsilon, 1 - epsilon
        moments = exact_uc_moments(q)
        assert_allclose(moments["rb_variance"][0], epsilon ** 2 / 4, rtol=1e-6, atol=0)
        result = rb_uc_marginals(q, 10000, seed=11, keep_samples=True)
        self.assertGreater(result.variance[0], 0)
        assert_allclose(result.variance[0], result.samples[:, 0].var(ddof=1), rtol=1e-5, atol=0)

    def test_native_joint_memberships_save_and_replay(self):
        q = random_q(5, 508)
        exact_mean, _, _, _ = enumerate_joint(q)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "memberships.npz"
            result = joint_uc_monte_carlo(q, 18000, seed=876, batch_size=457,
                                          keep_memberships=True, membership_path=saved)
            self.assertTrue(np.all(np.abs(result.marginals - exact_mean) < 6 * result.standard_errors + 1e-12))
            with np.load(saved) as replay:
                assert_array_equal(replay["memberships"], result.memberships)
                assert_array_equal(replay["q"], q)
                self.assertEqual(int(replay["seed"]), 876)
            assert_allclose(result.marginals, result.memberships.mean(0))
            assert_allclose(result.variance, result.memberships.var(0, ddof=1))
            assert_allclose(result.membership_size_joint.sum(1), result.marginals)
            sizes = np.arange(6)
            assert_allclose(result.membership_size_joint.sum(0), sizes * result.cardinality_probabilities)
            self.assertEqual(result.cardinality_probabilities[0], 0)
            assert_array_equal(result.half_decision, result.marginals >= .5)
            replay_value = expected_f1(result.gfm_decision, result.memberships,
                                       np.full(result.draws, 1 / result.draws))
            self.assertAlmostEqual(result.gfm_expected_f1, replay_value, places=14)
        repeated = joint_uc_monte_carlo(q, 18000, seed=876, batch_size=1000)
        assert_array_equal(repeated.membership_size_joint, result.membership_size_joint)

    def test_gfm_matches_bruteforce_expected_f1_for_joint_distribution(self):
        n = 4
        memberships = np.asarray(list(itertools.product((False, True), repeat=n)))
        probabilities = np.random.default_rng(89).dirichlet(np.ones(2 ** n))
        joint = np.zeros((n, n + 1))
        card = np.zeros(n + 1)
        for row, probability in zip(memberships, probabilities):
            joint[:, row.sum()] += probability * row
            card[row.sum()] += probability
        prediction, value = gfm_decision_from_joint(joint, card)
        values = [expected_f1(candidate, memberships, probabilities) for candidate in memberships]
        self.assertAlmostEqual(value, max(values), places=14)
        self.assertAlmostEqual(expected_f1(prediction, memberships, probabilities), max(values), places=14)
        inferred_prediction, inferred_value = gfm_decision_from_joint(joint)
        assert_array_equal(prediction, inferred_prediction)
        self.assertAlmostEqual(value, inferred_value, places=14)

    def test_same_marginals_can_need_different_gfm_decisions(self):
        singleton_joint = np.array([[0., .5, 0.], [0., .5, 0.]])
        empty_or_full_joint = np.array([[0., 0., .5], [0., 0., .5]])
        assert_allclose(singleton_joint.sum(1), empty_or_full_joint.sum(1))
        first, first_value = gfm_decision_from_joint(singleton_joint)
        second, second_value = gfm_decision_from_joint(empty_or_full_joint)
        assert_array_equal(first, [True, True])
        assert_array_equal(second, [False, False])
        self.assertAlmostEqual(first_value, 2 / 3)
        self.assertEqual(second_value, .5)
        assert_array_equal(marginal_half_decision([.499, .5, .501]), [False, True, True])

    def test_gfm_roundoff_ties_prefer_smaller_size(self):
        joint = np.zeros((4, 5))
        joint[0, 1] = .3
        joint[:, 4] = .3
        prediction, value = gfm_decision_from_joint(joint)
        assert_array_equal(prediction, [True, False, False, False])
        self.assertAlmostEqual(value, .42, places=15)

    def test_sampling_strictness_boundaries_and_draw_guards(self):
        q = random_q(4, 634)
        samples = sample_tournaments(q, 123, seed=59)
        self.assertEqual(samples.dtype, np.bool_)
        self.assertTrue(np.all((samples ^ samples.transpose(0, 2, 1))[:, ~np.eye(4, dtype=bool)]))
        self.assertFalse(np.diagonal(samples, axis1=1, axis2=2).any())
        assert_array_equal(samples, sample_tournaments(q, 123, seed=59))
        for method in (rb_uc_marginals, joint_uc_monte_carlo):
            for draws in (True, 0, 1, 2.5):
                with self.assertRaises(ValueError):
                    method(q, draws)
        with self.assertRaises(ValueError):
            gfm_decision_from_joint(np.ones((3, 3)))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional torch dependency is unavailable")
class TorchKernelTests(unittest.TestCase):
    def test_cpu_oracle_and_exact_parity(self):
        import torch
        samples = sample_tournaments(random_q(5, 649), 20, seed=12)
        assert_array_equal(exact_uc(torch.as_tensor(samples)).numpy(), exact_uc(samples))
        q = np.stack([random_q(5, 4), random_q(5, 5)])
        tensor = torch.as_tensor(q, dtype=torch.float64)
        assert_allclose(exact_uc_marginals_torch(tensor).numpy(), exact_uc_marginals(q), atol=1e-14)
        assert_allclose(soft_uc_plugin(tensor).numpy(), soft_uc_plugin(q), atol=1e-14)
        assert_allclose(soft_uc_plugin(torch.tensor([[.5]])).numpy(), [1.])

    def test_endpoint_small_n_gradients_and_invalid_tensors(self):
        import torch
        singleton = torch.tensor([[.5]], dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(exact_uc_marginals_torch(singleton).sum(), singleton)
        assert_allclose(gradient.numpy(), 0.)
        for value in (0., 1.):
            edge = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            q = torch.stack([torch.stack([edge * 0 + .5, edge]),
                             torch.stack([1 - edge, edge * 0 + .5])])
            score = exact_uc_marginals_torch(q)
            gradient, = torch.autograd.grad(.7 * score[0] + 1.3 * score[1], edge)
            self.assertAlmostEqual(float(gradient), -.6, places=14)
        for bad in (torch.tensor([[0, 1], [0, 0]]),
                    torch.tensor([[.5, .8], [.3, .5]]),
                    torch.tensor([[.5, float("nan")], [0., .5]])):
            with self.assertRaises(ValueError):
                exact_uc_marginals_torch(bad)

    def test_differentiable_reciprocal_formula_finite_differences_n3_to_n10(self):
        import torch
        # n=10 exercises the documented upper resource guard; four directions
        # at each n keep this diagnostic small while testing reciprocal inputs.
        for n in (3, 4, 5, 10):
            i, j = np.triu_indices(n, 1)
            values = random_q(n, 271 + n)[i, j]
            upper = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            q = torch.full((n, n), .5, dtype=torch.float64)
            q[i, j] = upper
            q[j, i] = 1 - upper
            weights = np.linspace(.7, 1.3, n)
            objective = (exact_uc_marginals_torch(q) * torch.as_tensor(weights)).sum()
            gradient, = torch.autograd.grad(objective, upper)
            for index in np.linspace(0, len(i) - 1, min(4, len(i)), dtype=int):
                perturbation = np.zeros(len(i))
                perturbation[index] = 1e-6
                outputs = []
                for shifted in (values + perturbation, values - perturbation):
                    matrix = np.full((n, n), .5)
                    matrix[i, j] = shifted
                    matrix[j, i] = 1 - shifted
                    outputs.append(weights @ exact_uc_marginals(matrix))
                difference = (outputs[0] - outputs[1]) / 2e-6
                self.assertAlmostEqual(float(gradient[index]), float(difference), delta=2e-8)
        with self.assertRaises(ValueError):
            exact_uc_marginals_torch(torch.full((11, 11), .5))

    def test_torch_native_joint_cpu_samples(self):
        q = random_q(4, 522)
        result = joint_uc_monte_carlo(q, 1200, seed=82, device="cpu", keep_memberships=True)
        assert_allclose(result.marginals, result.memberships.mean(0))
        repeat = joint_uc_monte_carlo(q, 1200, seed=82, device="cpu", keep_memberships=True)
        assert_array_equal(result.memberships, repeat.memberships)


if __name__ == "__main__":
    unittest.main()
