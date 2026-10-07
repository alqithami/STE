"""Regressions for v1's self-loop bug and corrected future-run boundaries."""
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_experiment import (partial_relation_uc, draw_memberships,
                            joint_from_memberships, normalized_config,
                            validate_future_output_path, source_hashes, HERE)
from stepc.data import seed_for
from stepc.graphs import hard_orientation_relation, strict_top_cycle
from stepc.posterior import mixture_orientation_probabilities, exact_uc

try:
    import torch
    from stepc.learning import (mixture_orientation_probabilities_torch,
                                PredictiveModel, supervised_loss, hard_uc_torch)
except ImportError:
    torch = None


def fixtures():
    q = np.full((3, 3), .5, dtype=np.float32)
    i, j = np.triu_indices(3, 1)
    q[i, j], q[j, i] = 1., 0.
    components = np.stack((q, q))
    # An explicit, deterministic float32 roundoff example; no random search.
    weights = np.array([.8, .20000012], dtype=np.float32)
    return components, weights


def legacy_sampler_reference(components, weights, draws, seed):
    """Independent reconstruction of the v1 native draw law for bit parity.

    This reference does not reconstruct the faulty v1 hard-q endpoint. Its
    component and edge streams/upper-triangle law are the preserved v1 sampler.
    """
    components = np.asarray(components, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = components.shape[-1]
    i, j = np.triu_indices(n, 1)
    rng = np.random.default_rng(seed)
    choices = np.random.default_rng(seed_for(seed, "global-components")).choice(
        len(weights), size=draws, p=weights / weights.sum())
    result = []
    for start in range(0, draws, 256):
        ids = choices[start:start+256]
        edges = rng.random((len(ids), len(i))) < components[ids][:, i, j]
        graphs = np.zeros((len(ids), n, n), dtype=bool)
        graphs[:, i, j], graphs[:, j, i] = edges, ~edges
        # Independent set-inclusion covering oracle on strict samples.
        for graph in graphs:
            wins = [set(np.flatnonzero(row)) for row in graph]
            result.append([not any(v in wins[u] and wins[v] <= wins[u]
                                   for u in range(n) if u != v)
                           for v in range(n)])
    return np.asarray(result, dtype=bool), choices


class CorrectedGraphBoundaryTests(unittest.TestCase):
    def test_float32_weights_reproduce_self_loop_empty_bug(self):
        components, weights = fixtures()
        self.assertGreater(float(weights.sum()), 1.)
        old_q = (weights[:, None, None] * components).sum(0)
        self.assertTrue((np.diag(old_q) > .5).all())
        self.assertGreater(float(old_q.max()), 1.)
        old_A = old_q > .5
        missing = (old_A[None, :, :] & ~old_A[:, None, :]).any(-1)
        old_cover = old_A & ~missing
        self.assertFalse((~old_cover.any(0)).any())
        with self.assertRaisesRegex(ValueError, "self-loops"):
            partial_relation_uc(old_A)
        corrected = mixture_orientation_probabilities(components, weights)
        self.assertTrue(np.array_equal(np.diag(corrected), np.full(3, .5)))
        self.assertTrue(np.array_equal(corrected + corrected.T, np.ones((3, 3))))
        self.assertTrue(((corrected >= 0) & (corrected <= 1)).all())
        np.testing.assert_array_equal(partial_relation_uc(hard_orientation_relation(corrected)),
                                      [True, False, False])

    def test_invalid_relations_rejected_and_probability_ties_preserved(self):
        for invalid in (np.eye(3, dtype=bool),
                        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)):
            for oracle in (partial_relation_uc, exact_uc, strict_top_cycle):
                with self.assertRaises(ValueError):
                    oracle(invalid)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            partial_relation_uc(np.zeros((3, 3)))
        q = np.full((3, 3), .5)
        np.fill_diagonal(q, .75)  # Decoder clears diagonal explicitly.
        A = hard_orientation_relation(q)
        self.assertFalse(A.any())
        self.assertTrue(partial_relation_uc(A).all())
        # Tolerance in q validation cannot excuse a bidirectional graph.
        roundoff_bidirectional = np.full((2, 2), np.float32(.50000006), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            hard_orientation_relation(roundoff_bidirectional)

    def test_exhaustive_small_tournaments_uc_and_tc_independent_oracles(self):
        for n in (1, 2, 3, 4):
            i, j = np.triu_indices(n, 1)
            for bits in itertools.product((False, True), repeat=len(i)):
                A = np.zeros((n, n), dtype=bool)
                A[i, j], A[j, i] = bits, ~np.asarray(bits, dtype=bool)
                np.testing.assert_array_equal(partial_relation_uc(A), exact_uc(A))
                expected = []
                for vertex in range(n):
                    visited, stack = {vertex}, [vertex]
                    while stack:
                        for neighbor in set(np.flatnonzero(A[stack.pop()])) - visited:
                            visited.add(neighbor); stack.append(neighbor)
                    expected.append(len(visited) == n)
                np.testing.assert_array_equal(strict_top_cycle(A), expected)

    def test_native_samples_joint_and_gfm_unchanged_by_marginal_repair(self):
        rng = np.random.default_rng(8107)
        for n in (3, 6):
            components = rng.random((2, n, n)).astype(np.float32)
            i, j = np.triu_indices(n, 1)
            components[:, j, i] = 1. - components[:, i, j]
            for component in components:
                np.fill_diagonal(component, .5)
            weights = np.array([.8, .20000012], dtype=np.float32)
            q = mixture_orientation_probabilities(components, weights)
            expected, expected_ids = legacy_sampler_reference(components, weights, 513, 73)
            actual, actual_ids = draw_memberships(q, 513, 73, components, weights)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_ids, expected_ids)
            a, b = joint_from_memberships(actual), joint_from_memberships(expected)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_weight_scaling_probability_bounds_and_invalid_inputs(self):
        components, weights = fixtures()
        q = mixture_orientation_probabilities(components, weights)
        np.testing.assert_allclose(mixture_orientation_probabilities(components, weights * 7), q,
                                   atol=1e-15, rtol=0)
        for invalid in ([0, 0], [-1, 2], [np.nan, 1], [1], [np.inf, 1], [1e308,1e308]):
            with self.assertRaises(ValueError):
                mixture_orientation_probabilities(components, invalid)
        invalid = components.copy(); invalid[0, 0, 1] = 1.1
        with self.assertRaises(ValueError):
            mixture_orientation_probabilities(invalid, weights)

    def test_immutable_parent_namespace_and_version_are_rejected(self):
        import json
        cfg = json.loads((HERE / "config.json").read_text())
        self.assertEqual(normalized_config(cfg, True, False)["release"],
                         "STE-Posterior-Certificates-v2")
        cfg["release"] = "STE-Posterior-Certificates-v1"
        with self.assertRaisesRegex(ValueError, "Config release"):
            normalized_config(cfg, True, False)
        for out in ("/home/ubuntu/ste-posterior-certificates-v1/production",
                    "/home/ubuntu/STE_Posterior_Certificates_v1/results"):
            with self.assertRaisesRegex(ValueError, "frozen v1"):
                validate_future_output_path(out)
        hashes = source_hashes()
        self.assertIn("stepc/graphs.py", hashes)
        self.assertIn("CORRECTION_NOTICE.md", hashes)
        self.assertIn("run_all.sh", hashes)

    def test_wrapper_refuses_existing_base_before_any_marker_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'v2-existing-evidence'
            base.mkdir()
            marker=base/'STATUS.json'
            marker.write_bytes(b'preserve this exact existing status')
            env={**os.environ,'STE_PYTHON':sys.executable,'STE_RESULTS_BASE':str(base),
                 'PYTHONDONTWRITEBYTECODE':'1'}
            result=subprocess.run(['bash',str(HERE/'run_all.sh'),'smoke'],env=env,
                                  capture_output=True,text=True,check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Existing/nonempty result base refused',result.stderr)
            self.assertEqual(marker.read_bytes(),b'preserve this exact existing status')
            self.assertEqual(list(base.iterdir()),[marker])


@unittest.skipIf(torch is None, "PyTorch unavailable; Torch QA must not be inferred")
class TorchCorrectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_float32_case_matches_normalized_numpy_sampling_law(self):
        components, weights = fixtures()
        c = torch.tensor(components, requires_grad=True)
        w = torch.tensor(weights, requires_grad=True)
        q = mixture_orientation_probabilities_torch(c, w)
        np.testing.assert_array_equal(q.detach().numpy(),
                                      mixture_orientation_probabilities(components, weights))
        self.assertTrue(torch.equal(torch.diag(q), torch.full((3,), .5, dtype=torch.float64)))
        self.assertTrue(torch.equal(q + q.T, torch.ones((3, 3), dtype=torch.float64)))
        q[0, 1].backward()
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_probabilities_and_gradients_for_random_and_saturated_components(self):
        for dtype in (torch.float32, torch.float64):
            for scale in (1., 100.):
                torch.manual_seed(19)
                logits = (torch.randn((2, 5, 5), dtype=dtype) * scale).requires_grad_()
                probabilities = torch.sigmoid(logits - logits.transpose(-1, -2))
                mix_logits = torch.tensor([.2, -.7], dtype=dtype, requires_grad=True)
                weights = torch.log_softmax(mix_logits, 0).exp()
                q = mixture_orientation_probabilities_torch(probabilities, weights)
                np.testing.assert_allclose(q.detach().numpy(),
                    mixture_orientation_probabilities(probabilities.detach().numpy(), weights.detach().numpy()),
                    atol=1e-15, rtol=0)
                self.assertTrue(((q >= 0) & (q <= 1)).all())
                self.assertTrue(torch.equal(q + q.T, torch.ones_like(q)))
                q[0, 1].backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertTrue(torch.isfinite(mix_logits.grad).all())

    def test_torch_graph_rejects_self_loops_and_bidirectional_edges(self):
        for invalid in (torch.eye(3, dtype=torch.bool),
                        torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)):
            with self.assertRaises(ValueError):
                hard_uc_torch(invalid)

    def test_existing_audit_marker_is_never_overwritten(self):
        from audit_results import audit
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            marker=out/'AUDIT.json'
            marker.write_bytes(b'preserve original audit')
            with self.assertRaisesRegex(RuntimeError,'Existing audit evidence refused'):
                audit(out)
            with self.assertRaisesRegex(ValueError,'frozen v1'):
                audit(out,replay_report='/home/ubuntu/ste-posterior-certificates-v1/external-review.json')
            self.assertEqual(marker.read_bytes(),b'preserve original audit')

    def test_corrected_supervised_loss_has_finite_usable_gradients(self):
        W = np.array([[0, 4, 1], [1, 0, 7], [5, 1, 0]])
        case = {"W": W, "T": np.zeros_like(W), "y": np.array([1, 0, 1]),
                "fullA": W > W.T}
        for kind in ("learned_edge", "posterior_mixture"):
            torch.manual_seed(73)
            model = PredictiveModel(kind, 8)
            loss, _ = supervised_loss(model, case, "cpu", weight=.1, draws=8)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)


if __name__ == "__main__":
    unittest.main()
