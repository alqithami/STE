"""Parser, observed-ballot sampling, truth and fold-isolation software checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stepc.data import (SOURCE_ORDER, ballot_counts, fold_split, get_profile_cases,
                        nested_subsamples, observed_view, parse_profile, prepare_data,
                        source_folds, uncovered_set, uncovered_set_independent)

BASE = Path(__file__).resolve().parents[1]
ORDINAL_FIXTURE = """# DATA TYPE: toi
# NUMBER ALTERNATIVES: 4
# NUMBER VOTERS: 6
# NUMBER UNIQUE ORDERS: 3
# ALTERNATIVE NAME 1: A
# ALTERNATIVE NAME 2: B
# ALTERNATIVE NAME 3: C
# ALTERNATIVE NAME 4: D
3: {1,2},3
2: 3,2,1
1: 4,1
"""


class ParserTests(unittest.TestCase):
    def setUp(self):
        # This handwritten parser fixture is never an experiment input.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "parser_test.toi"
        self.path.write_text(ORDINAL_FIXTURE)

    def test_ties_omissions_and_weighted_counts(self):
        profile = parse_profile(self.path)
        self.assertEqual(profile["multiplicity"].tolist(), [3, 2, 1])
        self.assertEqual(profile["ranks"].tolist(), [[0, 0, 1, -1], [2, 1, 0, -1], [1, -1, -1, 0]])
        W, T = ballot_counts(profile, profile["multiplicity"])
        self.assertEqual(W.tolist(), [[0, 0, 3, 0], [2, 0, 3, 0], [2, 2, 0, 0], [1, 0, 0, 0]])
        expected_ties = np.zeros((4, 4), dtype=np.int64)
        expected_ties[0, 1] = expected_ties[1, 0] = 3
        np.testing.assert_array_equal(T, expected_ties)
        # D was omitted in five votes; those votes give D no pair losses.
        self.assertEqual(int(W[:, 3].sum()), 0)

    def test_invalid_ballots_rejected(self):
        for replacement in ("3: {1,1},3", "3: {1,5},3", "3: {1,2},oops", "0: {1,2},3"):
            with self.subTest(replacement=replacement):
                self.path.write_text(ORDINAL_FIXTURE.replace("3: {1,2},3", replacement))
                with self.assertRaises(ValueError):
                    parse_profile(self.path)

    def test_count_and_sample_validation(self):
        profile = parse_profile(self.path)
        for bad in (np.array([-1, 2, 1]), np.array([4, 2, 1]), np.array([3.0, 2.0, 1.0])):
            with self.subTest(multiplicity=bad):
                with self.assertRaises(ValueError):
                    ballot_counts(profile, bad)
        for fractions in ([0.1, 0.1], [1.1]):
            with self.assertRaises(ValueError):
                nested_subsamples(profile, fractions, np.random.default_rng(0))

    def test_real_samples_are_nested_observed_voters(self):
        profile = parse_profile(BASE / "reference/data/profiles/00007-00000068.soi")
        first = nested_subsamples(profile, [0.05, 0.1, 0.25], np.random.default_rng(20261007))
        again = nested_subsamples(profile, [0.25, 0.05, 0.1], np.random.default_rng(20261007))
        previous = np.zeros_like(profile["multiplicity"])
        for fraction, (W, T, mult) in first.items():
            self.assertEqual(int(mult.sum()), max(1, int(np.floor(fraction * profile["voters"]))))
            self.assertTrue(np.all(previous <= mult))
            self.assertTrue(np.all(mult <= profile["multiplicity"]))
            for a, b in zip((W, T, mult), again[fraction]):
                np.testing.assert_array_equal(a, b)
            for a, b in zip((W, T), ballot_counts(profile, mult)):
                np.testing.assert_array_equal(a, b)
            previous = mult

    def test_uc_oracles_agree_on_every_small_tournament(self):
        for n in range(1, 5):
            ii, jj = np.triu_indices(n, 1)
            for bits in range(1 << len(ii)):
                A = np.zeros((n, n), dtype=bool)
                for edge, (a, b) in enumerate(zip(ii, jj)):
                    A[a, b] = bool((bits >> edge) & 1)
                    A[b, a] = not A[a, b]
                np.testing.assert_array_equal(uncovered_set(A), uncovered_set_independent(A))
                kings = (A | (A.astype(int) @ A.astype(int) > 0) | np.eye(n, dtype=bool)).all(axis=1)
                np.testing.assert_array_equal(uncovered_set(A), kings)


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = get_profile_cases(BASE, {"seed": 20261007, "data": {"fractions": [0.05, 0.1, 0.25], "resamples": 8}})

    def test_fixed_cohort_composition_and_all_samples(self):
        self.assertEqual(len(self.cohort), 36 * 8 * 3)
        first_case = {case["profile"]: case for case in self.cohort}
        self.assertEqual(len(first_case), 36)
        self.assertEqual(sum(case["uc_size"] == 1 for case in first_case.values()), 33)
        self.assertEqual(sum(case["core_class"] == "selective_non_singleton" for case in first_case.values()), 3)
        self.assertEqual(tuple(dict.fromkeys(case["source"] for case in self.cohort)), SOURCE_ORDER)
        for case in self.cohort:
            np.testing.assert_array_equal(case["fullA"], case["fullW"] > case["fullW"].T)
            np.testing.assert_array_equal(case["y"], uncovered_set_independent(case["fullA"]))
            self.assertEqual(int(case["sampled_multiplicity"].sum()), case["voters_sampled"])

    def test_source_folds_and_prediction_reference_isolation(self):
        for index, fold in enumerate(source_folds()):
            self.assertEqual(fold["test_source"], SOURCE_ORDER[index])
            self.assertEqual(fold["dev_source"], SOURCE_ORDER[(index + 1) % 6])
            self.assertEqual(len(fold["train_sources"]), 4)
            split = fold_split(self.cohort, fold)
            self.assertEqual({case["source"] for case in split["train"]}, set(fold["train_sources"]))
            self.assertEqual({case["source"] for case in split["dev"]}, {fold["dev_source"]})
            self.assertEqual({case["source"] for case in split["test"]}, {fold["test_source"]})
            for case in split["test"]:
                self.assertFalse(any(key in case for key in ("y", "fullA", "fullW", "fullT", "fulltruthUC", "uc_size", "core_class")))
                self.assertIn(case["case_id"], split["test_truth"])
        self.assertIs(observed_view(self.cohort[0])["W"], self.cohort[0]["W"])

    def test_preparation_archives_and_resume_integrity(self):
        config = {"seed": 20261007, "data": {"fractions": [0.05, 0.1, 0.25], "resamples": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = prepare_data(BASE, config, tmp_path)
            self.assertEqual(len(bundle["cases"]), 108)
            self.assertTrue((tmp_path / "SOURCE_FOLDS.json").exists())
            self.assertTrue((tmp_path / "DATA_COUNTS_MANIFEST.csv").exists())
            first = bundle["manifest"]["profiles"][0]
            with np.load(tmp_path / first["counts_npz"], allow_pickle=False) as data:
                self.assertEqual(data["W"].shape[0], 3)
                self.assertTrue(np.all(data["sampled_multiplicity"] <= data["original_multiplicity"]))
                self.assertTrue(np.all(np.diff(data["sampled_multiplicity"], axis=0) >= 0))
                self.assertEqual(data["sampled_multiplicity"].sum(axis=1).tolist(), [48, 97, 244])
            again = prepare_data(BASE, config, tmp_path)
            self.assertEqual(again["manifest"], bundle["manifest"])
            with self.assertRaisesRegex(ValueError, "different sampling"):
                prepare_data(BASE, {"seed": 20261007, "data": {"resamples": 2}}, tmp_path)


if __name__ == "__main__":
    unittest.main()
