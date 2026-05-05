import numpy as np

from ste_neurips.neurips_suite import (
    make_planted_core_tournament,
    sample_counts,
    posterior_membership_scores,
    top_cycle_from_adj,
    uncovered_from_adj,
    majority_adjacency,
    method_scores,
)


def test_oracle_planted_core_tc_uc_match():
    T = make_planted_core_tournament(n=15, core_size=3, seed=123)
    assert np.array_equal(top_cycle_from_adj(T.A), T.true_tc)
    assert np.array_equal(uncovered_from_adj(T.A), T.true_uc)
    assert T.true_tc.sum() == 3
    assert T.true_uc.sum() == 3


def test_transitive_control_is_singleton():
    T = make_planted_core_tournament(n=12, core_size=1, seed=7, mode="transitive")
    assert T.true_tc.sum() == 1
    assert T.true_uc.sum() == 1


def test_posterior_membership_scores_are_bounded():
    T = make_planted_core_tournament(n=12, core_size=3, seed=8)
    wins, comps = sample_counts(T.P, m_per_pair=5, missing_rate=0.1, label_noise=0.02, seed=9)
    scores = posterior_membership_scores(wins, comps, solution="uc", samples=10, seed=10)
    assert scores.shape == (12,)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_method_aliases_return_scores():
    T = make_planted_core_tournament(n=12, core_size=3, seed=11)
    wins, comps = sample_counts(T.P, m_per_pair=5, missing_rate=0.0, label_noise=0.0, seed=12)
    for method in ["ste_posterior_edge_uc", "ste_plugin_uc", "hard_uc", "winrate", "btl", "copeland"]:
        scores, meta = method_scores(method, wins, comps, K=11, seed=13, posterior_samples=10)
        assert len(scores) == 12
        assert np.all(np.isfinite(scores))


def test_condorcet_style_baselines_return_scores():
    import numpy as np
    from ste_neurips.neurips_suite import method_scores
    wins = np.array([
        [0, 5, 5],
        [1, 0, 5],
        [1, 1, 0],
    ], dtype=float)
    comps = wins + wins.T
    for method in ["schulze", "minimax", "ranked_pairs", "kemeny_local"]:
        scores, _ = method_scores(method, wins, comps, seed=0)
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))
        assert int(np.argmax(scores)) == 0
