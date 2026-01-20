#!/usr/bin/env python3
"""YAML-driven experiment runner for the STE IJCAI'26 paper.

This runner is designed to be:
- honest / non-placeholder: every metric is computed from actual data
  (synthetic simulations or real-world datasets you provide).
- reproducible: each run writes a meta.json and a copy of the config.
- paper-oriented: produces raw CSVs + paper-ready figures/tables.

Key principle: \hat{P} estimation is *explicit*.
Pairwise comparison data must be converted into a dense probabilistic tournament
matrix \hat{P}. This pipeline supports multiple estimators (empirical counts
or learned BTL), controlled from YAML.

Usage (from the ste/ directory):
  python run_all.py --config configs/ste_master.yaml --experiment all
  python run_all.py --experiment core_recovery --quick

Outputs:
  <output_root>/runs/<timestamp>/*.csv
  <output_root>/runs/<timestamp>/meta.json
  <output_root>/paper_assets/tables/*.tex
  <output_root>/paper_assets/figs/*.pdf

Important:
- "--quick" is for smoke tests only; it reduces seeds/bootstraps/epochs.
- Real-world experiments are OFF by default and will error if input files are missing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from configs.config_loader import load_yaml_config
from data.synthetic import generate_synthetic_tournament
from data.arena import load_chatbot_arena_pairwise
from data.agentbench import load_agentbench_pairwise

from prob_estimation import (
    split_comparisons,
    phat_from_empirical,
    fit_learned_btl,
    phat_from_scores,
    predictive_logloss,
    predictive_accuracy,
)

from ste_ops.ste import compute_ste_scores, top_cycle_membership_prob
from baselines.methods import BaselineEvaluator, scores_to_ranking

from eval.core_metrics import (
    f1_score_core,
    jaccard_index,
    expected_calibration_error,
    brier_score,
    top_k_f1,
    top_1_in_core,
)
from eval.stability import bootstrap_ste, compute_stability_metrics
from plots.figures import generate_all_figures


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_phat(P_hat: np.ndarray, tol: float = 1e-6) -> None:
    P = np.asarray(P_hat, dtype=np.float64)
    if P.shape[0] != P.shape[1]:
        raise ValueError(f"P_hat must be square, got {P.shape}")
    if not np.allclose(np.diag(P), 0.5, atol=tol):
        raise ValueError("P_hat diagonal must be 0.5")
    if not np.allclose(P + P.T, 1.0, atol=1e-5):
        # Not fatal for all settings, but we enforce for cleanliness.
        raise ValueError("P_hat must satisfy P_hat[i,j] + P_hat[j,i] = 1")


def _get_prob_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    prob = cfg.get('prob_estimation', {})
    if not isinstance(prob, dict):
        prob = {}
    return prob


def _split_cfg(prob_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    split = prob_cfg.get('split', {}) if isinstance(prob_cfg.get('split', {}), dict) else {}
    train_frac = float(split.get('train', 0.8))
    val_frac = float(split.get('val', 0.1))
    test_frac = float(split.get('test', 0.1))
    return train_frac, val_frac, test_frac





def _ste_kwargs(cfg: Dict[str, Any], tau: float, K: int, alpha: float) -> Dict[str, Any]:
    # Collect STE kwargs from YAML config (no placeholders).
    ste_cfg = cfg.get('ste', {}) if isinstance(cfg.get('ste', {}), dict) else {}
    return {
        'tau': float(tau),
        'K': int(K),
        'alpha': float(alpha),
        'reachability_mode': str(ste_cfg.get('reachability_mode', 'max_product')),
        'tau_softmin': float(ste_cfg.get('tau_softmin', tau)),
        'uncovered_mode': str(ste_cfg.get('uncovered_mode', 'lukasiewicz')),
        'tau_imp': float(ste_cfg.get('tau_imp', tau)),
        'tau_violation': float(ste_cfg.get('tau_violation', tau)),
        'tau_witness': float(ste_cfg.get('tau_witness', tau)),
        'tau_coverer': float(ste_cfg.get('tau_coverer', tau)),
        'beta': float(ste_cfg.get('beta_uncovered', 5.0)),
    }

def _estimate_phat_for_run(
    comparisons: np.ndarray,
    n: int,
    seed: int,
    prob_cfg: Dict[str, Any],
    quick: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Estimate P_hat for STE + a baseline P_hat, with splits and predictive metrics.

    Returns:
      P_hat_ste: probability matrix used by STE
      P_hat_base: empirical (train-only) matrix used for baselines that need P_hat
      metrics: dict with predictive logloss/accuracy + (optional) learned_BTL stats
      splits: (train, val, test)
    """
    method = str(prob_cfg.get('method', 'empirical_train')).lower()
    laplace_alpha = float(prob_cfg.get('laplace_alpha', 1.0))
    fill_value = float(prob_cfg.get('fill_value', 0.5))

    train_frac, val_frac, test_frac = _split_cfg(prob_cfg)
    train, val, test = split_comparisons(comparisons, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)

    # Baseline P_hat is always empirical train-only (standard / non-leaky).
    P_hat_base = phat_from_empirical(train, n=n, laplace_alpha=laplace_alpha, fill_value=fill_value)

    metrics: Dict[str, float] = {}

    if method == 'empirical_all':
        P_hat_ste = phat_from_empirical(comparisons, n=n, laplace_alpha=laplace_alpha, fill_value=fill_value)
    elif method == 'empirical_train':
        P_hat_ste = P_hat_base
    elif method == 'learned_btl':
        train_cfg = prob_cfg.get('learned_btl', {}) if isinstance(prob_cfg.get('learned_btl', {}), dict) else {}

        # Quick mode: reduce epochs/iters aggressively
        max_epochs = int(train_cfg.get('max_epochs', 200))
        temp_max_iters = int(train_cfg.get('temp_max_iters', 200))
        if quick:
            max_epochs = min(max_epochs, 40)
            temp_max_iters = min(temp_max_iters, 60)

        res = fit_learned_btl(
            train_comparisons=train,
            val_comparisons=val,
            n=n,
            seed=seed,
            max_epochs=max_epochs,
            batch_size=int(train_cfg.get('batch_size', 512)),
            lr=float(train_cfg.get('lr', 0.05)),
            weight_decay=float(train_cfg.get('weight_decay', 0.0)),
            early_stop_patience=int(train_cfg.get('patience', 20)),
            calibrate_temperature=bool(train_cfg.get('calibrate_temperature', True)),
            temp_max_iters=temp_max_iters,
            temp_lr=float(train_cfg.get('temp_lr', 0.1)),
            device=train_cfg.get('device', None),
        )
        P_hat_ste = phat_from_scores(res.scores, temperature=res.temperature)
        metrics['learned_btl_temperature'] = float(res.temperature)
        metrics['learned_btl_train_nll'] = float(res.train_nll)
        metrics['learned_btl_val_nll'] = float(res.val_nll)
    else:
        raise ValueError(f"Unknown prob_estimation.method: {method}")

    _validate_phat(P_hat_ste)
    _validate_phat(P_hat_base)

    # Predictive metrics (evaluated on held-out test split).
    metrics['logloss_train'] = float(predictive_logloss(P_hat_ste, train))
    metrics['logloss_val'] = float(predictive_logloss(P_hat_ste, val))
    metrics['logloss_test'] = float(predictive_logloss(P_hat_ste, test))
    metrics['acc_test'] = float(predictive_accuracy(P_hat_ste, test, threshold=0.5))

    return P_hat_ste, P_hat_base, metrics, (train, val, test)


# -----------------------------------------------------------------------------
# Synthetic experiments
# -----------------------------------------------------------------------------


def run_core_recovery(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Core Recovery vs Cyclicity")

    n_values = exp.get('n_values', [20, 50])
    rho_values = exp.get('rho_values', [0.0, 0.2, 0.4, 0.6, 0.8])
    eta = float(exp.get('eta', 0.05))
    mu = float(exp.get('mu', 0.1))
    m_per_pair = int(exp.get('m_per_pair', 10))

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])
    alpha = float(cfg['ste'].get('alpha', 1.0))
    beta = float(cfg['ste'].get('beta_uncovered', 5.0))

    reachability_mode = str(cfg['ste'].get('reachability_mode', 'max_product'))
    tau_witness = float(cfg['ste'].get('tau_witness', tau))

    prob_cfg = _get_prob_cfg(cfg)

    results: List[Dict[str, Any]] = []

    total = len(n_values) * len(rho_values) * n_seeds
    k = 0

    for n in n_values:
        for rho in rho_values:
            for seed_idx in range(n_seeds):
                k += 1
                seed = int(cfg['reproducibility']['seed']) + seed_idx
                print(f"\r  Progress: {k}/{total} (n={n}, rho={rho}, seed={seed})", end="")

                tournament = generate_synthetic_tournament(
                    n=int(n),
                    rho=float(rho),
                    eta=eta,
                    mu=mu,
                    m_per_pair=m_per_pair,
                    cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                    btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                    num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                    cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                    cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                    seed=seed,
                )

                P_hat_ste, P_hat_base, pred_metrics, splits = _estimate_phat_for_run(
                    comparisons=tournament.comparisons,
                    n=int(n),
                    seed=seed,
                    prob_cfg=prob_cfg,
                    quick=quick,
                )

                P_t = torch.from_numpy(P_hat_ste).float()
                t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=alpha))

                tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
                uc_probs = u_tau.detach().cpu().numpy()

                tc_pred = (tc_probs > 0.5).astype(float)
                uc_pred = (uc_probs > 0.5).astype(float)

                # Ranking-style evaluation for STE (threshold-free diagnostic).
                # This is useful to detect degenerate thresholding (e.g., predict-all).
                tc_rank = scores_to_ranking(tc_probs)
                uc_rank = scores_to_ranking(uc_probs)

                row: Dict[str, Any] = {
                    'n': int(n),
                    'rho': float(rho),
                    'eta': float(eta),
                    'mu': float(mu),
                    'seed': int(seed),
                    'm_per_pair': int(m_per_pair),
                    'ste_tc_f1': f1_score_core(tc_pred, tournament.true_top_cycle),
                    'ste_tc_jaccard': jaccard_index(tc_pred, tournament.true_top_cycle),
                    'ste_uc_f1': f1_score_core(uc_pred, tournament.true_uncovered),
                    'ste_uc_jaccard': jaccard_index(uc_pred, tournament.true_uncovered),
                    'ste_tc_topk_f1': top_k_f1(tc_rank, tournament.true_top_cycle),
                    'ste_uc_topk_f1': top_k_f1(uc_rank, tournament.true_uncovered),
                    'ste_top1_tc': top_1_in_core(tc_rank, tournament.true_top_cycle),
                    'ste_tc_pred_size': float(tc_pred.sum()),
                    'ste_uc_pred_size': float(uc_pred.sum()),
                    'true_tc_size': float(tournament.true_top_cycle.sum()),
                    'true_uc_size': float(tournament.true_uncovered.sum()),
                    **pred_metrics,
                }

                # Baselines (use train-only empirical P_hat for P_hat-dependent baselines)
                train, _, _ = splits
                baseline_eval = BaselineEvaluator(elo_epochs=int(cfg.get('baselines', {}).get('elo_epochs', 1)))
                baseline_scores = baseline_eval.evaluate_all(P_hat_base, train)
                for method, scores in baseline_scores.items():
                    ranking = scores_to_ranking(scores)
                    row[f'{method}_tc_f1'] = top_k_f1(ranking, tournament.true_top_cycle)
                    row[f'{method}_uc_f1'] = top_k_f1(ranking, tournament.true_uncovered)
                    row[f'{method}_top1_tc'] = top_1_in_core(ranking, tournament.true_top_cycle)

                results.append(row)

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'core_recovery_raw.csv'), index=False)
    return df


def run_robustness(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Robustness to Noise and Sparsity")

    n = int(exp.get('n', 15))
    rho = float(exp.get('rho', 0.3))
    eta_values = exp.get('eta_values', [0.0, 0.05, 0.1])
    mu_values = exp.get('mu_values', [0.0, 0.1, 0.3])
    m_per_pair = int(exp.get('m_per_pair', 10))

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])
    alpha = float(cfg['ste'].get('alpha', 1.0))
    beta = float(cfg['ste'].get('beta_uncovered', 5.0))

    reachability_mode = str(cfg['ste'].get('reachability_mode', 'max_product'))
    tau_witness = float(cfg['ste'].get('tau_witness', tau))

    prob_cfg = _get_prob_cfg(cfg)

    results: List[Dict[str, Any]] = []

    total = len(eta_values) * len(mu_values) * n_seeds
    k = 0

    for eta in eta_values:
        for mu in mu_values:
            for seed_idx in range(n_seeds):
                k += 1
                seed = int(cfg['reproducibility']['seed']) + seed_idx
                print(f"\r  Progress: {k}/{total} (eta={eta}, mu={mu}, seed={seed})", end="")

                tournament = generate_synthetic_tournament(
                    n=n,
                    rho=rho,
                    eta=float(eta),
                    mu=float(mu),
                    m_per_pair=m_per_pair,
                    cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                    btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                    num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                    cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                    cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                    seed=seed,
                )

                P_hat_ste, P_hat_base, pred_metrics, splits = _estimate_phat_for_run(
                    comparisons=tournament.comparisons,
                    n=n,
                    seed=seed,
                    prob_cfg=prob_cfg,
                    quick=quick,
                )

                P_t = torch.from_numpy(P_hat_ste).float()
                t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=alpha))

                tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
                uc_probs = u_tau.detach().cpu().numpy()

                tc_pred = (tc_probs > 0.5).astype(float)
                uc_pred = (uc_probs > 0.5).astype(float)

                tc_rank = scores_to_ranking(tc_probs)
                uc_rank = scores_to_ranking(uc_probs)

                row: Dict[str, Any] = {
                    'n': int(n),
                    'rho': float(rho),
                    'eta': float(eta),
                    'mu': float(mu),
                    'seed': int(seed),
                    'ste_tc_f1': f1_score_core(tc_pred, tournament.true_top_cycle),
                    'ste_uc_f1': f1_score_core(uc_pred, tournament.true_uncovered),
                    'ste_tc_topk_f1': top_k_f1(tc_rank, tournament.true_top_cycle),
                    'ste_uc_topk_f1': top_k_f1(uc_rank, tournament.true_uncovered),
                    'ste_tc_pred_size': float(tc_pred.sum()),
                    'ste_uc_pred_size': float(uc_pred.sum()),
                    **pred_metrics,
                }

                train, _, _ = splits
                baseline_eval = BaselineEvaluator(elo_epochs=int(cfg.get('baselines', {}).get('elo_epochs', 1)))
                baseline_scores = baseline_eval.evaluate_all(P_hat_base, train)
                for method in ['btl', 'elo', 'rank_centrality', 'win_rate']:
                    if method in baseline_scores:
                        ranking = scores_to_ranking(baseline_scores[method])
                        row[f'{method}_tc_f1'] = top_k_f1(ranking, tournament.true_top_cycle)

                results.append(row)

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'robustness_raw.csv'), index=False)
    return df


def run_calibration(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Calibration (ECE/Brier + Reliability Points)")

    n_values = exp.get('n_values', [10, 20, 30])
    rho = float(exp.get('rho', 0.3))
    eta = float(exp.get('eta', 0.05))
    mu = float(exp.get('mu', 0.1))
    m_per_pair = int(exp.get('m_per_pair', 10))

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])
    alpha = float(cfg['ste'].get('alpha', 1.0))
    beta = float(cfg['ste'].get('beta_uncovered', 5.0))

    reachability_mode = str(cfg['ste'].get('reachability_mode', 'max_product'))
    tau_witness = float(cfg['ste'].get('tau_witness', tau))

    prob_cfg = _get_prob_cfg(cfg)

    results: List[Dict[str, Any]] = []
    points: List[Dict[str, Any]] = []

    total = len(n_values) * n_seeds
    k = 0

    for n in n_values:
        for seed_idx in range(n_seeds):
            k += 1
            seed = int(cfg['reproducibility']['seed']) + seed_idx
            print(f"\r  Progress: {k}/{total} (n={n}, seed={seed})", end="")

            tournament = generate_synthetic_tournament(
                n=int(n),
                rho=rho,
                eta=eta,
                mu=mu,
                m_per_pair=m_per_pair,
                cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                seed=seed,
            )

            P_hat_ste, _, pred_metrics, _ = _estimate_phat_for_run(
                comparisons=tournament.comparisons,
                n=int(n),
                seed=seed,
                prob_cfg=prob_cfg,
                quick=quick,
            )

            P_t = torch.from_numpy(P_hat_ste).float()
            t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=alpha))

            tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
            uc_probs = u_tau.detach().cpu().numpy()

            results.append({
                'n': int(n),
                'seed': int(seed),
                'ece_tc': expected_calibration_error(tc_probs, tournament.true_top_cycle, n_bins=int(cfg.get('evaluation', {}).get('calibration_bins', 10))),
                'brier_tc': brier_score(tc_probs, tournament.true_top_cycle),
                'ece_uc': expected_calibration_error(uc_probs, tournament.true_uncovered, n_bins=int(cfg.get('evaluation', {}).get('calibration_bins', 10))),
                'brier_uc': brier_score(uc_probs, tournament.true_uncovered),
                **pred_metrics,
            })

            for a in range(int(n)):
                points.append({
                    'n': int(n),
                    'seed': int(seed),
                    'agent': int(a),
                    'tc_prob': float(tc_probs[a]),
                    'tc_label': float(tournament.true_top_cycle[a]),
                    'uc_prob': float(uc_probs[a]),
                    'uc_label': float(tournament.true_uncovered[a]),
                })

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'calibration_raw.csv'), index=False)

    df_points = pd.DataFrame(points)
    df_points.to_csv(os.path.join(run_dir, 'calibration_points.csv'), index=False)

    return df


def run_stability_vs_sparsity(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Bootstrap Stability vs Sparsity (mu)")

    n = int(exp.get('n', 20))
    rho = float(exp.get('rho', 0.3))
    eta = float(exp.get('eta', 0.05))
    mu_values = exp.get('mu_values', [0.0, 0.1, 0.3, 0.5, 0.7])
    m_per_pair = int(exp.get('m_per_pair', 10))

    # Use cfg evaluation bootstrap unless overridden by exp
    n_bootstrap = int(exp.get('n_bootstrap', cfg.get('evaluation', {}).get('n_bootstrap', 200)))
    if quick:
        n_bootstrap = min(n_bootstrap, 60)

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])

    results: List[Dict[str, Any]] = []

    total = len(mu_values) * n_seeds
    k = 0

    for mu in mu_values:
        for seed_idx in range(n_seeds):
            k += 1
            seed = int(cfg['reproducibility']['seed']) + seed_idx
            print(f"\r  Progress: {k}/{total} (mu={mu}, seed={seed})", end="")

            tournament = generate_synthetic_tournament(
                n=n,
                rho=rho,
                eta=eta,
                mu=float(mu),
                m_per_pair=m_per_pair,
                cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                seed=seed,
            )

            boot = bootstrap_ste(
                comparisons=tournament.comparisons,
                n=n,
                n_bootstrap=n_bootstrap,
                tau=tau,
                K=K,
                seed=seed,
                ste_kwargs=_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))),
            )
            metrics = compute_stability_metrics(boot, true_core=tournament.true_top_cycle)
            results.append({'mu': float(mu), 'seed': int(seed), **metrics})

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'stability_vs_sparsity_raw.csv'), index=False)
    return df


def run_stability(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Bootstrap Stability (single setting)")

    n = int(exp.get('n', 20))
    rho = float(exp.get('rho', 0.3))
    eta = float(exp.get('eta', 0.05))
    mu = float(exp.get('mu', 0.1))
    m_per_pair = int(exp.get('m_per_pair', 10))

    n_bootstrap = int(exp.get('n_bootstrap', cfg.get('evaluation', {}).get('n_bootstrap', 200)))
    if quick:
        n_bootstrap = min(n_bootstrap, 60)

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])

    results: List[Dict[str, Any]] = []

    for seed_idx in range(n_seeds):
        seed = int(cfg['reproducibility']['seed']) + seed_idx
        print(f"\r  Progress: {seed_idx + 1}/{n_seeds} (seed={seed})", end="")

        tournament = generate_synthetic_tournament(
            n=n,
            rho=rho,
            eta=eta,
            mu=mu,
            m_per_pair=m_per_pair,
            cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
            btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
            num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
            cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
            cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
            seed=seed,
        )

        boot = bootstrap_ste(
            comparisons=tournament.comparisons,
            n=n,
            n_bootstrap=n_bootstrap,
            tau=tau,
            K=K,
            seed=seed,
            ste_kwargs=_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))),
        )

        metrics = compute_stability_metrics(boot, true_core=tournament.true_top_cycle)
        results.append({'seed': int(seed), **metrics})

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'stability_raw.csv'), index=False)
    return df


def run_runtime(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Runtime Scaling")

    n_values = exp.get('n_values', [10, 20, 50, 100, 200])
    rho = float(exp.get('rho', 0.3))
    eta = float(exp.get('eta', 0.05))
    mu = float(exp.get('mu', 0.1))
    m_per_pair = int(exp.get('m_per_pair', 10))

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])

    results: List[Dict[str, Any]] = []

    total = len(n_values) * n_seeds
    k = 0

    for n in n_values:
        for seed_idx in range(n_seeds):
            k += 1
            seed = int(cfg['reproducibility']['seed']) + seed_idx
            print(f"\r  Progress: {k}/{total} (n={n}, seed={seed})", end="")

            tournament = generate_synthetic_tournament(
                n=int(n),
                rho=rho,
                eta=eta,
                mu=mu,
                m_per_pair=m_per_pair,
                cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                seed=seed,
            )

            # For runtime, we use empirical P_hat from all comparisons to avoid training overhead.
            P_hat = phat_from_empirical(tournament.comparisons, n=int(n), laplace_alpha=float(cfg.get('prob_estimation', {}).get('laplace_alpha', 1.0)))
            P_t = torch.from_numpy(P_hat).float()

            st = time.time()
            t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))))
            ste_time = time.time() - st

            tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
            tc_pred = (tc_probs > 0.5).astype(float)

            results.append({
                'n': int(n),
                'seed': int(seed),
                'ste_time': float(ste_time),
                'ste_tc_f1': f1_score_core(tc_pred, tournament.true_top_cycle),
            })

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'runtime_raw.csv'), index=False)
    return df


def run_ablation_K(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Ablation on Path Length K")

    n = int(exp.get('n', 20))
    rho = float(exp.get('rho', 0.4))
    K_values = exp.get('K_values', [2, 3, 4, 5])
    m_per_pair = int(exp.get('m_per_pair', 10))

    tau = float(cfg['ste']['tau'])

    results: List[Dict[str, Any]] = []

    total = len(K_values) * n_seeds
    k = 0

    for K in K_values:
        for seed_idx in range(n_seeds):
            k += 1
            seed = int(cfg['reproducibility']['seed']) + seed_idx
            print(f"\r  Progress: {k}/{total} (K={K}, seed={seed})", end="")

            tournament = generate_synthetic_tournament(
                n=n,
                rho=rho,
                eta=0.05,
                mu=0.1,
                m_per_pair=m_per_pair,
                cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                seed=seed,
            )

            P_hat = phat_from_empirical(tournament.comparisons, n=n, laplace_alpha=float(cfg.get('prob_estimation', {}).get('laplace_alpha', 1.0)))
            P_t = torch.from_numpy(P_hat).float()

            st = time.time()
            t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=int(K), alpha=float(cfg['ste'].get('alpha', 1.0))))
            dt = time.time() - st

            tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
            tc_pred = (tc_probs > 0.5).astype(float)

            results.append({'K': int(K), 'seed': int(seed), 'tc_f1': f1_score_core(tc_pred, tournament.true_top_cycle), 'time': float(dt)})

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'ablation_K_raw.csv'), index=False)
    return df


def run_ablation_tau(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    """Ablation on temperature tau (sharpness of majority edge sigmoid).

    This replaces the previous "annealing" placeholder/proxy.
    It is an honest sensitivity analysis: how STE performance changes with tau.
    """
    print_header("Synthetic: Ablation on Temperature tau")

    n = int(exp.get('n', 20))
    rho = float(exp.get('rho', 0.4))
    tau_values = exp.get('tau_values', [0.2, 0.1, 0.05, 0.02, 0.01])
    K = int(exp.get('K', cfg['ste']['K']))
    m_per_pair = int(exp.get('m_per_pair', 10))

    results: List[Dict[str, Any]] = []

    total = len(tau_values) * n_seeds
    k = 0

    for tau in tau_values:
        for seed_idx in range(n_seeds):
            k += 1
            seed = int(cfg['reproducibility']['seed']) + seed_idx
            print(f"\r  Progress: {k}/{total} (tau={tau}, seed={seed})", end="")

            tournament = generate_synthetic_tournament(
                n=n,
                rho=rho,
                eta=0.05,
                mu=0.1,
                m_per_pair=m_per_pair,
                cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
                btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
                num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
                cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
                cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
                seed=seed,
            )

            P_hat = phat_from_empirical(tournament.comparisons, n=n, laplace_alpha=float(cfg.get('prob_estimation', {}).get('laplace_alpha', 1.0)))
            P_t = torch.from_numpy(P_hat).float()

            t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=float(tau), K=int(K), alpha=float(cfg['ste'].get('alpha', 1.0))))
            tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
            uc_probs = u_tau.detach().cpu().numpy()

            tc_pred = (tc_probs > 0.5).astype(float)
            uc_pred = (uc_probs > 0.5).astype(float)

            results.append({
                'tau': float(tau),
                'K': int(K),
                'seed': int(seed),
                'tc_f1': f1_score_core(tc_pred, tournament.true_top_cycle),
                'uc_f1': f1_score_core(uc_pred, tournament.true_uncovered),
            })

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'ablation_tau_raw.csv'), index=False)
    return df


def run_baseline_comparison(cfg: Dict[str, Any], exp: Dict[str, Any], n_seeds: int, run_dir: str, quick: bool) -> pd.DataFrame:
    print_header("Synthetic: Baseline Comparison")

    n = int(exp.get('n', 20))
    rho = float(exp.get('rho', 0.4))
    eta = float(exp.get('eta', 0.05))
    mu = float(exp.get('mu', 0.1))
    m_per_pair = int(exp.get('m_per_pair', 10))

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])

    prob_cfg = _get_prob_cfg(cfg)

    results: List[Dict[str, Any]] = []

    for seed_idx in range(n_seeds):
        seed = int(cfg['reproducibility']['seed']) + seed_idx
        print(f"\r  Progress: {seed_idx + 1}/{n_seeds} (seed={seed})", end="")

        tournament = generate_synthetic_tournament(
            n=n,
            rho=rho,
            eta=eta,
            mu=mu,
            m_per_pair=m_per_pair,
            cycle_size=int(exp.get('cycle_size', cfg.get('synthetic', {}).get('cycle_size', 3))),
            btl_scale=float(exp.get('btl_scale', cfg.get('synthetic', {}).get('btl_scale', 1.0))),
            num_cycles=int(exp.get('num_cycles', cfg.get('synthetic', {}).get('num_cycles', 1))),
            cycle_edge_prob=float(exp.get('cycle_edge_prob', cfg.get('synthetic', {}).get('cycle_edge_prob', 0.7))),
            cycle_injection_mode=str(exp.get('cycle_injection_mode', cfg.get('synthetic', {}).get('cycle_injection_mode', 'edge_only'))),
            seed=seed,
        )

        P_hat_ste, P_hat_base, pred_metrics, splits = _estimate_phat_for_run(
            comparisons=tournament.comparisons,
            n=n,
            seed=seed,
            prob_cfg=prob_cfg,
            quick=quick,
        )

        P_t = torch.from_numpy(P_hat_ste).float()
        st = time.time()
        t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))))
        ste_time = time.time() - st

        tc_probs = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
        uc_probs = u_tau.detach().cpu().numpy()

        tc_pred = (tc_probs > 0.5).astype(float)
        uc_pred = (uc_probs > 0.5).astype(float)

        row: Dict[str, Any] = {
            'seed': int(seed),
            'ste_tc_f1': f1_score_core(tc_pred, tournament.true_top_cycle),
            'ste_tc_jaccard': jaccard_index(tc_pred, tournament.true_top_cycle),
            'ste_uc_f1': f1_score_core(uc_pred, tournament.true_uncovered),
            'ste_uc_jaccard': jaccard_index(uc_pred, tournament.true_uncovered),
            'ste_time': float(ste_time),
            **pred_metrics,
        }

        train, _, _ = splits
        baseline_eval = BaselineEvaluator(elo_epochs=int(cfg.get('baselines', {}).get('elo_epochs', 1)))
        st2 = time.time()
        baseline_scores = baseline_eval.evaluate_all(P_hat_base, train)
        base_time = time.time() - st2

        for method, scores in baseline_scores.items():
            ranking = scores_to_ranking(scores)
            row[f'{method}_tc_f1'] = top_k_f1(ranking, tournament.true_top_cycle)
            row[f'{method}_top1'] = top_1_in_core(ranking, tournament.true_top_cycle)

        row['baseline_time'] = float(base_time)
        results.append(row)

    print()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(run_dir, 'baseline_comparison_raw.csv'), index=False)
    return df


# -----------------------------------------------------------------------------
# Real-world experiments (optional)
# -----------------------------------------------------------------------------


def _phat_from_realworld(comparisons: np.ndarray, n: int, laplace_alpha: float = 1.0) -> np.ndarray:
    return phat_from_empirical(comparisons, n=n, laplace_alpha=float(laplace_alpha), fill_value=0.5)


def run_chatbot_arena_global(cfg: Dict[str, Any], exp: Dict[str, Any], run_dir: str) -> pd.DataFrame:
    print_header("Real-world: Chatbot Arena (global)")

    file_path = exp.get('file') or cfg.get('paths', {}).get('chatbot_arena_file')
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Chatbot Arena file not found: {file_path}")

    ds = load_chatbot_arena_pairwise(
        path=file_path,
        model_a_col=exp.get('model_a_col', 'model_a'),
        model_b_col=exp.get('model_b_col', 'model_b'),
        winner_col=exp.get('winner_col', 'winner'),
        category_col=exp.get('category_col', 'category'),
        tie_policy=str(exp.get('tie_policy', 'drop')),
    )

    n = len(ds.id2model)
    P_hat = _phat_from_realworld(ds.comparisons, n=n, laplace_alpha=float(cfg.get('prob_estimation', {}).get('laplace_alpha', 1.0)))

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])

    P_t = torch.from_numpy(P_hat).float()
    t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))))

    tc_prob = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
    uc_prob = u_tau.detach().cpu().numpy()

    baseline_eval = BaselineEvaluator(elo_epochs=int(cfg.get('baselines', {}).get('elo_epochs', 1)))
    baseline_scores = baseline_eval.evaluate_all(P_hat, ds.comparisons)

    top_k = int(exp.get('top_k', 10))

    df = pd.DataFrame({'model': ds.id2model, 'tc_prob': tc_prob, 'uc_prob': uc_prob})

    for method in ['win_rate', 'btl', 'elo', 'rank_centrality', 'trueskill']:
        if method in baseline_scores:
            df[f'{method}_score'] = baseline_scores[method]

    df_sorted = df.sort_values('uc_prob', ascending=False).head(top_k).reset_index(drop=True)
    df_sorted.to_csv(os.path.join(run_dir, 'chatbot_arena_global_raw.csv'), index=False)

    return df_sorted


def run_chatbot_arena_by_category(cfg: Dict[str, Any], exp: Dict[str, Any], run_dir: str) -> pd.DataFrame:
    print_header("Real-world: Chatbot Arena (by category)")

    file_path = exp.get('file') or cfg.get('paths', {}).get('chatbot_arena_file')
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Chatbot Arena file not found: {file_path}")

    ds = load_chatbot_arena_pairwise(
        path=file_path,
        model_a_col=exp.get('model_a_col', 'model_a'),
        model_b_col=exp.get('model_b_col', 'model_b'),
        winner_col=exp.get('winner_col', 'winner'),
        category_col=exp.get('category_col', 'category'),
        tie_policy=str(exp.get('tie_policy', 'drop')),
    )

    if 'category' not in ds.df_norm.columns:
        raise ValueError("Category column not found; cannot run by-category analysis.")

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])

    rows: List[Dict[str, Any]] = []
    top_k = int(exp.get('top_k', 8))

    for cat, grp in ds.df_norm.groupby('category'):
        comps = grp[['a_id', 'b_id', 'y']].to_numpy(dtype=np.int64)
        n = len(ds.id2model)
        P_hat = _phat_from_realworld(comps, n=n, laplace_alpha=float(cfg.get('prob_estimation', {}).get('laplace_alpha', 1.0)))
        P_t = torch.from_numpy(P_hat).float()
        t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))))
        tc_prob = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
        uc_prob = u_tau.detach().cpu().numpy()

        df_cat = pd.DataFrame({'model': ds.id2model, 'tc_prob': tc_prob, 'uc_prob': uc_prob})
        df_cat = df_cat.sort_values('uc_prob', ascending=False).head(top_k)
        for _, r in df_cat.iterrows():
            rows.append({'category': cat, 'model': r['model'], 'uc_prob': float(r['uc_prob']), 'tc_prob': float(r['tc_prob'])})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(run_dir, 'chatbot_arena_by_category_raw.csv'), index=False)
    return out


def run_agentbench_per_environment(cfg: Dict[str, Any], exp: Dict[str, Any], run_dir: str) -> pd.DataFrame:
    print_header("Real-world: AgentBench (per environment)")

    file_path = exp.get('file') or cfg.get('paths', {}).get('agentbench_file')
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"AgentBench file not found: {file_path}")

    ds = load_agentbench_pairwise(
        path=file_path,
        agent_a_col=exp.get('agent_a_col', 'agent_a'),
        agent_b_col=exp.get('agent_b_col', 'agent_b'),
        winner_col=exp.get('winner_col', 'winner'),
        env_col=exp.get('env_col', 'environment'),
        tie_policy=str(exp.get('tie_policy', 'drop')),
    )

    if 'environment' not in ds.df_norm.columns:
        raise ValueError("Environment column not found; cannot run per-environment analysis.")

    tau = float(cfg['ste']['tau'])
    K = int(cfg['ste']['K'])
    top_k = int(exp.get('top_k', 8))

    rows: List[Dict[str, Any]] = []
    n = len(ds.id2agent)

    for env, grp in ds.df_norm.groupby('environment'):
        comps = grp[['a_id', 'b_id', 'y']].to_numpy(dtype=np.int64)
        P_hat = _phat_from_realworld(comps, n=n, laplace_alpha=float(cfg.get('prob_estimation', {}).get('laplace_alpha', 1.0)))
        P_t = torch.from_numpy(P_hat).float()
        t_tau, u_tau = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=float(cfg['ste'].get('alpha', 1.0))))
        tc_prob = top_cycle_membership_prob(t_tau).detach().cpu().numpy()
        uc_prob = u_tau.detach().cpu().numpy()

        df_env = pd.DataFrame({'agent': ds.id2agent, 'tc_prob': tc_prob, 'uc_prob': uc_prob})
        df_env = df_env.sort_values('uc_prob', ascending=False).head(top_k)
        for _, r in df_env.iterrows():
            rows.append({'environment': env, 'agent': r['agent'], 'uc_prob': float(r['uc_prob']), 'tc_prob': float(r['tc_prob'])})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(run_dir, 'agentbench_per_environment_raw.csv'), index=False)
    return out


# -----------------------------------------------------------------------------
# Paper assets
# -----------------------------------------------------------------------------


def generate_latex_tables(results: Dict[str, pd.DataFrame], output_root: str) -> None:
    tables_dir = os.path.join(output_root, 'paper_assets', 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # Table: core recovery (example; adjust columns as you finalize baselines)
    if 'core_recovery' in results:
        df = results['core_recovery']

        cols = [c for c in ['ste_tc_f1', 'btl_tc_f1', 'elo_tc_f1', 'rank_centrality_tc_f1', 'win_rate_tc_f1'] if c in df.columns]
        if cols:
            agg = df.groupby(['n', 'rho']).agg({c: ['mean', 'std'] for c in cols}).round(3)

            formatted = pd.DataFrame(index=agg.index)
            for c in cols:
                formatted[c] = agg[(c, 'mean')].astype(str) + ' ± ' + agg[(c, 'std')].astype(str)

            formatted.columns = [c.replace('_tc_f1', '').replace('ste', 'STE').replace('_', ' ') for c in formatted.columns]

            with open(os.path.join(tables_dir, 'table6_core_recovery.tex'), 'w', encoding='utf-8') as f:
                f.write("% Table 6: Core Recovery (Top Cycle F1)\n")
                f.write(formatted.to_latex(escape=False))

    # Table: robustness
    if 'robustness' in results:
        df = results['robustness']
        cols = [c for c in ['ste_tc_f1', 'btl_tc_f1', 'elo_tc_f1'] if c in df.columns]
        if cols:
            agg = df.groupby(['eta', 'mu']).agg({c: ['mean', 'std'] for c in cols}).round(3)
            formatted = pd.DataFrame(index=agg.index)
            for c in cols:
                formatted[c] = agg[(c, 'mean')].astype(str) + ' ± ' + agg[(c, 'std')].astype(str)
            formatted.columns = [c.replace('_tc_f1', '').replace('ste', 'STE').replace('_', ' ') for c in formatted.columns]
            with open(os.path.join(tables_dir, 'table7_robustness.tex'), 'w', encoding='utf-8') as f:
                f.write("% Table 7: Robustness\n")
                f.write(formatted.to_latex(escape=False))

    # Table: calibration
    if 'calibration' in results:
        df = results['calibration']
        agg = df.groupby('n').agg({
            'ece_tc': ['mean', 'std'],
            'brier_tc': ['mean', 'std'],
            'ece_uc': ['mean', 'std'],
            'brier_uc': ['mean', 'std'],
        }).round(4)

        with open(os.path.join(tables_dir, 'table8_calibration.tex'), 'w', encoding='utf-8') as f:
            f.write("% Table 8: Calibration Analysis\n")
            f.write(agg.to_latex(escape=False))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='STE Experiment Runner (YAML)')
    parser.add_argument('--config', type=str, default='configs/ste_master.yaml')
    parser.add_argument('--exp', type=str, default='all', help='Experiment name or all')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output root')

    args = parser.parse_args()

    # Load YAML
    cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    cfg = load_yaml_config(cfg_path)

    if args.output_dir is not None:
        cfg.setdefault('paths', {})
        cfg['paths']['output_root'] = args.output_dir

    out_root = cfg.get('paths', {}).get('output_root', './outputs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(out_root, 'runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Save config copy for reproducibility
    try:
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))
    except Exception:
        pass

    # Seed
    set_seed(int(cfg.get('reproducibility', {}).get('seed', 42)))

    n_seeds = int(cfg.get('reproducibility', {}).get('num_runs', 10))
    n_seeds_runtime = int(cfg.get('reproducibility', {}).get('num_runs_runtime', max(2, n_seeds // 2)))

    if args.quick:
        n_seeds = min(3, n_seeds)
        n_seeds_runtime = min(2, n_seeds_runtime)

    print_header('STE Experiment Pipeline (YAML)')
    if args.quick:
        print('*** QUICK MODE ENABLED (SMOKE TEST ONLY) ***')
    print(f"Config: {args.config}")
    print(f"Run dir: {run_dir}")
    print(f"Seeds (main): {n_seeds}")
    print(f"Seeds (runtime): {n_seeds_runtime}")
    print(f"Prob estimator: {str(_get_prob_cfg(cfg).get('method', 'empirical_train'))}")

    # Meta
    meta = {
        'timestamp': timestamp,
        'run_dir': run_dir,
        'config': args.config,
        'quick': bool(args.quick),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch_version': getattr(torch, '__version__', 'unknown'),
        'cuda_available': bool(torch.cuda.is_available()),
        'seed_base': int(cfg.get('reproducibility', {}).get('seed', 42)),
        'num_runs': int(cfg.get('reproducibility', {}).get('num_runs', 10)),
    }
    with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    exp_cfg = cfg.get('experiments', {})

    if args.exp == 'all':
        exp_names = [k for k, v in exp_cfg.items() if isinstance(v, dict) and v.get('enabled', False)]
    else:
        exp_names = [args.exp]

    results: Dict[str, pd.DataFrame] = {}

    for name in exp_names:
        exp = exp_cfg.get(name)
        if exp is None:
            raise ValueError(f"Unknown experiment: {name}")

        if args.exp == 'all' and not exp.get('enabled', True):
            continue

        # Synthetic
        if name == 'core_recovery':
            results[name] = run_core_recovery(cfg, exp, n_seeds=n_seeds, run_dir=run_dir, quick=args.quick)
        elif name == 'robustness':
            results[name] = run_robustness(cfg, exp, n_seeds=n_seeds, run_dir=run_dir, quick=args.quick)
        elif name == 'calibration':
            results[name] = run_calibration(cfg, exp, n_seeds=n_seeds, run_dir=run_dir, quick=args.quick)
        elif name == 'stability_vs_sparsity':
            results[name] = run_stability_vs_sparsity(cfg, exp, n_seeds=n_seeds_runtime, run_dir=run_dir, quick=args.quick)
        elif name == 'stability':
            results[name] = run_stability(cfg, exp, n_seeds=n_seeds_runtime, run_dir=run_dir, quick=args.quick)
        elif name == 'runtime':
            results[name] = run_runtime(cfg, exp, n_seeds=n_seeds_runtime, run_dir=run_dir, quick=args.quick)
        elif name == 'ablation_K':
            results[name] = run_ablation_K(cfg, exp, n_seeds=n_seeds, run_dir=run_dir, quick=args.quick)
        elif name in ('ablation_tau', 'ablation_temperature'):
            results[name] = run_ablation_tau(cfg, exp, n_seeds=n_seeds, run_dir=run_dir, quick=args.quick)
        elif name == 'baseline_comparison':
            results[name] = run_baseline_comparison(cfg, exp, n_seeds=n_seeds, run_dir=run_dir, quick=args.quick)

        # Real-world
        elif name == 'chatbot_arena_global':
            results[name] = run_chatbot_arena_global(cfg, exp, run_dir=run_dir)
        elif name == 'chatbot_arena_by_category':
            results[name] = run_chatbot_arena_by_category(cfg, exp, run_dir=run_dir)
        elif name == 'agentbench_per_environment':
            results[name] = run_agentbench_per_environment(cfg, exp, run_dir=run_dir)
        else:
            raise ValueError(f"Experiment implemented in YAML but not in runner: {name}")

    # Tables
    generate_latex_tables(results, out_root)

    # Figures
    figs_dir = os.path.join(out_root, 'paper_assets', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    try:
        generate_all_figures(run_dir, figs_dir)
    except Exception as e:
        print(f"[WARN] Figure generation failed: {e}")

    print_header('Done')
    print(f"Raw results: {run_dir}")
    print(f"Tables: {os.path.join(out_root, 'paper_assets', 'tables')}")
    print(f"Figures: {figs_dir}")


if __name__ == '__main__':
    main()
