#!/usr/bin/env python3
"""Audit one end-to-end STE run and persist intermediate artifacts.

This script is designed to address the exact concern you raised: whether
results are *actually computed* versus being produced by placeholders.

It runs a **single** synthetic setting end-to-end and writes:
- P_true (ground-truth probabilistic tournament)
- sampled comparison rows
- P_hat (estimated tournament)
- STE intermediates: D (soft majority), R (reachability), cover
- final membership probabilities (tc_prob, uc_prob)
- a JSON report with metrics + SHA256 checksums

Usage (from repo root):
  python tools/audit_one_run.py --config configs/ste_master.yaml --n 20 --rho 0.4 --seed 42

You can then inspect the saved NPZ numerically to verify the pipeline is not
returning constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Ensure repo root is on sys.path when running as a script.
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config_loader import load_yaml_config
from data.synthetic import generate_synthetic_tournament
from prob_estimation import (
    split_comparisons,
    phat_from_empirical,
    fit_learned_btl,
    phat_from_scores,
    predictive_logloss,
    predictive_accuracy,
)
from ste_ops.ste import compute_ste_scores, top_cycle_membership_prob
from eval.core_metrics import f1_score_core, jaccard_index, expected_calibration_error, brier_score


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _validate_phat(P_hat: np.ndarray, tol: float = 1e-6) -> None:
    P = np.asarray(P_hat, dtype=np.float64)
    if P.shape[0] != P.shape[1]:
        raise ValueError(f"P_hat must be square, got {P.shape}")
    if not np.allclose(np.diag(P), 0.5, atol=tol):
        raise ValueError("P_hat diagonal must be 0.5")
    if not np.allclose(P + P.T, 1.0, atol=1e-5):
        raise ValueError("P_hat must satisfy P_hat[i,j] + P_hat[j,i] = 1")


def _ste_kwargs(cfg: Dict[str, Any], tau: float, K: int, alpha: float) -> Dict[str, Any]:
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
        'beta_uncovered': float(ste_cfg.get('beta_uncovered', 5.0)),
        'return_aux': True,
    }


def _estimate_phat(cfg: Dict[str, Any], comparisons: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build P_hat using the estimator configured in YAML."""

    prob_cfg = cfg.get('prob_estimation', {}) if isinstance(cfg.get('prob_estimation', {}), dict) else {}
    method = str(prob_cfg.get('method', 'empirical_train')).lower()

    laplace_alpha = float(prob_cfg.get('laplace_alpha', 1.0))
    fill_value = float(prob_cfg.get('fill_value', 0.5))

    split = prob_cfg.get('split', {}) if isinstance(prob_cfg.get('split', {}), dict) else {}
    train_frac = float(split.get('train', 0.8))
    val_frac = float(split.get('val', 0.1))
    test_frac = float(split.get('test', 0.1))

    train, val, test = split_comparisons(comparisons, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)

    metrics: Dict[str, float] = {}

    if method == 'empirical_all':
        P_hat = phat_from_empirical(comparisons, n=n, laplace_alpha=laplace_alpha, fill_value=fill_value)
    elif method == 'empirical_train':
        P_hat = phat_from_empirical(train, n=n, laplace_alpha=laplace_alpha, fill_value=fill_value)
    elif method == 'learned_btl':
        train_cfg = prob_cfg.get('learned_btl', {}) if isinstance(prob_cfg.get('learned_btl', {}), dict) else {}
        res = fit_learned_btl(
            train_comparisons=train,
            val_comparisons=val,
            n=n,
            seed=seed,
            max_epochs=int(train_cfg.get('max_epochs', 200)),
            batch_size=int(train_cfg.get('batch_size', 512)),
            lr=float(train_cfg.get('lr', 0.05)),
            weight_decay=float(train_cfg.get('weight_decay', 0.0)),
            early_stop_patience=int(train_cfg.get('patience', 20)),
            calibrate_temperature=bool(train_cfg.get('calibrate_temperature', True)),
            temp_max_iters=int(train_cfg.get('temp_max_iters', 200)),
            temp_lr=float(train_cfg.get('temp_lr', 0.1)),
            device=train_cfg.get('device', None),
        )
        P_hat = phat_from_scores(res.scores, temperature=res.temperature)
        metrics['learned_btl_temperature'] = float(res.temperature)
        metrics['learned_btl_train_nll'] = float(res.train_nll)
        metrics['learned_btl_val_nll'] = float(res.val_nll)
    else:
        raise ValueError(f"Unknown prob_estimation.method: {method}")

    _validate_phat(P_hat)

    metrics['logloss_train'] = float(predictive_logloss(P_hat, train))
    metrics['logloss_val'] = float(predictive_logloss(P_hat, val))
    metrics['logloss_test'] = float(predictive_logloss(P_hat, test))
    metrics['acc_test'] = float(predictive_accuracy(P_hat, test, threshold=0.5))

    return P_hat, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description='STE audit run (single setting)')
    ap.add_argument('--config', type=str, default='configs/ste_master.yaml')
    ap.add_argument('--output_root', type=str, default=None, help='override paths.output_root')

    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--rho', type=float, default=0.4)
    ap.add_argument('--eta', type=float, default=0.05)
    ap.add_argument('--mu', type=float, default=0.1)
    ap.add_argument('--m_per_pair', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--cycle_size', type=int, default=None)
    ap.add_argument('--num_cycles', type=int, default=None)
    ap.add_argument('--cycle_edge_prob', type=float, default=None)
    ap.add_argument('--btl_scale', type=float, default=None)
    ap.add_argument('--cycle_injection_mode', type=str, default=None)

    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    cfg = load_yaml_config(cfg_path)

    if args.output_root is not None:
        cfg.setdefault('paths', {})
        cfg['paths']['output_root'] = args.output_root

    out_root = cfg.get('paths', {}).get('output_root', './outputs')

    syn = cfg.get('synthetic', {}) if isinstance(cfg.get('synthetic', {}), dict) else {}

    cycle_size = int(args.cycle_size) if args.cycle_size is not None else int(syn.get('cycle_size', 5))
    num_cycles = int(args.num_cycles) if args.num_cycles is not None else int(syn.get('num_cycles', 3))
    cycle_edge_prob = float(args.cycle_edge_prob) if args.cycle_edge_prob is not None else float(syn.get('cycle_edge_prob', 0.9))
    btl_scale = float(args.btl_scale) if args.btl_scale is not None else float(syn.get('btl_scale', 0.3))
    cycle_injection_mode = str(args.cycle_injection_mode) if args.cycle_injection_mode is not None else str(syn.get('cycle_injection_mode', 'edge_only'))

    ste_cfg = cfg.get('ste', {}) if isinstance(cfg.get('ste', {}), dict) else {}
    tau = float(ste_cfg.get('tau', 0.05))
    K = int(ste_cfg.get('K', 3))
    alpha = float(ste_cfg.get('alpha', 1.0))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audit_dir = os.path.join(out_root, 'audit', f'{timestamp}_n{args.n}_rho{args.rho}_seed{args.seed}')
    os.makedirs(audit_dir, exist_ok=True)

    # Save the exact config used
    shutil.copy2(cfg_path, os.path.join(audit_dir, 'config_used.yaml'))

    # 1) Generate synthetic instance
    tour = generate_synthetic_tournament(
        n=int(args.n),
        rho=float(args.rho),
        eta=float(args.eta),
        mu=float(args.mu),
        m_per_pair=int(args.m_per_pair),
        seed=int(args.seed),
        cycle_size=cycle_size,
        btl_scale=btl_scale,
        num_cycles=num_cycles,
        cycle_edge_prob=cycle_edge_prob,
        cycle_injection_mode=cycle_injection_mode,
    )

    # 2) Estimate P_hat
    P_hat, pred_metrics = _estimate_phat(cfg, tour.comparisons, n=int(args.n), seed=int(args.seed))

    # 3) Run STE with aux outputs
    P_t = torch.from_numpy(P_hat).float()
    t_tc, u_uc, aux = compute_ste_scores(P_t, **_ste_kwargs(cfg, tau=tau, K=K, alpha=alpha))

    tc_prob = top_cycle_membership_prob(t_tc).detach().cpu().numpy().astype(np.float64)
    uc_prob = u_uc.detach().cpu().numpy().astype(np.float64)

    thr = float(cfg.get('evaluation', {}).get('threshold', 0.5))
    tc_pred = (tc_prob > thr).astype(np.float64)
    uc_pred = (uc_prob > thr).astype(np.float64)

    # 4) Metrics
    report: Dict[str, Any] = {
        'n': int(args.n),
        'rho': float(args.rho),
        'eta': float(args.eta),
        'mu': float(args.mu),
        'm_per_pair': int(args.m_per_pair),
        'seed': int(args.seed),
        'synthetic': {
            'cycle_size': int(cycle_size),
            'num_cycles': int(num_cycles),
            'cycle_edge_prob': float(cycle_edge_prob),
            'btl_scale': float(btl_scale),
            'cycle_injection_mode': str(cycle_injection_mode),
        },
        'ste': {
            'tau': float(tau),
            'K': int(K),
            'alpha': float(alpha),
            'reachability_mode': str(ste_cfg.get('reachability_mode', 'max_product')),
            'uncovered_mode': str(ste_cfg.get('uncovered_mode', 'lukasiewicz')),
        },
        'pred_metrics': pred_metrics,
        'core_metrics': {
            'tc_f1_threshold': float(f1_score_core(tc_pred, tour.true_top_cycle)),
            'tc_jaccard_threshold': float(jaccard_index(tc_pred, tour.true_top_cycle)),
            'uc_f1_threshold': float(f1_score_core(uc_pred, tour.true_uncovered)),
            'uc_jaccard_threshold': float(jaccard_index(uc_pred, tour.true_uncovered)),
            'true_tc_size': float(np.asarray(tour.true_top_cycle).sum()),
            'true_uc_size': float(np.asarray(tour.true_uncovered).sum()),
            'pred_tc_size': float(tc_pred.sum()),
            'pred_uc_size': float(uc_pred.sum()),
        },
        'calibration': {
            'ece_tc': float(expected_calibration_error(tc_prob, tour.true_top_cycle, n_bins=int(cfg.get('evaluation', {}).get('calibration_bins', 10)))),
            'brier_tc': float(brier_score(tc_prob, tour.true_top_cycle)),
            'ece_uc': float(expected_calibration_error(uc_prob, tour.true_uncovered, n_bins=int(cfg.get('evaluation', {}).get('calibration_bins', 10)))),
            'brier_uc': float(brier_score(uc_prob, tour.true_uncovered)),
        },
        'stats': {
            'tc_prob_min': float(np.min(tc_prob)),
            'tc_prob_mean': float(np.mean(tc_prob)),
            'tc_prob_max': float(np.max(tc_prob)),
            'uc_prob_min': float(np.min(uc_prob)),
            'uc_prob_mean': float(np.mean(uc_prob)),
            'uc_prob_max': float(np.max(uc_prob)),
            'phat_abs_margin_mean': float(np.mean(np.abs(P_hat - 0.5))),
        },
    }

    # 5) Save arrays
    arrays_path = os.path.join(audit_dir, 'audit_arrays.npz')
    np.savez_compressed(
        arrays_path,
        P_true=np.asarray(tour.P, dtype=np.float64),
        P_hat=np.asarray(P_hat, dtype=np.float64),
        comparisons=np.asarray(tour.comparisons, dtype=np.int64),
        true_top_cycle=np.asarray(tour.true_top_cycle, dtype=np.float64),
        true_uncovered=np.asarray(tour.true_uncovered, dtype=np.float64),
        tc_prob=tc_prob,
        uc_prob=uc_prob,
        D=aux['D'].detach().cpu().numpy().astype(np.float64),
        R=aux['R'].detach().cpu().numpy().astype(np.float64),
        cover=aux['cover'].detach().cpu().numpy().astype(np.float64),
    )

    report['artifacts'] = {
        'audit_dir': audit_dir,
        'arrays_file': os.path.basename(arrays_path),
        'arrays_sha256': _sha256(arrays_path),
    }

    report_path = os.path.join(audit_dir, 'audit_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('Audit run complete.')
    print(f'  Audit dir : {audit_dir}')
    print(f'  Arrays    : {arrays_path}')
    print(f'  Report    : {report_path}')
    print('  Key metrics:')
    cm = report['core_metrics']
    print(f"    TC F1 (thr) : {cm['tc_f1_threshold']:.4f} | pred_size={cm['pred_tc_size']:.0f}, true_size={cm['true_tc_size']:.0f}")
    print(f"    UC F1 (thr) : {cm['uc_f1_threshold']:.4f} | pred_size={cm['pred_uc_size']:.0f}, true_size={cm['true_uc_size']:.0f}")


if __name__ == '__main__':
    main()
