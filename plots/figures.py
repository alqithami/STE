"""Figure generation for STE experiments.

This module provides:
- a single orchestration entry point: generate_all_figures(run_dir, figs_dir)
- helper plot functions for the paper assets

All plots are saved as PDF for inclusion in LaTeX.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_f1_vs_rho(
    df: pd.DataFrame,
    out_pdf: str,
    metric: str = 'ste_tc_f1',
    ylabel: str = 'F1 (Top Cycle)',
) -> None:
    """Plot mean metric vs rho with error bars, grouped by n."""
    _ensure_dir(os.path.dirname(out_pdf))

    if 'rho' not in df.columns or 'n' not in df.columns:
        return

    fig = plt.figure(figsize=(5.2, 3.4))

    for n in sorted(df['n'].unique()):
        sub = df[df['n'] == n]
        agg = sub.groupby('rho')[metric].agg(['mean', 'std']).reset_index().sort_values('rho')
        plt.errorbar(agg['rho'], agg['mean'], yerr=agg['std'], marker='o', linewidth=1.5, label=f'n={int(n)}')

    plt.xlabel('Cycle strength $\\rho$')
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_runtime_scaling(df: pd.DataFrame, out_pdf: str) -> None:
    _ensure_dir(os.path.dirname(out_pdf))
    if 'n' not in df.columns or 'ste_time' not in df.columns:
        return

    agg = df.groupby('n')['ste_time'].agg(['mean', 'std']).reset_index().sort_values('n')

    fig = plt.figure(figsize=(5.2, 3.4))
    plt.errorbar(agg['n'], agg['mean'], yerr=agg['std'], marker='o', linewidth=1.5)
    plt.xlabel('Number of agents $n$')
    plt.ylabel('STE runtime (s)')
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_stability_vs_mu(df: pd.DataFrame, out_pdf: str, which: str = 'tc') -> None:
    _ensure_dir(os.path.dirname(out_pdf))
    if 'mu' not in df.columns:
        return

    col = f'{which}_stability_jaccard'
    if col not in df.columns:
        return

    agg = df.groupby('mu')[col].agg(['mean', 'std']).reset_index().sort_values('mu')

    fig = plt.figure(figsize=(5.2, 3.4))
    plt.errorbar(agg['mu'], agg['mean'], yerr=agg['std'], marker='o', linewidth=1.5)
    plt.xlabel('Sparsity $\\mu$')
    plt.ylabel('Bootstrap stability (Jaccard)')
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def _reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    accs = []
    counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        bin_centers.append(np.mean(probs[mask]))
        accs.append(np.mean(labels[mask]))
        counts.append(mask.sum())

    return np.array(bin_centers), np.array(accs), np.array(counts)


def plot_reliability_diagram(
    df_points: pd.DataFrame,
    out_pdf: str,
    prob_col: str,
    label_col: str,
    title: str,
    n_bins: int = 10,
) -> None:
    _ensure_dir(os.path.dirname(out_pdf))

    probs = df_points[prob_col].to_numpy(dtype=np.float64)
    labels = df_points[label_col].to_numpy(dtype=np.float64)

    x, y, _ = _reliability_curve(probs, labels, n_bins=n_bins)

    fig = plt.figure(figsize=(5.2, 3.4))
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.0)
    if x.size > 0:
        plt.plot(x, y, marker='o', linewidth=1.5)

    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical frequency')
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def generate_all_figures(run_dir: str, figs_dir: str) -> None:
    """Generate all paper figures given a run directory containing raw CSVs."""
    _ensure_dir(figs_dir)

    # 1) F1 vs rho
    core_path = os.path.join(run_dir, 'core_recovery_raw.csv')
    if os.path.exists(core_path):
        df = pd.read_csv(core_path)
        # Threshold-based set recovery (paper default)
        if 'ste_tc_f1' in df.columns:
            plot_f1_vs_rho(df, os.path.join(figs_dir, 'f1_vs_rho.pdf'), metric='ste_tc_f1', ylabel='F1 (Top Cycle, threshold)')

        # Ranking-style diagnostic (top-|C*|). Useful when thresholding is unstable.
        if 'ste_tc_topk_f1' in df.columns:
            plot_f1_vs_rho(df, os.path.join(figs_dir, 'f1_vs_rho_topk.pdf'), metric='ste_tc_topk_f1', ylabel='Top-k F1 (Top Cycle)')

    # 2) Runtime scaling
    runtime_path = os.path.join(run_dir, 'runtime_raw.csv')
    if os.path.exists(runtime_path):
        df = pd.read_csv(runtime_path)
        plot_runtime_scaling(df, os.path.join(figs_dir, 'runtime_scaling.pdf'))

    # 3) Stability vs sparsity (mu)
    stab_mu_path = os.path.join(run_dir, 'stability_vs_sparsity_raw.csv')
    if os.path.exists(stab_mu_path):
        df = pd.read_csv(stab_mu_path)
        plot_stability_vs_mu(df, os.path.join(figs_dir, 'stability_vs_mu_tc.pdf'), which='tc')
        plot_stability_vs_mu(df, os.path.join(figs_dir, 'stability_vs_mu_uc.pdf'), which='uc')

    # 4) Calibration reliability diagram
    points_path = os.path.join(run_dir, 'calibration_points.csv')
    if os.path.exists(points_path):
        pts = pd.read_csv(points_path)
        if 'tc_prob' in pts.columns and 'tc_label' in pts.columns:
            plot_reliability_diagram(
                pts,
                os.path.join(figs_dir, 'reliability_tc.pdf'),
                prob_col='tc_prob',
                label_col='tc_label',
                title='Reliability: Top Cycle membership',
                n_bins=10,
            )
        if 'uc_prob' in pts.columns and 'uc_label' in pts.columns:
            plot_reliability_diagram(
                pts,
                os.path.join(figs_dir, 'reliability_uc.pdf'),
                prob_col='uc_prob',
                label_col='uc_label',
                title='Reliability: Uncovered membership',
                n_bins=10,
            )

    # 5) (Optional) baseline comparison plots could be added here.
