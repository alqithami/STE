#!/usr/bin/env python3
"""Backward-compatible entry point for the STE experiment pipeline.

Historically, the pipeline used `run_all.py --experiment ...`.
This version forwards to the YAML-driven runner in `ste/run.py`.

Examples:
  python run_all.py                     # runs all enabled experiments in the YAML
  python run_all.py --quick             # quick mode
  python run_all.py --experiment core_recovery
  python run_all.py --config configs/ste_master.yaml --experiment all
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='STE Paper Pipeline (wrapper)')
    parser.add_argument('--config', type=str, default='configs/ste_master.yaml')
    parser.add_argument('--quick', action='store_true')
    # Accept both legacy and new flag names.
    parser.add_argument('--exp', '--experiment', dest='experiment', type=str, default='all', help='Experiment name or all')
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    # Forward args to ste.run
    argv = ['-m', 'ste.run', '--config', args.config, '--exp', args.experiment]
    if args.quick:
        argv.append('--quick')
    if args.output_dir is not None:
        argv.extend(['--output_dir', args.output_dir])

    # We import and call ste.run.main() directly to avoid spawning a subprocess.
    from run import main as run_main  # type: ignore

    sys.argv = ['ste.run'] + argv[2:]  # drop '-m ste.run'
    run_main()


if __name__ == '__main__':
    main()
