#!/usr/bin/env bash
set -euo pipefail

find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf .pytest_cache .mypy_cache .ruff_cache
rm -rf outputs/neurips_smoke outputs/neurips_final outputs/real_suite_final outputs/reviewer_artifact
rm -rf outputs/runs outputs/audit outputs/tmp
echo "Removed local generated outputs and Python caches."
