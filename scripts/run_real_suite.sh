#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MANIFEST="${1:-configs/real_datasets_manifest_template.yaml}"
OUT="${2:-outputs/real_suite_final}"
python -m ste_neurips.neurips_suite real-suite --manifest "$MANIFEST" --out "$OUT"
