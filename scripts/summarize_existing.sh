#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:?provide output directory}"
python -m ste_neurips.neurips_suite summarize --out "$OUT"
