#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-ste_neurips_reviewer_artifact.zip}
rm -f "$OUT"
find . \
  -path './.venv' -prune -o \
  -path './**/__pycache__' -prune -o \
  -path './outputs/neurips_smoke' -prune -o \
  -type f -print | zip -@ "$OUT"
echo "Wrote $OUT"
