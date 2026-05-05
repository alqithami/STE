#!/usr/bin/env bash
set -euo pipefail

python scripts/download_real_datasets.py \
  --dataset all \
  --out-dir data \
  --write-execution-templates \
  --update-manifest configs/real_datasets_manifest_downloaded.yaml

python -m ste_neurips.neurips_suite real-suite \
  --manifest configs/real_datasets_manifest_downloaded.yaml \
  --out outputs/real_suite_final \
  --bootstrap 1000 \
  --seed 0
