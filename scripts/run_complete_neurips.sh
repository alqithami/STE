#!/usr/bin/env bash
set -euo pipefail

# Complete STE experiment run for the NeurIPS paper.
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/run_complete_neurips.sh
# The HF token is only needed for gated datasets such as lmsys/chatbot_arena_conversations.

python -m pytest -q
bash scripts/run_smoke.sh
bash scripts/run_final_neurips.sh

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

bash scripts/package_reviewer_artifact.sh
