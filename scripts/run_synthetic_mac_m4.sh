#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ste_neurips.neurips_suite synthetic --config configs/synthetic_mac.yaml
