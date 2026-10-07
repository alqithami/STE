#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
TASK_PYTHON="${STE_PYTHON:-python3}"
mkdir -p results
exec 9>results/RUN_PROCESS.lock
flock -n 9 || { echo 'Another real-learning run holds this output lock.'; exit 1; }
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$TASK_PYTHON" run.py --device cuda --out results --stage preflight
"$TASK_PYTHON" run.py --device cuda --out results --stage run 2>&1 | tee -a results/RUN.log
"$TASK_PYTHON" package_results.py --out results 2>&1 | tee -a results/RUN.log
