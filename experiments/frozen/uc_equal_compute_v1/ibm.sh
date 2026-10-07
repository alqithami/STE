#!/usr/bin/env bash
# Task-local setup, detached launch, progress, and verified result packaging.
set -euo pipefail
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUN_ROOT=${STE_RUN_ROOT:-/mnt/caenl/active/ste-uc-equalcompute-v1}
mkdir -p -- "$RUN_ROOT"
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd -P)
RESULTS_DIR="$RUN_ROOT/results"
export STE_RUN_ROOT="$RUN_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export STE_PROGRESS_FILE="$RESULTS_DIR/PROGRESS.json"

verify_code() {
    if [[ ! -f "$CODE_DIR/SHA256SUMS" ]]; then
        printf 'Missing SHA256SUMS; extract the complete delivered archive.\n' >&2
        return 1
    fi
    (cd -- "$CODE_DIR" && sha256sum -c SHA256SUMS)
}

software_ok() {
    "$1" "$CODE_DIR/preflight.py" --root "$RUN_ROOT" --software-only >/dev/null 2>&1
}

setup_environment() {
    local candidate chosen="" base=""
    if [[ -s "$RUN_ROOT/PYTHON_PATH.txt" ]]; then
        IFS= read -r candidate < "$RUN_ROOT/PYTHON_PATH.txt"
        if [[ -x "$candidate" ]] && software_ok "$candidate"; then
            chosen="$candidate"
        fi
    fi
    if [[ -z "$chosen" ]]; then
        for candidate in \
            "$RUN_ROOT/.venv/bin/python" \
            /mnt/caenl/active/ste-gpu-v1/.venv/bin/python \
            /mnt/caenl/active/ste-gpu-v1/STE_IBM_GPU_v1/.venv/bin/python \
            /mnt/caenl/active/STE_IBM_GPU_v1/.venv/bin/python \
            /mnt/caenl/active/ste-ibm-gpu-v1/.venv/bin/python \
            /mnt/caenl/active/ste-ibm-gpu-v1/STE_IBM_GPU_v1/.venv/bin/python \
            /mnt/caenl/active/ste-nextstage-v1/.venv/bin/python \
            /mnt/caenl/active/ste-nextstage-v1/STE_NextStage_v1/.venv/bin/python; do
            if [[ -x "$candidate" ]] && software_ok "$candidate"; then
                chosen="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$chosen" ]]; then
        for candidate in python3.11 python3.12 python3.10 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && \
                "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else 1)'; then
                base=$(command -v "$candidate")
                break
            fi
        done
        if [[ -z "$base" ]]; then
            printf 'No supported Python found. Install Python 3.11 with venv support.\n' >&2
            return 1
        fi
        printf 'Creating a task-local environment at %s/.venv\n' "$RUN_ROOT"
        "$base" -m venv "$RUN_ROOT/.venv"
        chosen="$RUN_ROOT/.venv/bin/python"
        "$chosen" -m pip install --upgrade pip==25.0.1
        "$chosen" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
        "$chosen" -m pip install -r "$CODE_DIR/requirements.txt"
    fi
    "$chosen" "$CODE_DIR/preflight.py" --root "$RUN_ROOT" --software-only
    printf '%s\n' "$chosen" > "$RUN_ROOT/PYTHON_PATH.txt.tmp"
    mv -- "$RUN_ROOT/PYTHON_PATH.txt.tmp" "$RUN_ROOT/PYTHON_PATH.txt"
    printf 'PYTHON_ENVIRONMENT_READY: %s\n' "$chosen"
}

load_python() {
    if [[ ! -s "$RUN_ROOT/PYTHON_PATH.txt" ]]; then
        printf 'No environment recorded; run bash ibm.sh setup or start first.\n' >&2
        return 1
    fi
    IFS= read -r RUN_PYTHON < "$RUN_ROOT/PYTHON_PATH.txt"
    [[ -x "$RUN_PYTHON" ]] || { printf 'Recorded Python does not exist.\n' >&2; return 1; }
}

run_pipeline() {
    load_python
    cd -- "$CODE_DIR"
    trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s UTC=%s\n" "$rc" "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"; printf "PIPELINE_FAILED exit=%s; inspect results/run.log. Completion has not been claimed.\n" "$rc"; fi' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    printf '%s\n' "$$" > "$RUN_ROOT/run.pid"
    printf 'RUNNING UTC=%s\n' "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"
    printf '=== STE UC equal-compute production start %s ===\n' "$(date -u +%FT%TZ)"
    verify_code
    "$RUN_PYTHON" "$CODE_DIR/preflight.py" --root "$RUN_ROOT"
    "$RUN_PYTHON" "$CODE_DIR/run_experiment.py" --config "$CODE_DIR/config.json" --out "$RESULTS_DIR" --device cuda
    "$RUN_PYTHON" "$CODE_DIR/analyze.py" --results "$RESULTS_DIR" --config "$CODE_DIR/config.json"
    "$RUN_PYTHON" "$CODE_DIR/package_results.py" --results "$RESULTS_DIR" --outroot "$RUN_ROOT" --config "$CODE_DIR/config.json"
    "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT"
    printf 'COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"
    printf 'ALL_ARCHIVES_VERIFIED\nShare: %s/STE_UC_EqualCompute_v1_REVIEW.zip\n' "$RUN_ROOT"
}

case "${1:-help}" in
    setup)
        verify_code
        setup_environment
        ;;
    start)
        command -v flock >/dev/null 2>&1 || { printf 'flock is required (util-linux).\n' >&2; exit 1; }
        exec 9>"$RUN_ROOT/run.lock"
        flock -n 9 || { printf 'This experiment already has an active launcher; use status.\n' >&2; exit 1; }
        if [[ -s "$RUN_ROOT/pipeline.state" ]] && [[ -s "$RUN_ROOT/run.pid" ]]; then
            current_state=$(head -n 1 "$RUN_ROOT/pipeline.state")
            IFS= read -r current_pid < "$RUN_ROOT/run.pid"
            if [[ "$current_state" == STARTING* || "$current_state" == RUNNING* ]] && \
                [[ "$current_pid" =~ ^[0-9]+$ ]] && kill -0 "$current_pid" 2>/dev/null; then
                printf 'The recorded launcher is active; use bash ibm.sh status.\n' >&2
                exit 1
            fi
        fi
        if [[ -s "$RUN_ROOT/pipeline.state" ]] && [[ $(head -n 1 "$RUN_ROOT/pipeline.state") == COMPLETE* ]]; then
            printf 'This pipeline is already complete. Use bash ibm.sh verify; do not rerun completed work.\n'
            exit 0
        fi
        verify_code
        setup_environment
        load_python
        "$RUN_PYTHON" "$CODE_DIR/preflight.py" --root "$RUN_ROOT"
        "$RUN_PYTHON" "$CODE_DIR/software_tests.py" --device cuda --out "$RUN_ROOT/CUDA_SOFTWARE_TESTS.json"
        printf 'Running separate short CUDA software smoke; its outcomes are not study results.\n'
        STE_PROGRESS_FILE="$RUN_ROOT/smoke/PROGRESS.json" "$RUN_PYTHON" "$CODE_DIR/run_experiment.py" --config "$CODE_DIR/config.json" --out "$RUN_ROOT/smoke" --device cuda --smoke
        mkdir -p -- "$RESULTS_DIR"
        printf 'STARTING UTC=%s\n' "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"
        nohup bash "$CODE_DIR/ibm.sh" __run >> "$RESULTS_DIR/run.log" 2>&1 < /dev/null &
        launch_pid=$!
        printf '%s\n' "$launch_pid" > "$RUN_ROOT/run.pid"
        # The child opens and owns its own flock. Do not depend on fd inheritance
        # across nohup/exec, which some execution environments intentionally close.
        exec 9>&-
        launch_deadline=$((SECONDS + 15))
        while (( SECONDS < launch_deadline )); do
            launch_state=$(head -n 1 "$RUN_ROOT/pipeline.state")
            if [[ "$launch_state" == RUNNING* || "$launch_state" == COMPLETE* ]]; then
                break
            fi
            if [[ "$launch_state" == FAILED* ]] || ! kill -0 "$launch_pid" 2>/dev/null; then
                printf 'Detached launch failed; inspect %s/run.log.\n' "$RESULTS_DIR" >&2
                tail -n 20 "$RESULTS_DIR/run.log" >&2 || true
                exit 1
            fi
            sleep 0.1
        done
        if [[ "$launch_state" != RUNNING* && "$launch_state" != COMPLETE* ]]; then
            printf 'Launcher readiness was not confirmed; inspect bash ibm.sh status before retrying.\n' >&2
            exit 1
        fi
        printf 'LAUNCH_CONFIRMED PID=%s\nLog: %s/run.log\nThis is launch confirmation, not completion.\n' "$launch_pid" "$RESULTS_DIR"
        ;;
    __run)
        # Internal worker: hold an independently opened lock for the whole pipeline.
        exec 9>"$RUN_ROOT/run.lock"
        flock -w 15 9 || { printf 'Worker could not acquire the run lock.\n' >&2; exit 1; }
        run_pipeline
        ;;
    status)
        printf 'Run root: %s\n' "$RUN_ROOT"
        if [[ -s "$RUN_ROOT/pipeline.state" ]]; then cat "$RUN_ROOT/pipeline.state"; else printf 'NOT_STARTED\n'; fi
        if [[ -s "$RUN_ROOT/run.pid" ]]; then
            IFS= read -r pid < "$RUN_ROOT/run.pid"
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                ps -p "$pid" -o pid=,etime=,stat=,args= || true
            fi
        fi
        if [[ -s "$RESULTS_DIR/PROGRESS.json" ]]; then cat "$RESULTS_DIR/PROGRESS.json"; fi
        if [[ -s "$RESULTS_DIR/run.log" ]]; then tail -n 12 "$RESULTS_DIR/run.log"; fi
        printf '\nProduction completion requires bash ibm.sh verify to succeed.\n'
        ;;
    verify)
        load_python
        "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT"
        printf 'ALL_ARCHIVES_VERIFIED\nShare: %s/STE_UC_EqualCompute_v1_REVIEW.zip\n' "$RUN_ROOT"
        ;;
    package)
        # Recovery if training completed but analysis or archive creation was interrupted.
        command -v flock >/dev/null 2>&1 || { printf 'flock is required.\n' >&2; exit 1; }
        exec 9>"$RUN_ROOT/run.lock"
        flock -n 9 || { printf 'An active pipeline holds the run lock.\n' >&2; exit 1; }
        verify_code
        load_python
        [[ -s "$RESULTS_DIR/RUN_COMPLETE.json" ]] || { printf 'Training is incomplete; package recovery is not available.\n' >&2; exit 1; }
        "$RUN_PYTHON" "$CODE_DIR/analyze.py" --results "$RESULTS_DIR" --config "$CODE_DIR/config.json"
        "$RUN_PYTHON" "$CODE_DIR/package_results.py" --results "$RESULTS_DIR" --outroot "$RUN_ROOT" --config "$CODE_DIR/config.json"
        "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT"
        printf 'COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"
        printf 'ALL_ARCHIVES_VERIFIED\n'
        ;;
    *)
        printf 'Usage: bash ibm.sh {start|status|verify|package|setup}\n'
        printf 'Default root: /mnt/caenl/active/ste-uc-equalcompute-v1\n'
        printf 'For RunPod: STE_RUN_ROOT=/workspace/ste-uc-equalcompute-v1 bash ibm.sh start\n'
        ;;
esac
