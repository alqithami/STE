#!/usr/bin/env bash
# Task-local setup, detached CUDA launch, shard packaging, merge, and verification.
set -euo pipefail
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUN_ROOT=${STE_RUN_ROOT:-"$HOME/ste-harduc-confirm-v1"}
mkdir -p -- "$RUN_ROOT"
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd -P)
RESULTS_DIR="$RUN_ROOT/results"
export STE_RUN_ROOT="$RUN_ROOT"
export STE_COLLECTIONS=${STE_COLLECTIONS:-0,1,2,3,4,5,6,7,8,9,10,11}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export STE_PROGRESS_FILE="$RESULTS_DIR/PROGRESS.json"

verify_code() {
    [[ -s "$CODE_DIR/SHA256SUMS" ]] || { printf 'Missing SHA256SUMS; extract the complete delivered ZIP.\n' >&2; return 1; }
    (cd -- "$CODE_DIR" && sha256sum -c SHA256SUMS)
}

software_ok() {
    "$1" "$CODE_DIR/preflight.py" --root "$RUN_ROOT" --software-only >/dev/null 2>&1
}

setup_environment() {
    local candidate chosen="" previous_root installer uv_bin
    if [[ -n ${STE_PYTHON:-} ]]; then
        if [[ ! -x "$STE_PYTHON" ]]; then
            printf 'STE_PYTHON must be the full path of an executable Python interpreter.\n' >&2
            return 1
        fi
        "$STE_PYTHON" "$CODE_DIR/preflight.py" --root "$RUN_ROOT" --software-only
        chosen="$STE_PYTHON"
    fi
    if [[ -z "$chosen" && -s "$RUN_ROOT/PYTHON_PATH.txt" ]]; then
        IFS= read -r candidate < "$RUN_ROOT/PYTHON_PATH.txt"
        if [[ -x "$candidate" ]] && software_ok "$candidate"; then chosen="$candidate"; fi
    fi
    # Reuse a pinned completed-study environment without changing its packages.
    if [[ -z "$chosen" ]]; then
        for previous_root in \
            "$HOME/ste-uc-equalcompute-v1" \
            /mnt/caenl/active/ste-uc-equalcompute-v1 \
            /workspace/ste-uc-equalcompute-v1; do
            if [[ -s "$previous_root/PYTHON_PATH.txt" ]]; then
                IFS= read -r candidate < "$previous_root/PYTHON_PATH.txt"
                if [[ -x "$candidate" ]] && software_ok "$candidate"; then chosen="$candidate"; break; fi
            fi
        done
    fi
    if [[ -z "$chosen" ]]; then
        for candidate in \
            "$RUN_ROOT/.venv/bin/python" \
            "${VIRTUAL_ENV:-/nonexistent}/bin/python" \
            /mnt/caenl/active/ste-gpu-v1/.venv/bin/python \
            /mnt/caenl/active/ste-nextstage-v1/.venv/bin/python \
            /mnt/caenl/active/ste-nextstage-v1/STE_NextStage_v1/.venv/bin/python; do
            if [[ -x "$candidate" ]] && software_ok "$candidate"; then chosen="$candidate"; break; fi
        done
    fi
    if [[ -z "$chosen" ]]; then
        if [[ -e "$RUN_ROOT/.venv" ]]; then
            printf 'The task-local .venv exists but is incompatible. It was not deleted or modified.\n' >&2
            printf 'Set STE_PYTHON to a compatible interpreter or select a new STE_RUN_ROOT.\n' >&2
            return 1
        fi
        # A fresh host may ship Python 3.14. Install a compatible task-local Python,
        # leaving the operating-system Python and shell profiles unchanged.
        command -v curl >/dev/null 2>&1 || { printf 'curl is required for fresh-environment setup.\n' >&2; return 1; }
        mkdir -p -- "$RUN_ROOT/bootstrap" "$RUN_ROOT/tools"
        installer="$RUN_ROOT/bootstrap/uv-0.8.22-install.sh"
        curl --fail --show-error --silent --location \
            https://astral.sh/uv/0.8.22/install.sh --output "$installer"
        env UV_INSTALL_DIR="$RUN_ROOT/tools" UV_NO_MODIFY_PATH=1 sh "$installer"
        uv_bin="$RUN_ROOT/tools/uv"
        [[ -x "$uv_bin" ]] || { printf 'The uv installer did not produce the expected executable.\n' >&2; return 1; }
        export UV_CACHE_DIR="$RUN_ROOT/uv-cache"
        export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python"
        export UV_PYTHON_BIN_DIR="$RUN_ROOT/tools"
        "$uv_bin" python install 3.11.13
        "$uv_bin" venv --python 3.11.13 --seed "$RUN_ROOT/.venv"
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
    if [[ -n ${STE_PYTHON:-} ]]; then RUN_PYTHON="$STE_PYTHON"
    elif [[ -s "$RUN_ROOT/PYTHON_PATH.txt" ]]; then IFS= read -r RUN_PYTHON < "$RUN_ROOT/PYTHON_PATH.txt"
    else printf 'No environment recorded; run bash cloud.sh setup first.\n' >&2; return 1
    fi
    [[ -x "$RUN_PYTHON" ]] || { printf 'Recorded Python does not exist.\n' >&2; return 1; }
}

freeze_collections() {
    local requested previous
    requested=$("$RUN_PYTHON" -c 'import sys; from preflight import collections; print(",".join(map(str,collections(sys.argv[1]))))' "$STE_COLLECTIONS")
    if [[ -s "$RUN_ROOT/COLLECTIONS.txt" ]]; then
        IFS= read -r previous < "$RUN_ROOT/COLLECTIONS.txt"
        [[ "$previous" == "$requested" ]] || { printf 'This run root is already assigned to collections %s, not %s. Use a different run root.\n' "$previous" "$requested" >&2; return 1; }
    else
        printf '%s\n' "$requested" > "$RUN_ROOT/COLLECTIONS.txt.tmp"
        mv -- "$RUN_ROOT/COLLECTIONS.txt.tmp" "$RUN_ROOT/COLLECTIONS.txt"
    fi
    export STE_COLLECTIONS="$requested"
}

full_preflight() {
    local mode=${1:-full}
    local -a options=(--root "$RUN_ROOT" --collections "$STE_COLLECTIONS")
    if [[ -n ${STE_EXPECT_GPU_MODEL:-} ]]; then options+=(--expect-gpu "$STE_EXPECT_GPU_MODEL"); fi
    if [[ "$mode" == busy ]]; then options+=(--busy-only --report "$RUN_ROOT/GPU_IDLE_PREFLIGHT.json")
    else options+=(--report "$RUN_ROOT/CUDA_PREFLIGHT.json")
    fi
    "$RUN_PYTHON" "$CODE_DIR/preflight.py" "${options[@]}"
}

reserve_gpu() {
    if [[ ${STE_GPU_RESERVED:-} == yes ]]; then return; fi
    local gpu_uuid gpu_key
    full_preflight busy
    gpu_uuid=$("$RUN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_gpu_uuid"])' "$RUN_ROOT/GPU_IDLE_PREFLIGHT.json")
    gpu_key=${gpu_uuid//[^a-zA-Z0-9_-]/_}
    export STE_GPU_LOCK_FILE="/tmp/ste-harduc-confirm-$(id -u)-$gpu_key.lock"
    exec 8>"$STE_GPU_LOCK_FILE"
    flock -n 8 || { printf 'Another hard-UC pipeline has reserved this GPU.\n' >&2; return 1; }
    export STE_GPU_RESERVED=yes
}

archive_pipeline() {
    if [[ -s "$RESULTS_DIR/RUN_COMPLETE.json" ]]; then
        "$RUN_PYTHON" "$CODE_DIR/analyze.py" --results "$RESULTS_DIR" --config "$CODE_DIR/config.json"
        reserve_gpu
        full_preflight
        "$RUN_PYTHON" "$CODE_DIR/replay_selected.py" --results "$RESULTS_DIR" --config "$CODE_DIR/config.json" --device cuda:0
        "$RUN_PYTHON" "$CODE_DIR/package_results.py" --results "$RESULTS_DIR" --outroot "$RUN_ROOT" --config "$CODE_DIR/config.json"
        "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT"
        printf 'COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"
        printf 'ALL_ARCHIVES_VERIFIED\nShare: %s/STE_HardUC_Confirm_v1_REVIEW.zip\n' "$RUN_ROOT"
    elif [[ -s "$RESULTS_DIR/SHARD_COMPLETE.json" ]]; then
        reserve_gpu
        full_preflight
        "$RUN_PYTHON" "$CODE_DIR/replay_selected.py" --results "$RESULTS_DIR" --config "$CODE_DIR/config.json" --device cuda:0
        "$RUN_PYTHON" "$CODE_DIR/package_results.py" --results "$RESULTS_DIR" --outroot "$RUN_ROOT" --config "$CODE_DIR/config.json" --shard
        "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT" --shard
        printf 'SHARD_VERIFIED UTC=%s collections=%s\n' "$(date -u +%FT%TZ)" "$STE_COLLECTIONS" > "$RUN_ROOT/pipeline.state"
        printf 'SHARD_ARCHIVES_VERIFIED\nThis shard is ready to merge; the full experiment is not complete.\n'
        printf 'Shard archives: %s/STE_HardUC_Confirm_v1_SHARD_*.zip\n' "$RUN_ROOT"
    else
        printf 'Training is incomplete: neither RUN_COMPLETE.json nor SHARD_COMPLETE.json exists.\n' >&2
        return 1
    fi
}

run_pipeline() {
    load_python
    cd -- "$CODE_DIR"
    trap 'rc=$?; if (( rc != 0 )); then printf "FAILED exit=%s UTC=%s\n" "$rc" "$(date -u +%FT%TZ)" > "$RUN_ROOT/pipeline.state"; printf "PIPELINE_FAILED exit=%s; inspect results/run.log. Completion has not been claimed.\n" "$rc"; fi' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    printf '%s\n' "$$" > "$RUN_ROOT/run.pid"
    printf 'RUNNING UTC=%s collections=%s\n' "$(date -u +%FT%TZ)" "$STE_COLLECTIONS" > "$RUN_ROOT/pipeline.state"
    printf '=== STE hard UC confirmation production start %s ===\n' "$(date -u +%FT%TZ)"
    verify_code
    full_preflight
    "$RUN_PYTHON" "$CODE_DIR/run_experiment.py" --config "$CODE_DIR/config.json" --out "$RESULTS_DIR" --device cuda --collections "$STE_COLLECTIONS"
    archive_pipeline
}

command_name=${1:-help}
case "$command_name" in
    setup)
        verify_code
        setup_environment
        ;;
    start)
        command -v flock >/dev/null 2>&1 || { printf 'flock is required (util-linux).\n' >&2; exit 1; }
        exec 9>"$RUN_ROOT/run.lock"
        flock -n 9 || { printf 'This run root has an active pipeline; use status.\n' >&2; exit 1; }
        if [[ -s "$RUN_ROOT/pipeline.state" ]] && [[ -s "$RUN_ROOT/run.pid" ]]; then
            current_state=$(head -n 1 "$RUN_ROOT/pipeline.state")
            IFS= read -r current_pid < "$RUN_ROOT/run.pid"
            if [[ "$current_state" == STARTING* || "$current_state" == RUNNING* ]] && [[ "$current_pid" =~ ^[0-9]+$ ]] && kill -0 "$current_pid" 2>/dev/null; then
                printf 'The recorded launcher is active; use bash cloud.sh status.\n' >&2; exit 1
            fi
        fi
        if [[ -s "$RUN_ROOT/pipeline.state" ]] && [[ $(head -n 1 "$RUN_ROOT/pipeline.state") == COMPLETE* || $(head -n 1 "$RUN_ROOT/pipeline.state") == SHARD_VERIFIED* ]]; then
            printf 'This run or shard is already complete. Use bash cloud.sh verify.\n'; exit 0
        fi
        verify_code
        setup_environment
        load_python
        cd -- "$CODE_DIR"
        freeze_collections
        reserve_gpu
        full_preflight
        "$RUN_PYTHON" "$CODE_DIR/software_tests.py" --device cuda --out "$RUN_ROOT/CUDA_SOFTWARE_TESTS.json"
        printf 'Running separate CUDA software smoke; these outcomes are not study results.\n'
        STE_PROGRESS_FILE="$RUN_ROOT/smoke/PROGRESS.json" "$RUN_PYTHON" "$CODE_DIR/run_experiment.py" --config "$CODE_DIR/config.json" --out "$RUN_ROOT/smoke" --device cuda --smoke
        mkdir -p -- "$RESULTS_DIR"
        printf 'STARTING UTC=%s collections=%s\n' "$(date -u +%FT%TZ)" "$STE_COLLECTIONS" > "$RUN_ROOT/pipeline.state"
        nohup bash "$CODE_DIR/cloud.sh" __run >> "$RESULTS_DIR/run.log" 2>&1 < /dev/null &
        launch_pid=$!
        printf '%s\n' "$launch_pid" > "$RUN_ROOT/run.pid"
        exec 8>&- 9>&-
        launch_deadline=$((SECONDS + 15))
        while (( SECONDS < launch_deadline )); do
            launch_state=$(head -n 1 "$RUN_ROOT/pipeline.state")
            if [[ "$launch_state" == RUNNING* || "$launch_state" == COMPLETE* || "$launch_state" == SHARD_VERIFIED* ]]; then break; fi
            if [[ "$launch_state" == FAILED* ]] || ! kill -0 "$launch_pid" 2>/dev/null; then
                printf 'Detached launch failed; inspect %s/run.log.\n' "$RESULTS_DIR" >&2
                tail -n 20 "$RESULTS_DIR/run.log" >&2 || true
                exit 1
            fi
            sleep 0.1
        done
        if [[ "$launch_state" != RUNNING* && "$launch_state" != COMPLETE* && "$launch_state" != SHARD_VERIFIED* ]]; then
            printf 'Launcher readiness was not confirmed; inspect status before retrying.\n' >&2; exit 1
        fi
        printf 'LAUNCH_CONFIRMED PID=%s collections=%s\nLog: %s/run.log\nThis is launch confirmation, not completion.\n' "$launch_pid" "$STE_COLLECTIONS" "$RESULTS_DIR"
        ;;
    __run)
        exec 9>"$RUN_ROOT/run.lock"
        flock -w 15 9 || { printf 'Worker could not acquire the run lock.\n' >&2; exit 1; }
        [[ -n ${STE_GPU_LOCK_FILE:-} ]] || { printf 'Worker needs the GPU reservation created by start.\n' >&2; exit 1; }
        exec 8>"$STE_GPU_LOCK_FILE"
        flock -w 15 8 || { printf 'Worker could not acquire the GPU reservation.\n' >&2; exit 1; }
        export STE_GPU_RESERVED=yes
        run_pipeline
        ;;
    status)
        printf 'Run root: %s\n' "$RUN_ROOT"
        if [[ -s "$RUN_ROOT/pipeline.state" ]]; then cat "$RUN_ROOT/pipeline.state"; else printf 'NOT_STARTED\n'; fi
        if [[ -s "$RUN_ROOT/COLLECTIONS.txt" ]]; then printf 'Collection IDs: '; cat "$RUN_ROOT/COLLECTIONS.txt"; fi
        if [[ -s "$RUN_ROOT/run.pid" ]]; then
            IFS= read -r pid < "$RUN_ROOT/run.pid"
            if [[ "$pid" =~ ^[0-9]+$ ]]; then ps -p "$pid" -o pid=,etime=,stat=,args= || true; fi
        fi
        if [[ -s "$RESULTS_DIR/PROGRESS.json" ]]; then cat "$RESULTS_DIR/PROGRESS.json"; fi
        if [[ -s "$RESULTS_DIR/run.log" ]]; then tail -n 12 "$RESULTS_DIR/run.log"; fi
        printf '\nCompletion requires archive verification. SHARD_VERIFIED still requires merging all 12 collections.\n'
        ;;
    verify)
        load_python
        if [[ -s "$RUN_ROOT/SHARD_ARCHIVES_READY.json" ]] && [[ ! -s "$RESULTS_DIR/RUN_COMPLETE.json" ]]; then
            "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT" --shard
            printf 'SHARD_ARCHIVES_VERIFIED; full study completion has not been claimed.\n'
        else
            "$RUN_PYTHON" "$CODE_DIR/verify_archive.py" --outroot "$RUN_ROOT"
            printf 'ALL_ARCHIVES_VERIFIED\nShare: %s/STE_HardUC_Confirm_v1_REVIEW.zip\n' "$RUN_ROOT"
        fi
        ;;
    package)
        command -v flock >/dev/null 2>&1 || { printf 'flock is required.\n' >&2; exit 1; }
        exec 9>"$RUN_ROOT/run.lock"
        flock -n 9 || { printf 'An active pipeline holds the run lock.\n' >&2; exit 1; }
        verify_code
        load_python
        if [[ -s "$RUN_ROOT/COLLECTIONS.txt" ]]; then IFS= read -r STE_COLLECTIONS < "$RUN_ROOT/COLLECTIONS.txt"; fi
        archive_pipeline
        ;;
    merge)
        shift
        (( $# >= 2 )) || { printf 'Usage: bash cloud.sh merge SHARD_A.zip SHARD_B.zip [additional shards]\n' >&2; exit 1; }
        command -v flock >/dev/null 2>&1 || { printf 'flock is required.\n' >&2; exit 1; }
        exec 9>"$RUN_ROOT/run.lock"
        flock -n 9 || { printf 'An active pipeline holds the run lock.\n' >&2; exit 1; }
        verify_code
        setup_environment
        load_python
        "$RUN_PYTHON" "$CODE_DIR/merge.py" --inputs "$@" --out "$RESULTS_DIR"
        archive_pipeline
        ;;
    *)
        printf 'Usage: bash cloud.sh {setup|start|status|package|verify|merge SHARD_A.zip SHARD_B.zip}\n'
        printf 'Default run root: $HOME/ste-harduc-confirm-v1\n'
        printf 'Shard A: STE_COLLECTIONS=0,1,2,3,4,5 bash cloud.sh start\n'
        printf 'Shard B: STE_COLLECTIONS=6,7,8,9,10,11 bash cloud.sh start\n'
        ;;
esac
