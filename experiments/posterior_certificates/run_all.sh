#!/usr/bin/env bash
set -Eeuo pipefail

task_code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_python="${STE_PYTHON:-$(command -v python3)}"
task_results_base="${STE_RESULTS_BASE:-$HOME/ste-posterior-certificates-v2}"
task_mode="${1:-pipeline}"
case "$task_mode" in pipeline|smoke|pilot|production) ;; *) echo "Use pipeline, smoke, pilot, or production" >&2; exit 2;; esac
test -x "$task_python"
cd "$task_code_root"
"$task_python" - "$task_results_base" <<'PYGUARD'
import sys
from pathlib import Path
from run_experiment import validate_future_output_path
base=validate_future_output_path(sys.argv[1])
if base.exists() and (not base.is_dir() or any(base.iterdir())):
    raise SystemExit('Existing/nonempty result base refused before writing any pipeline markers; use a fresh root')
PYGUARD
cd "$task_code_root"
sha256sum -c SHA256SUMS
mkdir -p "$task_results_base"
exec 9>"$task_results_base/pipeline.lock"
flock -n 9 || { echo "Another instance already owns this pipeline" >&2; exit 3; }
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=20261007 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '%s\n' "$$" > "$task_results_base/pipeline.pid"

task_failed() {
  task_exit_code=$?
  "$task_python" - "$task_results_base" "$task_exit_code" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);(p/'PIPELINE_FAILED.json').write_text(json.dumps({'exit_code':int(sys.argv[2]),'time_unix':time.time(),'instruction':'Preserve all outputs; inspect the failed stage. Never restart into its existing directory.'},indent=2))
PY
  exit "$task_exit_code"
}
trap task_failed ERR

task_run_stage() {
  local task_stage="$1"
  local task_out="$task_results_base/$task_stage"
  test ! -e "$task_out" || { echo "Existing output refused: $task_out" >&2; return 4; }
  if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
    echo "GPU has an active compute process; refusing to start $task_stage" >&2
    return 5
  fi
  local -a task_options=()
  if test "$task_stage" = smoke; then task_options+=(--smoke); fi
  if test "$task_stage" = pilot; then task_options+=(--pilot); fi
  "$task_python" - "$task_results_base" "$task_stage" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);(p/'STATUS.json').write_text(json.dumps({'stage':sys.argv[2],'status':'running','time_unix':time.time()},indent=2))
PY
  "$task_python" -u run_experiment.py --config config.json --out "$task_out" --device cuda "${task_options[@]}"
  "$task_python" - "$task_out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=p/'COMPLETE.json'
if not m.exists():raise SystemExit('No audited COMPLETE marker: refusing to continue')
report=json.loads(m.read_text())
print('STAGE_COMPLETE',p.name,json.dumps(report,sort_keys=True))
PY
  "$task_python" package_results.py --out "$task_out"
}

if test "$task_mode" = pipeline; then
  "$task_python" -m unittest discover -s tests -v
  "$task_python" run_diagnostics.py --out "$task_results_base/certificate_diagnostics"
  "$task_python" audit_diagnostics.py --out "$task_results_base/certificate_diagnostics" --audit "$task_results_base/CERTIFICATE_DIAGNOSTIC_AUDIT.json"
  "$task_python" - "$task_results_base" <<'PY'
import json,sys
from pathlib import Path
from package_results import make_archive
base=Path(sys.argv[1]);out=base/'certificate_diagnostics'
report=json.loads((base/'CERTIFICATE_DIAGNOSTIC_AUDIT.json').read_text())
if report.get('status')!='PASS':raise SystemExit('Certificate diagnostic audit failed')
payload={'certificate_diagnostics/'+p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
payload['CERTIFICATE_DIAGNOSTIC_AUDIT.json']=(base/'CERTIFICATE_DIAGNOSTIC_AUDIT.json').read_bytes()
print(json.dumps(make_archive(base/'STE_Posterior_Certificates_v2_certificate_DIAGNOSTICS.zip',payload,report)))
PY
  task_run_stage smoke
  task_run_stage pilot
  task_run_stage production
else
  task_run_stage "$task_mode"
fi
sha256sum -c SHA256SUMS
"$task_python" - "$task_results_base" "$task_mode" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);value={'status':'completed','mode':sys.argv[2],'time_unix':time.time(),'interpretation':'Only production is a completed exploratory learning study; smoke is software QA and pilot is timing/feasibility.'}
(p/'STATUS.json').write_text(json.dumps(value,indent=2))
if sys.argv[2]=='pipeline':(p/'PIPELINE_COMPLETE.json').write_text(json.dumps(value,indent=2))
PY
