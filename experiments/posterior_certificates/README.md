# STE posterior/certificates: corrected forward software v2

This directory is a self-contained **future-run** implementation of the
exploratory human-profile posterior UC benchmark. The completed study used
immutable v1 software. Its native findings and the exclusion of the faulty
secondary mixture hard-q diagnostic are recorded in `CORRECTION_NOTICE.md`.
No v2 production results or retraining are claimed. Historical code/results
must remain unchanged and must be replayed with the original software.

The full pipeline includes observed-count training, source-held-out development
selection, eight native/secondary systems, saved predictions and checkpoints,
selected-state and saved-draw replay, raw-count reconstruction, budget/coverage
checks, descriptive summaries and verified archive packaging. Actual PrefLib
ballots and original profile metadata are bundled under `reference/data/`.
The earlier root-repository UC/TC operators are archival and are not imported
by this pipeline. The active strict UC and guarded TC utility use validated
graph contracts; this learning benchmark reports UC endpoints only.

Read `PROTOCOL.md` and `CORRECTION_NOTICE.md` before interpreting results.
The cohort has six sources and 36 profiles, but only three selective targets.
No p-values, population superiority, posterior calibration or acceptance claims
follow from this benchmark. Expected-F1 GFM decoding is established prior art.

## Software verification

Use Python 3.11 or 3.12 with NumPy, SciPy and a compatible PyTorch installation.
Requirements are ranges; each run records actual versions and its device.

```bash
cd experiments/posterior_certificates
python3 -m venv .venv
. .venv/bin/activate
# Install a PyTorch build appropriate for your CPU/GPU using its official index.
python -m pip install -r requirements.txt
sha256sum -c SHA256SUMS
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Software-only CPU smoke (fitting is QA, never production evidence):

```bash
python -u run_experiment.py --config config.json \
  --out "$HOME/ste-posterior-certificates-v2-software-qa/smoke" \
  --device cpu --smoke
python package_results.py \
  --out "$HOME/ste-posterior-certificates-v2-software-qa/smoke"
```

Smoke refuses an existing output directory. The pipeline and archive packager
refuse preexisting destinations; preserve failure evidence and inspect it before
using a newly named root. V2 rejects the historical v1 namespace and checkpoints.
There is no restart/resume operation that merges an incomplete run with new work.

## Fresh CUDA run

Install the official compatible CUDA PyTorch build in your environment, then
select its executable and a **new** v2 result base. This command starts actual
experiments; it is not required to inspect the completed v1 evidence.

```bash
cd experiments/posterior_certificates
export STE_PYTHON="$PWD/.venv/bin/python"
export STE_RESULTS_BASE="$HOME/ste-posterior-certificates-v2-new-run"
sha256sum -c SHA256SUMS
bash run_all.sh pipeline 2>&1 | tee "$HOME/ste-posterior-certificates-v2-new-run.log"
```

The wrapper checks for an active GPU process and a concurrent lock, runs tests
and separately audited fixed-count certificate diagnostics, then requires a
successful smoke audit before pilot and production. Freeze source/config before
launch. GPU fitting is mandatory for pilot and production; no silent CPU
fallback is allowed. The upper allocation is 120 seconds per system/fold/init,
with mixture candidate search sharing that allocation; unused scalar-fit time
is reported, not padded. Budget overruns preserve exploratory outputs and mark
allocation validity false.

## Completion and downloads

`FIT_COMPLETE.json` means fitting ended. `COMPLETE.json` requires raw-count and
coverage checks, source/data/checkpoint hashes, selected-state replay, saved-draw
reconstruction and metric aggregation checks. Idle GPU usage is not completion.
Production interpretation additionally requires scientific inspection of the
raw artifacts and limitations; a passed internal audit alone is insufficient.

Rechecking completed v2 evidence writes a new external report, preserving the
original completion/audit markers:

```bash
python audit_results.py --out "$STE_RESULTS_BASE/production" --device cpu \
  --replay-report "$HOME/ste-posterior-certificates-v2-external-replay.json"
```

```bash
cat "$STE_RESULTS_BASE/STATUS.json"
cat "$STE_RESULTS_BASE/production/COMPLETE.json"
tail -f "$HOME/ste-posterior-certificates-v2-new-run.log"
cd "$STE_RESULTS_BASE"
sha256sum -c STE_Posterior_Certificates_v2_production_FULL.zip.sha256
sha256sum -c STE_Posterior_Certificates_v2_production_REVIEW.zip.sha256
```

The full archive retains checkpoints and raw joint UC samples. The smaller
review archive omits those large arrays and cannot replace full replay. Copy
both ZIPs and their checksum files from your chosen SSH host, then verify the
same hashes on the destination. On macOS use `shasum -a 256 -c FILE.zip.sha256`.

`RESULTS_CERTIFICATES_V1.md` is explicitly historical fixed-model diagnostic
evidence. `THEORY.md` and `THEORY.tex` retain full supporting proofs and their
assumptions. Neither is a report of v2 learning results.
