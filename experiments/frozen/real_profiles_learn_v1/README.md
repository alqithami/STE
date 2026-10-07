# Genuine human-profile learning, exploratory

Run this package independently of the synthetic hard-decoding confirmation.
Read `PROTOCOL.md` before interpreting its outputs. It contains 36 genuine
strict-majority human preference profiles, only three selective UC references,
and a completely frozen six-source holdout design. It can establish whether
the existing model learns useful reconstruction behavior from real records; it
cannot establish population-core truth or broad real-data superiority.

All raw input files are already bundled. No dataset download or authentication
is required. The model and tournament operators are unchanged copies of the
paper's audited implementation. The production allocation is 2 hours 24 minutes
of training/development, plus baseline/evaluation/packaging. Output archives
include real counts, selected weights, candidate budgets and all outcomes.

## Existing GPU environment

The already verified environment is:

```bash
export STE_PYTHON="$HOME/ste-uc-equalcompute-v1/.venv/bin/python"
cd "$HOME/STE_RealProfiles_Learn_v1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$STE_PYTHON" run.py --device cuda --stage preflight
"$STE_PYTHON" run.py --device cuda --smoke --out "$HOME/STE_real_gpu_smoke"
```

The smoke output is software verification only. Do not put its values in the
paper. After CUDA smoke passes, the independently allocated production run is:

```bash
nohup bash run_all.sh > launch.log 2>&1 &
echo "$!" > launch.pid
tail -F launch.log
```

Run one production GPU job per GPU. Do not overlap this job with the primary
synthetic confirmation on the same GPU: unequal contention would compromise
the measured wall-budget comparison. A separate GPU server can run it in
parallel; otherwise queue it after the synthetic job and check remaining time.

## Completion and sharing

From a second terminal on the server:

```bash
cd "$HOME/STE_RealProfiles_Learn_v1"
"$HOME/ste-uc-equalcompute-v1/.venv/bin/python" run.py --out results --stage status
```

Completion requires all six source folds and successful full hash verification.
The final launch log must show `REAL_LEARNING_ARCHIVES_READY`; an idle GPU or
the background process ending does not prove success. If interrupted, rerun
`bash run_all.sh`: incomplete candidates restart from their paired seed and
completed candidates/folds are reused only after checksum verification.

The two archives are `STE_RealProfiles_Learn_v1_REVIEW.zip` and
`STE_RealProfiles_Learn_v1_FULL.zip`, each with a `.zip.sha256` companion.
The review archive omits weight files but contains all development decisions,
test predictions, counts, manifests and tables. The full archive also contains
every saved candidate weight and optimizer state. Both are verified internally
before the ready marker is printed. Share the review ZIP first and retain the
full ZIP for independent replay. Negative outcomes are retained automatically.

## Fresh GPU environment

Use Python 3.11 or 3.12. The existing Python 3.14 system interpreter is not the
verified experiment environment. A compatible PyTorch CUDA environment on a
second server/RunPod can install `requirements.txt`; select a CUDA-capable
PyTorch 2.6 build compatible with the installed driver. Run the CUDA preflight
and smoke before production. The driver reported CUDA version does not mean
that a Python CUDA toolkit or PyTorch build is already installed.

## Tables

`DESCRIPTIVE_PRIMARY_SUMMARY.csv` reports all-class source-macro F1 at 10%.
`SOURCE_MEANS.csv` and `DESCRIPTIVE_SOURCE_DIFFERENCES.csv` retain source-level
outcomes. `THREE_SELECTIVE_PROFILE_MEANS.csv` reports the three non-singleton
references individually. `PROFILE_MEANS.csv` retains exact recovery and
cardinality. No p-values, confidence intervals, or acceptance claims are
automatically generated. Read the scope/interpretation JSON files alongside the
tables. Keep this exploratory study distinct from the paper's existing primary
selective soft-set test.
