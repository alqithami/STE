# STE Posterior Certificates v1

A separate exploratory research extension of the submitted STE paper. It tests
whether learning a conditional distribution of tournament orientations helps
reconstruct a finite human profile's uncovered set, against direct count,
Jeffreys, empirical-Bayes and native neural baselines. The submitted manuscript,
old code and old endpoints are unchanged.

Read `PROTOCOL.md` before interpreting outputs. The cohort was examined earlier;
there are only three selective references among 36 profiles. No p-values or
population/general superiority claims are authorized. The learned mixture is a
predictive distribution, not automatically a calibrated Bayesian posterior.
Expected-F1 decoding is established prior art.

## Contents

- `run_experiment.py`: observed-count training, source-held-out selection and
  inference; CUDA is mandatory for pilot and production.
- `audit_results.py`: raw-ballot/count reconstruction, full coverage checks,
  selected-checkpoint replay, saved-draw regeneration, hash checks and independent
  recomputation of every reported summary. `COMPLETE.json` requires this audit.
- `run_diagnostics.py`: a separate fixed IID Bernoulli certificate diagnostic;
  `audit_diagnostics.py` independently reconstructs it without importing `stepc`.
- `stepc/`: posterior UC kernels, logical certificates and learning systems.
- `THEORY.md` / `THEORY.tex`: full supporting proofs and explicit assumptions;
  theorem novelty has not been established.
- `reference/`: byte-preserved public cohort files and the earlier UC operator.
- `tests/`: exhaustive small-graph, posterior, gradient, parser and split tests.
- `config.json`: fixed exploratory protocol and budgets.
- `run_all.sh`: guarded smoke → timing pilot → production pipeline.
- `package_results.py`: verified full and compact-review archives.

## Run

Use the existing CUDA-enabled Python environment on the authorized server:

```bash
cd /home/ubuntu/STE_Posterior_Certificates_v1
sha256sum -c SHA256SUMS
export STE_PYTHON=/home/ubuntu/ste-uc-equalcompute-v1/.venv/bin/python3.11
bash run_all.sh pipeline
```

The complete pipeline runs software tests and the fixed synthetic certificate
diagnostic first. A smoke audit must succeed before the timing pilot, and the
pilot audit must succeed before production. Stages refuse an existing output
directory, an active GPU job or a concurrent pipeline lock. A failed stage's
evidence must be preserved; restart with a new result base only after inspection.
Do not change source or configuration after the source manifest is frozen.

Production allocates three hours of fitting work across five systems, six folds
and three paired initializations. Data preparation, held-out Monte Carlo
inference, checkpoint replay, analysis and packaging add time. This allocation
is not a wall-clock completion promise. The pilot is 15 minutes of allocated
fitting work plus its overhead. Smoke and pilot are not production findings.

For another CUDA host, create a Python 3.11 environment and install a compatible
CUDA PyTorch build, NumPy and SciPy. `requirements.txt` records accepted ranges;
the actual interpreter, Torch/CUDA/device and package versions are preserved in
the run lock. CPU fallback is allowed only for software smoke or replay audit.

## Completion and sharing

`FIT_COMPLETE.json` only means fitting finished. Scientific review requires
`COMPLETE.json`, payload hash verification and selected-state replay, plus reading
the raw predictions and independently reconstructed source/profile summaries.
Idle GPU usage is not evidence of completion. An overrun retains descriptive
outputs with `exploratory_budget_deviation`; it does not establish equal compute.

```bash
python audit_results.py --help
python package_results.py --out /home/ubuntu/ste-posterior-certificates-v1/production
```

The full ZIP keeps selected checkpoints and raw UC draws. The compact review ZIP
omits those large arrays and cannot replace a full audit. Each ZIP's internal
payload hashes and CRCs are checked and an external `.sha256` is written.
Commands for this user's existing SSH connection are in `LAUNCH_AND_SHARE.txt`.
