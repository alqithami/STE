# STE hard-UC confirmation

This is a new, prospective synthetic experiment. It does not replace the completed
soft-set experiment, and it does not establish human-preference or LLM transfer.
`PROTOCOL.md` defines the fixed scientific comparisons. `LAUNCH_AND_SHARE.md`
contains the complete upload, launch, monitoring, merge and download workflow.

## What actually runs

Twelve fresh independent collections and three paired initializations compare six
separately timed systems: pair-hard, ordinary direct head, relational direct head,
STE-hard, ordinary auxiliary-hard and relational auxiliary-hard. Every system is
selected for its deployed readout on development data. Hard UC uses the predicted
majority relation and no membership cutoff. The structural UC training loss remains
the frozen STE operator; the hard decoder is used for selection and deployment.

The primary family contains only the two selective UC F1 contrasts at size 24:
STE-hard against the two native direct heads. The common-hard controls and size 48
are descriptive secondary analyses. Statistical inference uses independent
collections, averaging the three initializations within each collection. The
protocol and analysis retain negative, mixed and invalid-budget outcomes.

There are no fixed result arrays or placeholder outputs. The pipeline writes trained
checkpoints, development decisions, test predictions, measured budgets, data hashes,
replay checks, descriptive summaries and the prespecified statistical analysis.
CUDA software tests and the short smoke run are explicitly separate from study
results. They do not provide acceptance evidence.

## Hardware and time

Use a free NVIDIA L40S on each production server. Two independent servers can run
collections 0–5 and 6–11 concurrently. All shards must use the same GPU model,
capability, pinned numerical environment and identical delivered source. The merge
rejects incompatible or duplicate collection shards. Do not split an individual
collection across servers or run two shards concurrently on the same GPU.

The fixed candidate allocations total **13 hours 12 minutes** on one GPU or
**6 hours 36 minutes per GPU** on two equally sized shards. These figures exclude
setup, software QA, data generation, test evaluation, packaging and verification.
They are allocations, not promised completion times. A measured budget outside the
fixed 2% tolerance prevents confirmatory equal-compute inference; the outcomes are
still reported descriptively.

Production uses one CUDA device, float32, disabled TF32, and no mixed precision.
CPU production is refused. This package does not change the GPU driver, stop other
jobs, or delete existing experiment data.

## Environment and commands

Required scientific packages are torch 2.6.0+cu124, numpy 1.26.4, scipy 1.13.1 and
pandas 2.2.3, with Python 3.10–3.12. Set `STE_PYTHON` to a compatible existing
interpreter to reuse it without modifying that environment. Otherwise setup checks
the saved environment of the prior UC study, then creates a task-local environment.
For a fresh host, it installs uv 0.8.22 and Python 3.11.13 under the new run root
using the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/).
It leaves system Python and shell profiles unchanged. Setup needs network access
to Astral/GitHub, PyPI and the PyTorch wheel index only when an environment is new.

```bash
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1"
bash cloud.sh setup
bash cloud.sh start
bash cloud.sh status
tail -n 30 -F "$STE_RUN_ROOT/results/run.log"
```

Press Ctrl-C to stop `tail`; the detached experiment continues.

`STE_COLLECTIONS` defaults to all twelve IDs. For two servers, use respectively
`0,1,2,3,4,5` and `6,7,8,9,10,11` before `start`. Collection IDs are frozen per run
root, and each shard runs all six systems and all three initializations. You may set
`STE_EXPECT_GPU_MODEL='NVIDIA L40S'` to require this exact reported model at launch.
`CUDA_VISIBLE_DEVICES` defaults to 0 and must identify exactly one device.

```bash
bash cloud.sh package  # Recover analysis/packaging after training completed.
bash cloud.sh verify   # Independently check the archive and its manifest.
```

`start` checks delivered hashes, software pins, other GPU compute jobs, free storage,
CUDA arithmetic, software tests and a short smoke run before detached production.
A root lock prevents duplicate launches; a device lock prevents other copies of this
package from reserving the same GPU under the same user. The preflight checks other
compute jobs before creating a torch CUDA context. It refuses a busy GPU and never
stops another process. At least 8 GiB must be free after environment setup; provision
more for dependencies, download caches, checkpoints and archives.

## What completion means

- `LAUNCH_CONFIRMED` means only that a detached worker started.
- `SHARD_VERIFIED` means a partial collection shard has completed and its archive
  passed verification. It is **not** completion of the twelve-collection study.
- `COMPLETE` plus a successful `bash cloud.sh verify` means all twelve collections,
  the prescribed analysis, integrity checks and final archives have completed.
- `FAILED` is not completion. Retain the files and inspect the last log lines;
  do not erase a partial run or relaunch a finished shard to seek different results.

For two servers, download both shard ZIPs and checksums, place them on one server,
then merge into a new run root:

```bash
export STE_RUN_ROOT="$HOME/ste-harduc-confirm-v1-merged"
bash cloud.sh merge /full/path/to/SHARD_A.zip /full/path/to/SHARD_B.zip
bash cloud.sh verify
```

The merge requires exactly twelve unique completed collections. It performs analysis
only after all collections pass checks. No new training occurs during merge. A free
GPU matching the study hardware is required for selected-checkpoint replay before
either shard or full packaging; keep Server A free until merged replay finishes.

The final review file is `STE_HardUC_Confirm_v1_REVIEW.zip`. This compact archive
supports independent selected-checkpoint replay and review of development choices,
test predictions and metrics. Keep `STE_HardUC_Confirm_v1_RESULTS.zip` as the complete
evidence archive: exact full analysis reruns also require the unselected checkpoint
files retained there. If the review ZIP exceeds the upload-size limit, use its
generated `.part001`, `.part002`, … files and
`STE_HardUC_Confirm_v1_REVIEW_PARTS.json`, retaining their checksums. Do not stop or
release the server until archive verification and download checks succeed.
