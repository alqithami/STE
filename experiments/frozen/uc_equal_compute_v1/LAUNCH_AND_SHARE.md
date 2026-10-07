# Portable launch, monitoring and result preservation

These commands reproduce the already completed UC equal-compute study on a new
Linux GPU host. Connect using your own host address, username and matching private
SSH key. No server address or private-key path is required by the experiment.
The original full result archive retains the unmodified pre-execution package.

## Prepare and launch

Copy the extracted manuscript source to your chosen host, then enter its
code/uc_equal_compute_v1 directory. Verify the supplied source copy:

```bash
sha256sum -c SHA256SUMS
sha256sum -c frozen/FROZEN_SHA256SUMS.txt
export STE_RUN_ROOT="$HOME/ste-uc-equalcompute-reproduction"
bash ibm.sh start
```

Use a fresh persistent STE_RUN_ROOT for a new reproduction. Choose your host's
persistent storage location and override STE_RUN_ROOT explicitly when needed.
The launcher requires Python 3.10--3.12, verifies or creates a pinned environment,
checks CUDA placement and an idle selected GPU, performs separate software QA and
an isolated short smoke, then launches production as a detached process. It
never stops unrelated jobs. The allocation is 8 h 48 min of charged work; setup,
testing, analysis and packaging take additional time. Hardware changes can alter
optimizer update counts under a time-budget protocol.

LAUNCH_CONFIRMED means the job started, not that it completed. Keep the GPU host
and its persistent storage available until verified archives are preserved.

## Continuously watch new output

In the same shell with STE_RUN_ROOT exported:

```bash
tail -n 80 -F "$STE_RUN_ROOT/results/run.log"
```

Ctrl+C stops the viewer while detached production continues. For status:

```bash
bash ibm.sh status
```

## Verify completion

After the log reports ALL_ARCHIVES_VERIFIED:

```bash
bash ibm.sh verify
```

This checks both full and review archives, sidecar SHA-256 values, internal
manifests and completion records. RUN_COMPLETE.json alone is not archive
completion. Budget validity is reported separately from successful packaging.
Retain every outcome, including a failed budget audit or an unfavorable result.
If training finished but analysis/packaging was interrupted, `bash ibm.sh package`
rebuilds archives under the unchanged run lock. Do not edit source or configuration
within an existing run.

## Preserve results

Copy these four files from STE_RUN_ROOT to a local or persistent archive:

- STE_UC_EqualCompute_v1_RESULTS.zip and its .sha256 sidecar;
- STE_UC_EqualCompute_v1_REVIEW.zip and its .sha256 sidecar.

Use your host's file-transfer tool or scp with your own login details. On Linux,
verify the downloaded sidecars with `sha256sum -c`; on macOS use `shasum -a 256 -c`.
Do not remove server results until both downloaded archives verify. The full
archive retains model states and training datasets. The review archive omits
weights and training datasets; it cannot support weight-to-prediction replay.
