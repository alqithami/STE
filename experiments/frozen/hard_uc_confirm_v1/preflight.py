#!/usr/bin/env python3
"""Pinned software and idle-device checks. Query compute jobs before CUDA allocation."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def query(arguments):
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is required. Check the GPU and its driver before launch.")
    result = subprocess.run(["nvidia-smi", *arguments], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    return list(csv.reader(io.StringIO(result.stdout)))


def idle_gpu(expected_model=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if len(visible) != 1 or not visible[0].strip():
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES (default 0).")
    identifier = visible[0].strip()
    rows = query(["-i", identifier, "--query-gpu=uuid,name,memory.total,driver_version",
                  "--format=csv,noheader,nounits"])
    if len(rows) != 1 or len(rows[0]) < 4:
        raise RuntimeError("Could not identify the single selected GPU.")
    uuid, name, memory, driver = [value.strip() for value in rows[0][:4]]
    if expected_model and name != expected_model:
        raise RuntimeError(f"Expected {expected_model}, found {name}. Use the same GPU model for all shards.")
    processes = []
    for row in query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                      "--format=csv,noheader,nounits"]):
        if len(row) < 4:
            continue
        app_uuid, pid, process, used = [value.strip() for value in row[:4]]
        if app_uuid == uuid and pid != str(os.getpid()):
            processes.append({"gpu_uuid": app_uuid, "pid": pid,
                              "process": process, "memory_MiB": used})
    report = {"selected_gpu_uuid": uuid, "gpu_name": name,
              "gpu_memory_MiB": memory, "driver_version": driver,
              "CUDA_VISIBLE_DEVICES": identifier,
              "other_visible_compute_processes": processes,
              "busy_check_before_torch_cuda": True}
    if processes:
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError("The selected GPU has other compute processes. No jobs were stopped. "
                           "Use a free GPU before starting this timed experiment.")
    return report


def software():
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        raise RuntimeError("Use Python 3.10, 3.11, or 3.12. System Python 3.14 is unsupported.")
    import numpy
    import scipy
    import pandas
    import torch
    versions = {"numpy": numpy.__version__, "scipy": scipy.__version__,
                "pandas": pandas.__version__, "torch": str(torch.__version__)}
    for name, wanted in {"numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.3"}.items():
        if versions[name] != wanted:
            raise RuntimeError(f"{name} must be {wanted}; found {versions[name]}")
    if versions["torch"] != "2.6.0+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("Use torch 2.6.0+cu124 with its CUDA 12.4 runtime.")
    return torch, {"python": sys.version, "python_path": sys.executable,
                   "versions": versions, "cuda_runtime": torch.version.cuda}


def collections(value):
    try:
        ids = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise RuntimeError("STE_COLLECTIONS must be comma-separated integers in 0..11.")
    if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= 12 for i in ids):
        raise RuntimeError("STE_COLLECTIONS must contain distinct collection IDs in 0..11.")
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--software-only", action="store_true")
    parser.add_argument("--busy-only", action="store_true")
    parser.add_argument("--expect-gpu")
    parser.add_argument("--report")
    parser.add_argument("--collections", default="0,1,2,3,4,5,6,7,8,9,10,11")
    args = parser.parse_args()
    if args.software_only and args.busy_only:
        raise RuntimeError("Choose only one limited preflight mode.")
    report = {"collection_ids": collections(args.collections)}
    # Busy-only requires no scientific packages and never imports torch.
    if not args.software_only:
        report.update(idle_gpu(args.expect_gpu))
    if not args.busy_only:
        torch, sw = software()
        report.update(sw)
        if not args.software_only:
            root = Path(args.root).resolve()
            root.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(root)
            st = os.statvfs(root)
            report.update(storage_path=str(root), free_bytes=usage.free, free_inodes=st.f_favail)
            if usage.free < 8 * 1024 ** 3:
                raise RuntimeError("Less than 8 GiB free at the run root; leave space for checkpoints and archives.")
            if st.f_files > 0 and st.f_favail < 10000:
                raise RuntimeError("Fewer than 10,000 free inodes at the run root.")
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Production needs one available CUDA GPU; CPU production is refused.")
            report["torch_device_name"] = torch.cuda.get_device_name(0)
            x = torch.ones((16, 16), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            if not bool(torch.all(y == 16)):
                raise RuntimeError("CUDA arithmetic check failed.")
            report["cuda_arithmetic_check"] = "passed"
            report["storage_note"] = "Filesystem free space is not an account or volume quota check."
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PREFLIGHT_FAILED: " + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
