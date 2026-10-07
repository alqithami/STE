#!/usr/bin/env python3
"""Check the selected CUDA device and storage before this experiment starts."""
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
    p = subprocess.run(["nvidia-smi", *arguments], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError("nvidia-smi failed: " + p.stderr.strip())
    return list(csv.reader(io.StringIO(p.stdout)))


def check_software():
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
        raise RuntimeError("Use Python 3.10, 3.11, or 3.12; setup prefers 3.11.")
    import numpy
    import scipy
    import pandas
    import torch
    versions = {"numpy": numpy.__version__, "scipy": scipy.__version__,
                "pandas": pandas.__version__, "torch": torch.__version__}
    wanted = {"numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.3"}
    for name, version in wanted.items():
        if versions[name] != version:
            raise RuntimeError(f"{name} must be {version}; found {versions[name]}")
    if str(torch.__version__) != "2.6.0+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("Use torch 2.6.0 with CUDA 12.4, matching the completed run.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this Python environment.")
    return torch, versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--software-only", action="store_true")
    args = ap.parse_args()
    torch, versions = check_software()
    report = {"python": sys.version, "python_path": sys.executable,
              "versions": versions, "cuda": torch.version.cuda,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
              "device_count": torch.cuda.device_count(),
              "device_0": torch.cuda.get_device_name(0)}
    if not args.software_only:
        root = Path(args.root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
        st = os.statvfs(root)
        report.update(storage_path=str(root), free_bytes=usage.free,
                      free_inodes=st.f_favail)
        if usage.free < 5 * 1024 ** 3:
            raise RuntimeError("Less than 5 GiB filesystem space is free at the run root.")
        if st.f_files > 0 and st.f_favail < 10000:
            raise RuntimeError("Fewer than 10,000 free inodes at the run root.")
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("nvidia-smi is required to check GPU availability.")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        selected_uuids = set()
        for identifier in visible:
            identifier = identifier.strip()
            if not identifier:
                raise RuntimeError("CUDA_VISIBLE_DEVICES has an empty device identifier.")
            for row in query(["-i", identifier, "--query-gpu=uuid", "--format=csv,noheader"]):
                if row:
                    selected_uuids.add(row[0].strip())
        processes = []
        for row in query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                          "--format=csv,noheader,nounits"]):
            if len(row) < 4:
                continue
            uuid, pid, name, memory = [s.strip() for s in row[:4]]
            if uuid in selected_uuids and pid != str(os.getpid()):
                processes.append({"gpu_uuid": uuid, "pid": pid,
                                  "process": name, "memory_MiB": memory})
        report["selected_gpu_uuids"] = sorted(selected_uuids)
        report["other_visible_compute_processes"] = processes
        if processes:
            print(json.dumps(report, indent=2), flush=True)
            raise RuntimeError("The selected GPU has other compute processes. No jobs were stopped. "
                               "Use a free GPU or RunPod before starting the timed experiment.")
        # Execute and synchronize one real CUDA operation; availability alone is insufficient.
        x = torch.ones((16, 16), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        if not bool(torch.all(y == 16)):
            raise RuntimeError("CUDA arithmetic check failed.")
        report["cuda_arithmetic_check"] = "passed"
        report["storage_note"] = "Filesystem free space is not a RunPod account or volume quota check."
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PREFLIGHT_FAILED: " + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
