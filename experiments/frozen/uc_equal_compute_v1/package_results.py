#!/usr/bin/env python3
"""Package verified measured outputs, with immutable byte snapshots and manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "frozen"))
from ste.common import unit_complete
from analyze import atomic_json, effective_config
from verify_archive import verify, verify_outroot, FULL_MARKER, SMOKE_MARKER


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def payload_files(code, results, review):
    items = []
    for p in sorted(code.rglob("*")):
        if not p.is_file() or any(part in (".venv", "__pycache__", ".git") for part in p.relative_to(code).parts) or p.suffix in (".pyc", ".tmp", ".zip"):
            continue
        if p.is_symlink():
            raise RuntimeError("Refusing a source symlink: " + str(p))
        items.append((p, "code/" + p.relative_to(code).as_posix()))
    for p in sorted(results.rglob("*")):
        if not p.is_file() or p.name in ("RUN.lock", "RUNNING.lock", "ARCHIVES_READY.json") or p.suffix in (".tmp", ".zip", ".sha256"):
            continue
        if any(part in ("__pycache__", ".venv") for part in p.relative_to(results).parts):
            continue
        if review and (p.suffix == ".pt" or p.name == "train.npz" and p.parent.name == "datasets"):
            continue
        if p.is_symlink():
            raise RuntimeError("Refusing a result symlink: " + str(p))
        items.append((p, "run/" + p.relative_to(results).as_posix()))
    return items


def create_archive(path, code, results, cfg, constraints, review):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    manifest = {"release": "STE_UC_EqualCompute_v1", "smoke": bool(cfg.get("smoke", False)),
                "review": bool(review), "classification": "SOFTWARE SMOKE; NOT SCIENTIFIC EVIDENCE" if cfg.get("smoke") else "complete fresh prospective follow-up; completion alone does not imply superiority",
                "primary_inference_valid": bool(constraints["primary_inference_valid"]),
                "budget_valid": bool(constraints["budget_valid"]),
                "log_policy": "Each log is captured as the exact bytes read during packaging; later verification/final shell messages are outside this immutable snapshot.",
                "review_omissions": ["checkpoint and optimizer weights (.pt)", "training datasets (train.npz)"] if review else [],
                "files": []}
    with temp.open("w+b") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for source, member in payload_files(code, results, review):
                # One read determines both the manifest hash and the archived bytes. This
                # prevents a concurrently appended launcher log from producing hash races.
                data = source.read_bytes()
                row = {"path": member, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                manifest["files"].append(row)
                z.writestr(member, data)
            z.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fsync_directory(path.parent)
    from verify_archive import sha
    archive_digest = sha(path)
    checksum = path.with_name(path.name + ".sha256")
    checksum_temp = checksum.with_name(checksum.name + ".tmp")
    with checksum_temp.open("w") as f:
        f.write(archive_digest + "  " + path.name + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(checksum_temp, checksum)
    fsync_directory(path.parent)
    result = verify(path)
    return {**result, "name": path.name, "checksum_name": checksum.name}


def package(results, outroot, config):
    results = Path(results).resolve()
    outroot = Path(outroot).resolve()
    cfg = effective_config(results, config)
    if not (results / "RUN_COMPLETE.json").is_file() or not unit_complete(results / "analysis"):
        raise RuntimeError("Run and verified analysis must complete before successful packaging")
    constraints = json.loads((results / "analysis" / "INTERPRETATION_CONSTRAINTS.json").read_text())
    smoke = bool(cfg.get("smoke", False))
    if smoke != bool(constraints["smoke"]):
        raise RuntimeError("Analysis classification differs from the effective run")
    if not smoke and (cfg["learning"]["collections"] != 12 or cfg["learning"]["initializations"] != 3):
        raise RuntimeError("Refusing a full archive for an incomplete prospective study")
    outroot.mkdir(parents=True, exist_ok=True)
    # A stale ready marker must not survive a failed replacement of one archive.
    (outroot / "ARCHIVES_READY.json").unlink(missing_ok=True)
    prefix = "STE_UC_EqualCompute_v1_SMOKE" if smoke else "STE_UC_EqualCompute_v1"
    full = create_archive(outroot / (prefix + "_RESULTS.zip"), CODE, results, cfg, constraints, False)
    review = create_archive(outroot / (prefix + "_REVIEW.zip"), CODE, results, cfg, constraints, True)
    status = SMOKE_MARKER if smoke else FULL_MARKER
    marker = {"status": status, "smoke": smoke, "full": full, "review": review,
              "budget_valid": bool(constraints["budget_valid"]),
              "primary_inference_valid": bool(constraints["primary_inference_valid"]),
              "integrity_complete": True,
              "scientific_notice": "Archive integrity and experiment completion do not imply a favorable result or valid equal-compute inference."}
    atomic_json(outroot / "ARCHIVES_READY.json", marker)
    fsync_directory(outroot)
    checked = verify_outroot(outroot)
    print(json.dumps(checked, indent=2, sort_keys=True), flush=True)
    print(status, flush=True)
    print("Share: " + str(outroot / (prefix + "_REVIEW.zip")), flush=True)
    print("Keep full raw checkpoint archive: " + str(outroot / (prefix + "_RESULTS.zip")), flush=True)
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--outroot", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    package(args.results, args.outroot, args.config)
