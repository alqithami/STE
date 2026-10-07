#!/usr/bin/env python3
"""Standard-library verification of STE complete-run and review result archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

FULL_MARKER = "STE_UC_EQUALCOMPUTE_V1_ARCHIVES_READY"
SMOKE_MARKER = "STE_UC_EQUALCOMPUTE_V1_SMOKE_ARCHIVES_READY"
MANIFEST = "ARCHIVE_MANIFEST.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and all(p not in ("", ".", "..") for p in path.parts)


def verify(path, sidecar=True):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError("Archive missing: " + str(path))
    archive_digest = sha(path)
    if sidecar:
        checksum_path = path.with_name(path.name + ".sha256")
        if not checksum_path.is_file():
            raise RuntimeError("Archive checksum sidecar missing: " + str(checksum_path))
        fields = checksum_path.read_text().strip().split(maxsplit=1)
        if len(fields) != 2 or fields[0] != archive_digest or fields[1].lstrip(" *") != path.name:
            raise RuntimeError("Archive checksum mismatch: " + str(path))
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(not safe_name(name) for name in names):
            raise RuntimeError("Duplicate or unsafe archive member")
        failed = z.testzip()
        if failed is not None:
            raise RuntimeError("Archive CRC failure: " + failed)
        if MANIFEST not in names:
            raise RuntimeError("Archive lacks its payload manifest")
        manifest = json.loads(z.read(MANIFEST))
        rows = manifest.get("files", [])
        payload = set(names) - {MANIFEST}
        listed = [row["path"] for row in rows]
        if len(listed) != len(set(listed)) or set(listed) != payload:
            raise RuntimeError("Manifest does not list every payload exactly once")
        table = {}
        for row in rows:
            data = z.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError("Manifest byte/hash failure: " + row["path"])
            table[row["path"]] = row
        required = {"run/CONFIG.json", "run/FROZEN_CONFIG.json", "run/RUN_COMPLETE.json",
                    "run/analysis/COMPLETE.json", "run/analysis/INTERPRETATION_CONSTRAINTS.json",
                    "run/analysis/PRIMARY_COMPARISONS.csv", "run/analysis/PRIMARY_COLLECTION_DIFFERENCES.csv",
                    "run/analysis/CANDIDATE_BUDGET_AUDIT.csv", "run/analysis/SYSTEM_BUDGET_AUDIT.csv",
                    "run/analysis/PER_CASE_METRICS.csv", "code/analyze.py", "code/package_results.py", "code/verify_archive.py"}
        if not required.issubset(payload):
            raise RuntimeError("Missing completion/reanalysis payload: " + repr(sorted(required - payload)))
        cfg = json.loads(z.read("run/CONFIG.json"))
        smoke = bool(cfg.get("smoke", False))
        if smoke != bool(manifest.get("smoke", False)):
            raise RuntimeError("Archive smoke classification disagrees with effective configuration")
        lc = cfg["learning"]
        if not smoke and (lc["collections"] != 12 or lc["initializations"] != 3):
            raise RuntimeError("Full-run archive is not the complete 12 by 3 study")
        review = bool(manifest.get("review", False))
        for c in range(lc["collections"]):
            base = f"run/learning/collection_{c:02d}/"
            expected = {base + "COMPLETE.json", base + "per_case_metrics.csv", base + "datasets/development.npz"}
            expected.update(base + f"datasets/test_n{n}.npz" for n in lc["test_sizes"])
            if not review:
                expected.add(base + "datasets/train.npz")
            for init in range(lc["initializations"]):
                ibase = base + f"uc/init_{init:02d}/"
                expected.add(ibase + "CHOICES_BEFORE_TEST.json")
                for objective in lc["objectives"]:
                    for weight in ([0.] if objective == "pair" else lc["core_weights"]):
                        candidate = ibase + f"{objective}_lambda_{weight:g}/"
                        expected.update(candidate + name for name in ("COMPLETE.json", "timing.json", "development.json"))
            if not expected.issubset(payload):
                raise RuntimeError("Missing complete collection/candidate payload: " + repr(sorted(expected - payload)))
        # Original completion records are retained verbatim. The review archive deliberately
        # omits checkpoint weights and training data, but retains every recorded hash.
        omitted_references = []
        for name in names:
            if not name.startswith("run/") or not name.endswith("/COMPLETE.json"):
                continue
            marker = json.loads(z.read(name))
            references = marker.get("files")
            if not isinstance(references, dict):
                raise RuntimeError("Malformed unit completion record: " + name)
            parent = str(PurePosixPath(name).parent)
            for relative, expected_digest in references.items():
                if not safe_name(relative):
                    raise RuntimeError("Unsafe completion reference: " + name)
                member = parent + "/" + relative
                if member not in table:
                    allowed_omission = review and (member.endswith(".pt") or member.endswith("/datasets/train.npz"))
                    if not allowed_omission:
                        raise RuntimeError("Completion record points to missing archived payload: " + member)
                    omitted_references.append(member)
                elif table[member]["sha256"] != expected_digest:
                    raise RuntimeError("Completion hash disagrees with actual archived payload: " + member)
        constraints = json.loads(z.read("run/analysis/INTERPRETATION_CONSTRAINTS.json"))
        if bool(constraints["smoke"]) != smoke or (smoke and constraints.get("primary_inference_valid")):
            raise RuntimeError("Invalid smoke inference classification")
        if bool(manifest["primary_inference_valid"]) != bool(constraints["primary_inference_valid"]) or bool(manifest["budget_valid"]) != bool(constraints["budget_valid"]):
            raise RuntimeError("Manifest and analysis validity classification disagree")
        primary = list(csv.DictReader(io.StringIO(z.read("run/analysis/PRIMARY_COMPARISONS.csv").decode())))
        if len(primary) != 2 or {r["right"] for r in primary} != {"aux", "relational_aux"}:
            raise RuntimeError("Unexpected prospective primary comparison family")
        for row in primary:
            if int(row["units"]) != lc["collections"]:
                raise RuntimeError("Primary endpoint silently omits a collection")
            if not constraints["primary_inference_valid"] and any(row.get(key, "") for key in ("p", "p_holm", "ci_low", "ci_high")):
                raise RuntimeError("Smoke or invalid-budget archive contains formal primary inference")
        result = {"path": str(path), "bytes": path.stat().st_size, "sha256": archive_digest,
                  "payload_files_verified": len(rows), "crc_valid": True,
                  "review": review, "smoke": smoke, "collections": lc["collections"],
                  "initializations": lc["initializations"], "primary_inference_valid": bool(constraints["primary_inference_valid"]),
                  "budget_valid": bool(constraints["budget_valid"]),
                  "review_omitted_checkpoint_or_train_references": len(set(omitted_references))}
    return result


def verify_outroot(outroot):
    outroot = Path(outroot).resolve()
    marker_path = outroot / "ARCHIVES_READY.json"
    if not marker_path.is_file():
        raise RuntimeError("ARCHIVES_READY.json absent: packaging has not completed")
    marker = json.loads(marker_path.read_text())
    if marker.get("status") not in (FULL_MARKER, SMOKE_MARKER):
        raise RuntimeError("Unknown archive completion marker")
    checked = []
    for role in ("full", "review"):
        row = marker[role]
        path = outroot / row["name"]
        if not path.resolve().is_relative_to(outroot):
            raise RuntimeError("Completion marker archive path escapes output root")
        actual = verify(path)
        if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
            raise RuntimeError("Completion marker does not describe the verified archive")
        if actual["review"] != (role == "review"):
            raise RuntimeError("Wrong archive role")
        if actual["smoke"] != (marker["status"] == SMOKE_MARKER):
            raise RuntimeError("Marker smoke classification mismatch")
        if actual["primary_inference_valid"] != bool(marker["primary_inference_valid"]) or actual["budget_valid"] != bool(marker["budget_valid"]):
            raise RuntimeError("Completion marker and archived inference/budget validity disagree")
        checked.append(actual)
    return {"status": marker["status"], "verified": checked,
            "integrity_complete": True, "scientific_budget_valid": marker["budget_valid"],
            "primary_inference_valid": marker["primary_inference_valid"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    parser.add_argument("--outroot", type=Path)
    args = parser.parse_args()
    if args.outroot is None and not args.archives:
        parser.error("Supply --outroot ROOT or one or more ZIP archives")
    if args.outroot is not None and args.archives:
        parser.error("Use --outroot or explicit archives, not both")
    output = verify_outroot(args.outroot) if args.outroot is not None else [verify(p) for p in args.archives]
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
