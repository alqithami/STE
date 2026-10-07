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

FULL_MARKER = "STE_HARDUC_CONFIRM_V1_ARCHIVES_READY"
SMOKE_MARKER = "STE_HARDUC_CONFIRM_V1_SMOKE_ARCHIVES_READY"
SHARD_MARKER = "STE_HARDUC_CONFIRM_V1_SHARD_ARCHIVES_READY"
MANIFEST = "ARCHIVE_MANIFEST.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not name.startswith(("/", "\\")) and "\\" not in name and all(p not in ("", ".", "..") for p in name.split("/")) and all(p not in ("", ".", "..") for p in path.parts)


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
        shard = bool(manifest.get("shard", False))
        required = {"run/CONFIG.json", "run/FROZEN_CONFIG.json", "run/RUN_LOCK.json", "run/PROTOCOL_LOCK.json",
                    "run/SHARD_COMPLETE.json" if shard else "run/RUN_COMPLETE.json",
                    "run/learning/COMPLETE.json", "run/SELECTED_CHECKPOINT_REPLAY.json", "code/analyze.py", "code/package_results.py", "code/verify_archive.py", "code/replay_selected.py"}
        if not shard:
            required.update({"run/analysis/COMPLETE.json", "run/analysis/INTERPRETATION_CONSTRAINTS.json",
                    "run/analysis/PRIMARY_COMPARISONS.csv", "run/analysis/PRIMARY_COLLECTION_DIFFERENCES.csv",
                    "run/analysis/CANDIDATE_BUDGET_AUDIT.csv", "run/analysis/SYSTEM_BUDGET_AUDIT.csv",
                    "run/analysis/PER_CASE_METRICS.csv"})
        if not required.issubset(payload):
            raise RuntimeError("Missing completion/reanalysis payload: " + repr(sorted(required - payload)))
        cfg = json.loads(z.read("run/CONFIG.json"))
        smoke = bool(cfg.get("smoke", False))
        if smoke != bool(manifest.get("smoke", False)):
            raise RuntimeError("Archive smoke classification disagrees with effective configuration")
        lc = cfg["learning"]
        if not smoke and (lc["collections"] != 12 or lc["initializations"] != 3):
            raise RuntimeError("Production archive is not based on the complete 12 by 3 protocol")
        lock = json.loads(z.read("run/RUN_LOCK.json"))
        ids = lock["collection_ids"]
        if ids != sorted(set(ids)) or any(not isinstance(c, int) or c < 0 or c >= lc["collections"] for c in ids):
            raise RuntimeError("Invalid shard collection identities")
        if manifest["collection_ids"] != ids or (not shard and ids != list(range(lc["collections"]))):
            raise RuntimeError("Full archive must contain every collection exactly once")
        replay = json.loads(z.read("run/SELECTED_CHECKPOINT_REPLAY.json"))
        if replay.get("status") != "STE_HARDUC_SELECTED_CHECKPOINT_REPLAY_VERIFIED" or replay.get("collections") != ids or replay.get("prediction_units") != len(ids) * lc["initializations"] * len(lc["objectives"]) * len(lc["test_sizes"]) or bool(replay.get("smoke")) != smoke:
            raise RuntimeError("Selected checkpoint replay is incomplete or inconsistent")
        replay_rows = {(row["collection"], row["initialization"], row["system"], row["n"]): row for row in replay["units"]}
        if len(replay_rows) != replay["prediction_units"] or len(replay_rows) != len(replay["units"]):
            raise RuntimeError("Selected replay contains duplicate or omitted units")
        if lock["config"] != cfg or lock["supplied_config"] != json.loads(z.read("run/FROZEN_CONFIG.json")):
            raise RuntimeError("Configuration bytes differ from the run lock")
        protocol = json.loads(z.read("run/PROTOCOL_LOCK.json"))
        if protocol != {key: lock[key] for key in ("config", "supplied_config", "source_sha256")}:
            raise RuntimeError("Prospective global protocol identity mismatch")
        for source, expected_hash in lock["source_sha256"].items():
            if not safe_name(source) or "code/" + source not in table or table["code/" + source]["sha256"] != expected_hash:
                raise RuntimeError("Source differs from the immutable prospective protocol: " + source)
        completion = json.loads(z.read("run/SHARD_COMPLETE.json" if shard else "run/RUN_COMPLETE.json"))
        if completion["learning_complete_sha256"] != table["run/learning/COMPLETE.json"]["sha256"] or completion["run_lock_sha256"] != table["run/RUN_LOCK.json"]["sha256"]:
            raise RuntimeError("Root completion marker does not identify archived verified run")
        review = bool(manifest.get("review", False))
        omissions = manifest.get("review_omissions", [])
        omitted = {row["path"]: row for row in omissions}
        if len(omitted) != len(omissions) or (not review and omitted) or any(name in payload for name in omitted):
            raise RuntimeError("Invalid or duplicate review omission inventory")
        selected = set()
        for c in ids:
            base = f"run/learning/collection_{c:02d}/"
            expected = {base + "COMPLETE.json", base + "per_case_metrics.csv", base + "datasets/development.npz"}
            expected.update(base + f"datasets/test_n{n}.npz" for n in lc["test_sizes"])
            if not review: expected.add(base + "datasets/train.npz")
            for init in range(lc["initializations"]):
                ibase = base + f"uc/init_{init:02d}/"
                expected.update({ibase + "CHOICES_BEFORE_TEST.json", ibase + "SELECTION_FROZEN.json", ibase + "SELECTION_COST.json", ibase + "ALL_DEVELOPMENT_CANDIDATES.json"})
                choices = json.loads(z.read(ibase + "CHOICES_BEFORE_TEST.json"))
                for choice in choices.values():
                    checkpoint = "run/" + choice["checkpoint"]
                    if not safe_name(checkpoint): raise RuntimeError("Unsafe selected checkpoint")
                    selected.add(checkpoint); expected.add(checkpoint)
                    for n in lc["test_sizes"]:
                        key = (c, init, choice["objective"], n)
                        prediction = ibase + f"predictions/{choice['objective']}_l{choice['weight']:g}_ck{choice['checkpoint_index']:02d}_n{n}.npz"
                        expected.add(prediction)
                        row = replay_rows.get(key)
                        if row is None or row["checkpoint"] != choice["checkpoint"] or not row["hard_uc_exact_match"] or row["cases"] != lc["test_graphs_per_size"]:
                            raise RuntimeError("Replay unit differs from selected held-out system")
                        if checkpoint not in table or prediction not in table or row["checkpoint_sha256"] != table[checkpoint]["sha256"] or row["prediction_sha256"] != table[prediction]["sha256"]:
                            raise RuntimeError("Replay state/prediction hash chain differs from archived bytes")
                for objective in lc["objectives"]:
                    for weight in ([0.] if objective == "pair-hard" else lc["core_weights"]):
                        candidate = ibase + f"{objective}_lambda_{weight:g}/"
                        expected.update(candidate + name for name in ("COMPLETE.json", "timing.json", "development.json", "initial.pt", "DEVICE_PROOF.json"))
                        expected.update(candidate + f"development_checkpoint_{index:02d}.npz" for index in range(1, len(cfg["budget"]["checkpoint_fractions"]) + 1))
            if not expected.issubset(payload):
                raise RuntimeError("Missing complete collection/candidate/replay payload: " + repr(sorted(expected - payload)))
        for member in omitted:
            allowed = member.endswith("/datasets/train.npz") or member.endswith("/history.json") or (member.endswith(".pt") and "/checkpoint_" in member and member not in selected)
            if not safe_name(member) or not allowed or not member.startswith("run/"):
                raise RuntimeError("Review omits a required replay payload: " + member)
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
                    allowed_omission = review and member in omitted and omitted[member]["sha256"] == expected_digest
                    if not allowed_omission:
                        raise RuntimeError("Completion record points to missing archived payload: " + member)
                    omitted_references.append(member)
                elif table[member]["sha256"] != expected_digest:
                    raise RuntimeError("Completion hash disagrees with actual archived payload: " + member)
        if not shard:
            constraints = json.loads(z.read("run/analysis/INTERPRETATION_CONSTRAINTS.json"))
            if bool(constraints["smoke"]) != smoke or (smoke and constraints.get("primary_inference_valid")):
                raise RuntimeError("Invalid smoke inference classification")
            if bool(manifest["primary_inference_valid"]) != bool(constraints["primary_inference_valid"]) or bool(manifest["budget_valid"]) != bool(constraints["budget_valid"]):
                raise RuntimeError("Manifest and analysis validity classification disagree")
            primary = list(csv.DictReader(io.StringIO(z.read("run/analysis/PRIMARY_COMPARISONS.csv").decode())))
            if len(primary) != 2 or {r["right"] for r in primary} != {"aux-direct", "relational-direct"}:
                raise RuntimeError("Unexpected prospective primary comparison family")
            for row in primary:
                if int(row["units"]) != lc["collections"] or row["left"] != "ste-hard" or row["left_readout"] != "hard_uc" or row["left_rule"] != "hard" or row["right_readout"] != "direct" or row["right_rule"] != "absolute":
                    raise RuntimeError("Primary endpoint omits a collection or changes the prescribed deployed comparison")
                if not constraints["primary_inference_valid"] and any(row.get(key, "") for key in ("p", "p_holm", "ci_low", "ci_high")):
                    raise RuntimeError("Smoke or invalid-budget archive contains formal primary inference")
        else:
            constraints = {"primary_inference_valid": False, "budget_valid": None}
            if manifest["primary_inference_valid"] or "run/analysis/PRIMARY_COMPARISONS.csv" in payload:
                raise RuntimeError("Shard must never carry global primary inference")
        result = {"path": str(path), "bytes": path.stat().st_size, "sha256": archive_digest,
                  "payload_files_verified": len(rows), "crc_valid": True,
                  "review": review, "smoke": smoke, "shard": shard, "collection_ids": ids,
                  "collections": len(ids), "initializations": lc["initializations"],
                  "primary_inference_valid": bool(constraints["primary_inference_valid"]),
                  "budget_valid": constraints["budget_valid"],
                  "review_omitted_references": len(set(omitted_references))}
    return result


def verify_parts(manifest_path, assemble=None):
    """Verify every transport chunk and the ordered concatenation, optionally rebuild."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["parts"]
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError("Empty or duplicate transport parts")
    if not safe_name(manifest["archive_name"]) or "/" in manifest["archive_name"]:
        raise RuntimeError("Unsafe original archive name")
    target = Path(assemble).resolve() if assemble is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".assembling")
        handle = temp.open("wb")
    else:
        temp = handle = None
    combined = hashlib.sha256(); total = 0
    try:
        for index, row in enumerate(rows, 1):
            name = row["name"]
            if name != manifest["archive_name"] + f".part{index:03d}" or not safe_name(name) or "/" in name:
                raise RuntimeError("Unsafe or out-of-order transport part")
            path = manifest_path.parent / name
            part_hash = hashlib.sha256(); part_bytes = 0
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    part_hash.update(block); combined.update(block)
                    part_bytes += len(block); total += len(block)
                    if handle is not None: handle.write(block)
            if part_bytes != row["bytes"] or part_hash.hexdigest() != row["sha256"] or part_bytes > manifest["maximum_part_bytes"]:
                raise RuntimeError("Transport part byte/hash mismatch: " + name)
        if total != manifest["archive_bytes"] or combined.hexdigest() != manifest["archive_sha256"]:
            raise RuntimeError("Ordered transport concatenation differs from original archive")
        if handle is not None:
            handle.flush()
            import os
            os.fsync(handle.fileno()); handle.close(); handle = None
            if target.exists() and sha(target) != combined.hexdigest():
                raise RuntimeError("Refusing to replace a different existing assembled archive")
            os.replace(temp, target)
            checksum = target.with_name(target.name + ".sha256")
            checksum.write_text(combined.hexdigest() + "  " + target.name + "\n")
            verify(target)
    finally:
        if handle is not None: handle.close()
        if temp is not None: temp.unlink(missing_ok=True)
    return {"transport_manifest": str(manifest_path), "parts_verified": len(rows),
            "archive_bytes": total, "archive_sha256": combined.hexdigest(), "assembled_path": str(target) if target else None}


def verify_outroot(outroot, shard=False):
    outroot = Path(outroot).resolve()
    marker_path = outroot / ("SHARD_ARCHIVES_READY.json" if shard else "ARCHIVES_READY.json")
    if not marker_path.is_file():
        raise RuntimeError("ARCHIVES_READY.json absent: packaging has not completed")
    marker = json.loads(marker_path.read_text())
    if shard:
        if marker.get("status") != SHARD_MARKER:
            raise RuntimeError("Unknown shard completion marker")
        checked = verify(outroot / marker["archive"]["name"])
        if not checked["shard"] or checked["sha256"] != marker["archive"]["sha256"] or checked["collection_ids"] != marker["collection_ids"]:
            raise RuntimeError("Shard completion marker differs from archive")
        if "transport" in marker: verify_parts(outroot / marker["transport"]["manifest_name"])
        return {"status": SHARD_MARKER, "verified": [checked], "integrity_complete": True, "primary_inference_valid": False}
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
    if "review_transport" in marker: verify_parts(outroot / marker["review_transport"]["manifest_name"])
    return {"status": marker["status"], "verified": checked,
            "integrity_complete": True, "scientific_budget_valid": marker["budget_valid"],
            "primary_inference_valid": marker["primary_inference_valid"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    parser.add_argument("--outroot", type=Path)
    parser.add_argument("--shard", action="store_true")
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--assemble", type=Path)
    args = parser.parse_args()
    if args.parts is not None:
        if args.outroot or args.archives or args.shard:
            parser.error("Use --parts independently of archive/outroot/shard verification")
        print(json.dumps(verify_parts(args.parts, args.assemble), indent=2, sort_keys=True), flush=True)
        raise SystemExit(0)
    if args.assemble is not None:
        parser.error("--assemble requires --parts")
    if args.outroot is None and not args.archives:
        parser.error("Supply --outroot ROOT or one or more ZIP archives")
    if args.outroot is not None and args.archives:
        parser.error("Use --outroot or explicit archives, not both")
    output = verify_outroot(args.outroot, args.shard) if args.outroot is not None else [verify(p) for p in args.archives]
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
