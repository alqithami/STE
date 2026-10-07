#!/usr/bin/env python3
"""Prospective hard-UC confirmation with synchronized, explicitly audited budgets.

Production requires the pinned CUDA environment. --smoke is software QA only;
--pilot is a separate timing trial and cannot produce inferential results.
The original data generator, model, training loss, UC operator, and metrics are imported
unchanged from frozen/ste. No production experiment has been run by this file's
authors while preparing this release.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Must precede torch import; deterministic CUDA GEMM requires this setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frozen"))
import numpy as np
import pandas as pd
import scipy
import torch
from torch.nn import functional as F

from ste.common import (atomic_json, atomic_npz, configure_torch, emit,
                        environment, finish_unit, seed_for, sha, unit_complete,
                        write_rows)
from ste.data import make_split
from ste.metrics import decide, select_threshold, set_metrics, thresholds
from ste.models import PairModel, pair_loss
from ste.operators import hard_core, soft_core


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PhaseClock:
    """Wall time includes Python work; CUDA event time is separately reported."""
    def __init__(self, device):
        self.device = device
        self.event_start = self.event_end = None

    def start(self):
        synchronize(self.device)
        if self.device.type == "cuda":
            self.event_start = torch.cuda.Event(enable_timing=True)
            self.event_end = torch.cuda.Event(enable_timing=True)
            self.event_start.record()
        self.wall_start = time.perf_counter()
        return self

    def stop(self):
        if self.event_end is not None:
            self.event_end.record()
        synchronize(self.device)
        wall = time.perf_counter() - self.wall_start
        cuda = self.event_start.elapsed_time(self.event_end) / 1000 if self.event_start else None
        return wall, cuda


def atomic_torch_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        torch.save(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def model_hash(model, shared_only=False):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if shared_only and not (name.startswith("encoder.") or name.startswith("pair.")):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_fingerprint():
    files = list(HERE.glob("*.py"))
    files.extend(HERE / name for name in ("config.json", "cloud.sh", "PROTOCOL.md",
                 "requirements.txt", "README.md", "LAUNCH_AND_SHARE.md"))
    files.extend(sorted((HERE / "frozen").rglob("*.py")))
    files.append(HERE / "frozen" / "FROZEN_SHA256SUMS.txt")
    return {str(path.relative_to(HERE)): sha(path) for path in sorted(set(files))}


SYSTEMS = {
    "pair-hard": ("pair", "hard_uc"), "aux-direct": ("aux", "direct"),
    "relational-direct": ("relational_aux", "direct"), "ste-hard": ("ste", "hard_uc"),
    "aux-hard": ("aux", "hard_uc"), "relational-hard": ("relational_aux", "hard_uc")}


def training_objective(cfg, system):
    return cfg["learning"]["systems"][system]["training_objective"]


def decision_rule(readout):
    return "hard" if readout == "hard_uc" else "absolute"


def candidate_weights(cfg, system):
    return [0.0] if training_objective(cfg, system) == "pair" else cfg["learning"]["core_weights"]


def parse_collections(value, cfg):
    if value is None or value == "all":
        return list(range(cfg["learning"]["collections"]))
    try:
        ids = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise ValueError("Collections must be comma-separated integer identifiers.") from error
    if not ids or len(ids) != len(set(ids)) or any(x < 0 or x >= cfg["learning"]["collections"] for x in ids):
        raise ValueError("Collection identifiers must be unique and inside the frozen study.")
    return sorted(ids)


def validate_config(cfg):
    lc = cfg["learning"]
    expected_systems = {s: {"training_objective": o, "readout": r} for s, (o, r) in SYSTEMS.items()}
    if cfg["seed"] != 9872513 or cfg["release"] != "STE-HardUC-Confirm-v1":
        raise ValueError("Fresh seed and release identifier are prospectively fixed.")
    if lc["targets"] != ["uc"] or lc["objectives"] != list(SYSTEMS) or lc["systems"] != expected_systems:
        raise ValueError("The six independently timed systems are prospectively fixed.")
    if lc["readouts"] != {s: [r] for s, (_, r) in SYSTEMS.items()}:
        raise ValueError("Readouts cannot be changed after protocol freeze.")
    if cfg["decisions"]["rules"] != ["hard", "absolute"]:
        raise ValueError("Hard systems have no membership cutoff; native heads use development calibration.")
    if cfg["budget"]["checkpoint_fractions"] != [0.025, 0.1, 0.25, 0.5, 1.0] or cfg["budget"]["relative_tolerance"] != 0.02:
        raise ValueError("Five checkpoint fractions and fixed 2% budget tolerance are required.")
    if lc["core_weights"] != [0.1, 1.0] or lc["temperature"] != 0.035:
        raise ValueError("Frozen training temperatures and lambda candidates must be preserved.")
    if cfg["budget"]["pair_candidate_multiplier"] != 2.0:
        raise ValueError("Pair-only receives one 220-second fit; supervised systems receive two 110-second fits.")
    if not cfg.get("smoke") and not cfg.get("pilot"):
        required = {"collections": 12, "initializations": 3, "train_n": 12, "train_graphs": 512,
                    "development_graphs": 128, "test_sizes": [24, 48], "test_graphs_per_size": 256,
                    "batch_size": 16, "eval_batch_size": 8,
                    "families": ["ordered", "planted", "separated", "rank_mixture", "random"]}
        for key, value in required.items():
            if lc[key] != value:
                raise ValueError(f"Production config {key} differs from the frozen design.")
        if cfg["budget"]["seconds_per_candidate"] != 110.0:
            raise ValueError("Production candidate budget must be 110 seconds.")


def choose_device(cfg, requested):
    if requested == "cpu" and not cfg.get("smoke"):
        raise RuntimeError("CPU is allowed only with --smoke. Production and pilot require CUDA.")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: refusing silent CPU fallback.")
        device = torch.device(requested)
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    actual = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
              "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}
    if not cfg.get("smoke"):
        mismatches = {k: {"expected": v, "actual": actual[k]}
                      for k, v in cfg["required_versions"].items() if actual[k] != v}
        if mismatches:
            raise RuntimeError("Pinned production environment mismatch: " + json.dumps(mismatches))
    return device


def configure_mode(cfg, smoke, pilot):
    cfg = copy.deepcopy(cfg)
    cfg["smoke"], cfg["pilot"] = bool(smoke), bool(pilot)
    if smoke or pilot:
        lc = cfg["learning"]
        lc.update(collections=1, initializations=1, train_graphs=32,
                  development_graphs=16, test_graphs_per_size=8)
        cfg["budget"]["seconds_per_candidate"] = 0.25 if smoke else 3.0
        cfg["inference"]["disabled_reason"] = "software smoke QA" if smoke else "timing pilot"
    return cfg


def lock_run(out, cfg, device, supplied_cfg, collection_ids):
    out.mkdir(parents=True, exist_ok=True)
    configure_torch(seed_for(cfg["seed"], "global-configuration"), device)
    runtime = environment(device, cfg.get("smoke", False))
    runtime["pilot"] = cfg.get("pilot", False)
    props = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    runtime_lock = {"torch": runtime["torch"], "numpy": runtime["numpy"],
                    "scipy": runtime["scipy"], "pandas": runtime["pandas"],
                    "cuda_runtime": runtime["cuda_runtime"], "device": str(device),
                    "gpu_name": props.name if props else None,
                    "gpu_capability": [props.major, props.minor] if props else None}
    protocol = {"config": cfg, "supplied_config": supplied_cfg, "source_sha256": source_fingerprint()}
    lock = {**protocol, "runtime": runtime_lock, "collection_ids": collection_ids}
    protocol_path = out / "PROTOCOL_LOCK.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Global protocol lock differs from the delivered immutable protocol.")
    path = out / "RUN_LOCK.json"
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise RuntimeError("Resume rejected: config, runner/frozen source, or device environment changed. Use a new run directory.")
    else:
        # The shell starts nohup with run.log already open and may create a
        # progress snapshot before this lock. Those launch-only files are safe;
        # any study artifacts without an identifying lock remain forbidden.
        unknown = [p.name for p in out.iterdir() if p.name not in {"run.log", "PROGRESS.json"}]
        if unknown:
            raise RuntimeError("Unrecognized nonempty output directory; refusing to mix runs.")
        atomic_json(path, lock)
        atomic_json(protocol_path, protocol)
        atomic_json(out / "CONFIG.json", cfg)
        atomic_json(out / "FROZEN_CONFIG.json", supplied_cfg)
    atomic_json(out / "ENVIRONMENT.json", runtime)
    return runtime


def verify_or_create_dataset(path, cfg, collection, phase, n, count):
    hash_path = path.with_suffix(path.suffix + ".sha256.json")
    if path.exists():
        if not hash_path.exists() or json.loads(hash_path.read_text())["sha256"] != sha(path):
            raise RuntimeError("Existing dataset integrity check failed: " + str(path))
    elif hash_path.exists():
        raise RuntimeError("Dataset hash exists but dataset is missing: " + str(path))
    data = make_split(path, cfg, collection, phase, n, count)
    if not hash_path.exists():
        atomic_json(hash_path, {"sha256": sha(path), "collection": collection,
                               "phase": phase, "n": n, "count": count})
    return data


@torch.no_grad()
def predict(model, data, cfg, device, requested_readouts=None):
    model.eval()
    structural_needed = requested_readouts is None or "structural" in requested_readouts
    hard_needed = requested_readouts is None or "hard_uc" in requested_readouts
    output = {"P": [], "direct": []}
    if structural_needed:
        output["structural"] = []
    for start in range(0, len(data["W"]), cfg["learning"]["eval_batch_size"]):
        stop = start + cfg["learning"]["eval_batch_size"]
        W = torch.as_tensor(data["W"][start:stop], dtype=torch.float32, device=device)
        T = torch.as_tensor(data["T"][start:stop], dtype=torch.float32, device=device)
        P, direct = model(W, T)
        values = [("P", P), ("direct", direct)]
        if structural_needed:
            values.append(("structural", soft_core(P, "uc", cfg["learning"]["temperature"])))
        for name, value in values:
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite evaluation output: " + name)
            output[name].append(value.cpu().numpy())
    result = {name: np.concatenate(values) for name, values in output.items()}
    if hard_needed:
        result["hard_uc"] = np.array([hard_core(matrix > .5, "uc") for matrix in result["P"]])
    return result


def select_development_readout(prediction, labels, cfg, readout):
    """Select native calibration, or evaluate a cutoff-free binary hard UC set."""
    if readout == "hard_uc":
        decoded = prediction["hard_uc"]
        if decoded.dtype != np.bool_:
            raise RuntimeError("Hard UC development readout must contain Boolean sets.")
        value = float(np.mean([set_metrics(y, hard)["f1"] for y, hard in zip(labels, decoded)]))
        return .5, value  # Edge threshold metadata, never a membership cutoff.
    return select_threshold(prediction[readout], labels, thresholds(cfg), "absolute")


def archive_interrupted_candidate(path):
    if not path.exists() or not any(path.iterdir()):
        return None
    archive_root = path.parent / "aborted_attempts"
    archive_root.mkdir(exist_ok=True)
    destination = archive_root / f"{path.name}_{time.time_ns()}"
    os.replace(path, destination)
    atomic_json(destination / "ABORTED_ATTEMPT.json", {
        "reason": "Interrupted candidate restarted from the same paired initialization; no partial training is reused.",
        "formal_budget_accounting": "This attempt is excluded from the restarted candidate's allocated budget; its previously logged cost remains archived.",
        "resume_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return str(destination)


def candidate_order(collection, init, cfg):
    values = list(cfg["learning"]["objectives"])
    position = collection * cfg["learning"]["initializations"] + init
    shift = position % len(values)
    values = values[shift:] + values[:shift]
    if (position // len(values)) % 2:
        values.reverse()
    candidates = []
    for objective in values:
        weights = list(candidate_weights(cfg, objective))
        if position % 2:
            weights.reverse()
        candidates.extend((objective, weight) for weight in weights)
    return candidates


def train_candidate(cfg, collection, init, objective, weight, train, dev, path, device, out):
    if unit_complete(path):
        emit(stage="candidate_resume_verified", collection=collection, initialization=init,
             objective=objective, weight=weight)
        return json.loads((path / "development.json").read_text())
    aborted = archive_interrupted_candidate(path)
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "STARTED.json", {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "collection": collection, "init": init, "objective": objective,
                "weight": weight, "archived_interrupted_attempt": aborted})
    prep_start = time.perf_counter()
    lc = cfg["learning"]
    training = training_objective(cfg, objective)
    seed = seed_for(cfg["seed"], "initialization", collection, init)
    configure_torch(seed, device)
    model = PairModel(lc["hidden"], training == "relational_aux").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lc["learning_rate"])
    W = torch.as_tensor(train["W"], dtype=torch.float32, device=device)
    T = torch.as_tensor(train["T"], dtype=torch.float32, device=device)
    Y = torch.as_tensor(train["y_uc"], dtype=torch.float32, device=device)
    synchronize(device)
    initial_hash = model_hash(model)
    shared_hash = model_hash(model, shared_only=True)
    atomic_torch_save(path / "initial.pt", {"model": model.state_dict(), "initialization_seed": seed})
    proof = {"collection": collection, "init": init, "objective": objective, "weight": weight,
             "initialization_seed": seed, "model_state_sha256": initial_hash,
             "shared_encoder_pair_sha256": shared_hash,
             "model_parameter_devices": sorted({str(p.device) for p in model.parameters()}),
             "training_count_device": str(W.device), "training_tie_device": str(T.device),
             "training_label_device": str(Y.device), "parameter_count": sum(p.numel() for p in model.parameters()),
             "smoke": cfg.get("smoke", False), "pilot": cfg.get("pilot", False)}
    if device.type == "cuda" and (not W.is_cuda or any(not p.is_cuda for p in model.parameters())):
        raise RuntimeError("GPU placement failure before training.")
    atomic_json(path / "DEVICE_PROOF.json", proof)
    prep_seconds = time.perf_counter() - prep_start
    selective = (dev["y_uc"].sum(1) > 1) & (dev["y_uc"].sum(1) < lc["train_n"])
    if not selective.any():
        raise RuntimeError("Development split has no selective non-singleton cases; fixed selector undefined.")
    budget = cfg["budget"]["seconds_per_candidate"] * (cfg["budget"]["pair_candidate_multiplier"] if training == "pair" else 1)
    fractions = cfg["budget"]["checkpoint_fractions"]
    records, checkpoint_log, history = [], [], []
    updates, epoch, offset = 0, 0, len(W)
    order = None
    training_seconds = development_seconds = serialization_seconds = 0.0
    cuda_train_seconds = cuda_development_seconds = 0.0
    last_batch_seconds = 0.0
    previous_checkpoint_cost = 0.0
    last_progress = 0.0
    synchronize(device)
    started = time.perf_counter()
    emit(stage="candidate_start", collection=collection, initialization=init, objective=objective,
         weight=weight, budget_seconds=budget, device=str(device))
    for index, fraction in enumerate(fractions, 1):
        target = fraction * budget
        reserve = previous_checkpoint_cost
        trigger = max(0.0, target - reserve)
        skipped = time.perf_counter() - started >= trigger
        while time.perf_counter() - started < trigger:
            clock = PhaseClock(device).start()
            if offset >= len(W):
                epoch += 1
                order = np.random.default_rng(seed_for(cfg["seed"], "batch-order", collection, init, epoch)).permutation(len(W))
                offset = 0
            ids = torch.as_tensor(order[offset:offset + lc["batch_size"]], dtype=torch.long, device=device)
            offset += len(ids)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            P, direct = model(W[ids], T[ids])
            pair = pair_loss(P, W[ids])
            core = torch.zeros((), device=device)
            if training in ("aux", "relational_aux"):
                core = F.binary_cross_entropy(direct.clamp(1e-6, 1-1e-6), Y[ids])
            elif training == "ste":
                scores = soft_core(P, "uc", lc["temperature"])
                core = F.binary_cross_entropy(scores.clamp(1e-6, 1-1e-6), Y[ids])
            loss = pair + weight * core
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), lc["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite training gradient.")
            optimizer.step()
            last_batch_seconds, cuda_seconds = clock.stop()
            training_seconds += last_batch_seconds
            if cuda_seconds is not None:
                cuda_train_seconds += cuda_seconds
            updates += 1
            history.append({"update": updates, "epoch": epoch, "offset": offset,
                            "pair_loss": float(pair.detach()), "core_loss": float(core.detach()),
                            "total_loss": float(loss.detach()), "gradient_norm_pre_clip": float(norm),
                            "batch_wall_seconds": last_batch_seconds,
                            "batch_cuda_seconds": cuda_seconds,
                            "charged_elapsed_seconds": time.perf_counter() - started})
            elapsed = time.perf_counter() - started
            if elapsed - last_progress >= 10:
                emit(stage="candidate_progress", collection=collection, initialization=init,
                     objective=objective, weight=weight, update_count=updates, epoch=epoch,
                     charged_seconds=elapsed, budget_seconds=budget, next_checkpoint=index,
                     device=str(device))
                # A durable snapshot makes aborted-attempt cost auditable. This
                # in-loop bookkeeping counts toward the synchronized wall clock.
                atomic_json(path / "LIVE_PROGRESS.json", {"charged_seconds": time.perf_counter() - started,
                            "update_count": updates, "epoch": epoch, "objective": objective,
                            "weight": weight, "checkpoint_completed": index-1})
                last_progress = elapsed
        synchronize(device)
        checkpoint_start = time.perf_counter()
        boundary_elapsed = checkpoint_start - started
        clock = PhaseClock(device).start()
        # Native direct-head baselines are charged only for the readout they
        # actually use. Structural test diagnostics are computed later outside
        # every training/selection budget.
        prediction = predict(model, dev, cfg, device, lc["readouts"][objective])
        pending = []
        for readout in lc["readouts"][objective]:
            threshold, value = select_development_readout(
                {name: values[selective] for name, values in prediction.items()},
                dev["y_uc"][selective], cfg, readout)
            pending.append({"checkpoint_index": index, "checkpoint_fraction": fraction,
                            "epoch": epoch, "update_count": updates, "weight": weight,
                            "readout": readout, "rule": decision_rule(readout), "threshold": threshold,
                            "development_f1": value, "development_selective_count": int(selective.sum()),
                            "selection_core_class": "selective_non_singleton",
                            "checkpoint": str((path / f"checkpoint_{index:02d}.pt").relative_to(out))})
        phase_wall, phase_cuda = clock.stop()
        development_seconds += phase_wall
        if phase_cuda is not None:
            cuda_development_seconds += phase_cuda
        serialize_start = time.perf_counter()
        atomic_torch_save(path / f"checkpoint_{index:02d}.pt", {
            "model": model.state_dict(), "objective": objective, "weight": weight,
            "checkpoint_index": index, "epoch": epoch, "update_count": updates,
            "initialization_seed": seed})
        atomic_npz(path / f"development_checkpoint_{index:02d}.npz", **prediction,
                   y=dev["y_uc"], case_id=dev["case_id"], family=dev["family"], selective_mask=selective)
        records.extend(pending)
        atomic_json(path / "development.json", records)
        atomic_json(path / "history.json", history)
        serialization_seconds += time.perf_counter() - serialize_start
        synchronize(device)
        checkpoint_cost = time.perf_counter() - checkpoint_start
        elapsed = time.perf_counter() - started
        checkpoint_log.append({"checkpoint_index": index, "fraction": fraction,
                               "target_seconds": target, "reserved_checkpoint_seconds": reserve,
                               "training_trigger_seconds": trigger, "boundary_elapsed_seconds": boundary_elapsed,
                               "charged_elapsed_seconds": elapsed, "checkpoint_seconds": checkpoint_cost,
                               "last_training_batch_seconds": last_batch_seconds,
                               "no_training_since_previous_checkpoint": skipped})
        previous_checkpoint_cost = max(previous_checkpoint_cost, checkpoint_cost)
        emit(stage="candidate_checkpoint", collection=collection, initialization=init,
             objective=objective, weight=weight, checkpoint=index, checkpoints=len(fractions),
             charged_seconds=elapsed, budget_seconds=budget, update_count=updates, epoch=epoch)
    synchronize(device)
    charged = time.perf_counter() - started
    post_start = time.perf_counter()
    optimizer_devices = sorted({str(value.device) for state in optimizer.state.values()
                                for name, value in state.items() if torch.is_tensor(value) and name != "step"})
    if device.type == "cuda" and optimizer_devices != [str(device)]:
        raise RuntimeError("Optimizer state is not on the selected CUDA device.")
    atomic_json(path / "OPTIMIZER_DEVICE.json", {"devices": optimizer_devices,
                "scalar_step_devices": sorted({str(value.device) for state in optimizer.state.values()
                    for name, value in state.items() if torch.is_tensor(value) and name == "step"})})
    tolerance = cfg["budget"]["relative_tolerance"]
    valid = abs(charged - budget) <= tolerance * budget and not any(x["no_training_since_previous_checkpoint"] for x in checkpoint_log)
    timing = {"objective": objective, "weight": weight, "collection": collection, "init": init,
              "budget_seconds": budget, "charged_seconds": charged,
              "relative_budget_error": (charged-budget)/budget,
              "overshoot_seconds": max(0.0, charged-budget), "undershoot_seconds": max(0.0, budget-charged),
              "last_quantum_seconds": last_batch_seconds + checkpoint_log[-1]["checkpoint_seconds"],
              "last_minibatch_seconds": last_batch_seconds,
              "final_checkpoint_seconds": checkpoint_log[-1]["checkpoint_seconds"],
              "relative_tolerance": tolerance, "budget_valid": bool(valid),
              "train_seconds": training_seconds, "development_seconds": development_seconds,
              "checkpoint_serialization_seconds": serialization_seconds,
              "miscellaneous_charged_seconds": max(0.0, charged-training_seconds-development_seconds-serialization_seconds),
              "cuda_train_seconds": cuda_train_seconds if device.type == "cuda" else None,
              "cuda_development_seconds": cuda_development_seconds if device.type == "cuda" else None,
              "preparation_excluded_seconds": prep_seconds,
              "post_budget_audit_seconds_before_manifest": time.perf_counter()-post_start,
              "checkpoints": len(checkpoint_log), "checkpoint_records": checkpoint_log,
              "complete": True, "update_count": updates,
              "epochs_completed": epoch if offset == len(W) and epoch else max(0, epoch-1),
              "epoch_reached": epoch, "examples_seen": (max(0, epoch-1)*len(W)+offset) if epoch else 0,
              "device": str(device), "smoke": cfg.get("smoke", False), "pilot": cfg.get("pilot", False)}
    atomic_json(path / "timing.json", timing)
    finish_unit(path, [p for p in sorted(path.iterdir()) if p.is_file() and p.name != "COMPLETE.json" and p.suffix != ".tmp"])
    emit(stage="candidate_complete", collection=collection, initialization=init, objective=objective,
         weight=weight, charged_seconds=charged, budget_seconds=budget, budget_valid=valid, update_count=updates)
    return records


def select_choices(ip, cfg, candidate_records):
    started = time.perf_counter()
    choices, scans = {}, {}
    for objective in cfg["learning"]["objectives"]:
        scan_started = time.perf_counter()
        for readout in cfg["learning"]["readouts"][objective]:
            available = [r for r in candidate_records if r["objective"] == objective and r["readout"] == readout]
            choice = max(available, key=lambda r: (r["development_f1"], -r["checkpoint_index"],
                        -r["weight"], -abs(r["threshold"]-.5), -r["threshold"]))
            choices[f"{objective}/{readout}/{decision_rule(readout)}"] = dict(choice)
        scans[objective] = time.perf_counter() - scan_started
    path = ip / "CHOICES_BEFORE_TEST.json"
    if path.exists() and json.loads(path.read_text()) != choices:
        raise RuntimeError("Persisted pre-test choices changed on resume.")
    if not path.exists():
        atomic_json(path, choices)
        atomic_json(ip / "SELECTION_FROZEN.json", {"choices_sha256": sha(path),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "criterion": cfg["decisions"]["development_selection"],
                   "test_used_for_selection": False})
    if json.loads((ip / "SELECTION_FROZEN.json").read_text())["choices_sha256"] != sha(path):
        raise RuntimeError("Pre-test choice hash mismatch.")
    cost_path = ip / "SELECTION_COST.json"
    all_path = ip / "ALL_DEVELOPMENT_CANDIDATES.json"
    if cost_path.exists():
        # A resumed partial collection verifies frozen choices; it must not
        # replace the original charged selection cost with a fresh audit time.
        if not all_path.exists() or json.loads(all_path.read_text()) != candidate_records:
            raise RuntimeError("Frozen candidate universe changed on resume.")
        return choices
    atomic_json(all_path, candidate_records)
    charged = time.perf_counter() - started
    shared = max(0., charged - sum(scans.values())) / len(scans)
    costs = {name: {"winner_scan_seconds": value, "shared_freeze_seconds": shared,
                   "charged_seconds": value + shared} for name, value in scans.items()}
    atomic_json(cost_path, {"systems": costs, "charged_seconds_total": sum(x["charged_seconds"] for x in costs.values()),
                "complete": True, "included_in_complete_system_budget": True,
                "shared_cost_policy": "equal share across six complete systems",
                "audit_record_serialization_excluded": True})
    return choices


def add_edge_diagnostics(prediction, data):
    p = prediction["P"].astype(np.float64)
    q = data["P"].astype(np.float64)
    ii, jj = np.triu_indices(p.shape[1], 1)
    estimates, truth = p[:, ii, jj], q[:, ii, jj]
    clipped = np.clip(estimates, 1e-6, 1-1e-6)
    # Expected Bernoulli log loss against latent probabilities, not sampled W.
    prediction["edge_prob_mse"] = ((estimates-truth)**2).mean(1)
    prediction["edge_brier"] = prediction["edge_prob_mse"] + (truth*(1-truth)).mean(1)
    prediction["edge_logloss"] = (-(truth*np.log(clipped)+(1-truth)*np.log1p(-clipped))).mean(1)
    prediction["majority_tie_pairs"] = (estimates == .5).sum(1).astype(np.int32)
    A = p > .5
    prediction["hard_unoriented_pairs"] = (~(A | A.transpose(0, 2, 1)))[:, ii, jj].sum(1).astype(np.int32)
    prediction["reciprocal_rounding_pairs"] = ((p[:, ii, jj] == .5) ^ (p[:, jj, ii] == .5)).sum(1).astype(np.int32)
    # Each stored direction is thresholded independently. Float32 reciprocity
    # is approximate; upper-triangle ties alone do not count unoriented pairs.
    prediction["hard_uc"] = np.array([hard_core(matrix > .5, "uc") for matrix in p])
    return prediction


def evaluate_choices(ip, choices, td, cfg, device, out, collection, init, n):
    rows, required, cache = [], [], {}
    start = time.perf_counter()
    for key, choice in choices.items():
        objective = choice["objective"]
        checkpoint = choice["checkpoint"]
        path = ip / "predictions" / f"{objective}_l{choice['weight']:g}_ck{choice['checkpoint_index']:02d}_n{n}.npz"
        if checkpoint not in cache:
            model = PairModel(cfg["learning"]["hidden"], training_objective(cfg, objective) == "relational_aux").to(device)
            model.load_state_dict(torch.load(out / checkpoint, map_location=device, weights_only=True)["model"])
            cache[checkpoint] = model
        # Prediction units have their own durable integrity markers so a resumed
        # partial collection never trusts an unverified previously written NPZ.
        pred_marker = path.with_suffix(path.suffix + ".sha256.json")
        if path.exists():
            if not pred_marker.exists() or json.loads(pred_marker.read_text())["sha256"] != sha(path):
                raise RuntimeError("Saved selected prediction changed or incomplete: " + str(path))
            with np.load(path) as loaded:
                prediction = {name: loaded[name] for name in loaded.files}
        else:
            prediction = add_edge_diagnostics(predict(cache[checkpoint], td, cfg, device), td)
            prediction.update(y=td["y_uc"], family=td["family"], case_id=td["case_id"])
            atomic_npz(path, **prediction)
            atomic_json(pred_marker, {"sha256": sha(path), "checkpoint_sha256": sha(out / checkpoint),
                                      "choices_sha256": sha(ip / "CHOICES_BEFORE_TEST.json")})
        required.extend([path, pred_marker])
        scores = prediction[choice["readout"]]
        for i, (score, y) in enumerate(zip(scores, td["y_uc"])):
            hard_metrics = set_metrics(y, prediction["hard_uc"][i])
            row = {"collection": collection, "init": init, "target": "uc", "n": n,
                   "family": str(td["family"][i]), "case_id": str(td["case_id"][i]),
                   "objective": objective, "readout": choice["readout"], "rule": choice["rule"],
                   "threshold": choice["threshold"], "weight": choice["weight"],
                   "checkpoint_index": choice["checkpoint_index"], "epoch": choice["epoch"],
                   "update_count": choice["update_count"], "device": str(device),
                   "prediction_file": str(path.relative_to(out)), "prediction_index": i,
                   "edge_prob_mse": float(prediction["edge_prob_mse"][i]),
                   "edge_brier": float(prediction["edge_brier"][i]),
                   "edge_logloss": float(prediction["edge_logloss"][i]),
                   "majority_tie_pairs": int(prediction["majority_tie_pairs"][i]),
                   "hard_unoriented_pairs": int(prediction["hard_unoriented_pairs"][i]),
                   "reciprocal_rounding_pairs": int(prediction["reciprocal_rounding_pairs"][i]),
                   "hard_f1": hard_metrics["f1"], "hard_exact": hard_metrics["exact"],
                   **set_metrics(y, score.astype(bool) if choice["readout"] == "hard_uc" else decide(score, "absolute", choice["threshold"]), score)}
            rows.append(row)
    for i, y in enumerate(td["y_uc"]):
        for control, predicted in (("all", np.ones(n, bool)), ("none", np.zeros(n, bool))):
            rows.append({"collection": collection, "init": init, "target": "uc", "n": n,
                         "family": str(td["family"][i]), "case_id": str(td["case_id"][i]),
                         "objective": control, "readout": "native", "rule": "native", "threshold": 0,
                         "weight": 0, "checkpoint_index": 0, "epoch": 0, "update_count": 0,
                         "device": "control", "prediction_file": "", "prediction_index": i,
                         "edge_prob_mse": np.nan, "edge_brier": np.nan, "edge_logloss": np.nan,
                         "majority_tie_pairs": np.nan, "hard_unoriented_pairs": np.nan, "reciprocal_rounding_pairs": np.nan,
                         "hard_f1": np.nan, "hard_exact": np.nan,
                         **set_metrics(y, predicted)})
    timing_path = ip / f"TEST_EVALUATION_n{n}.json"
    atomic_json(timing_path, {"seconds_this_invocation": time.perf_counter()-start,
                            "n": n, "outside_training_selection_budget": True,
                            "saved_prediction_units": len(set(required))//2})
    required.append(timing_path)
    return rows, required


def common_warmup(cfg, device, out):
    start = time.perf_counter()
    configure_torch(seed_for(cfg["seed"], "global-common-warmup"), device)
    W = torch.ones((cfg["learning"]["batch_size"], 12, 12), device=device)
    T = torch.zeros_like(W)
    for relational in (False, True):
        model = PairModel(cfg["learning"]["hidden"], relational).to(device)
        P, direct = model(W, T)
        (pair_loss(P, W)+direct.mean()+soft_core(P, "uc", cfg["learning"]["temperature"]).mean()).backward()
    synchronize(device)
    atomic_json(out / "COMMON_WARMUP.json", {"seconds_this_invocation": time.perf_counter()-start,
                "outside_all_candidate_budgets": True, "affects_no_candidate_initial_state": True})


def run(cfg, out, device, supplied_cfg=None, collection_ids=None):
    overall = time.perf_counter()
    if supplied_cfg is None:
        supplied_cfg = json.loads((HERE / "config.json").read_text())
    if collection_ids is None:
        collection_ids = list(range(cfg["learning"]["collections"]))
    full_run = collection_ids == list(range(cfg["learning"]["collections"]))
    marker_name = "RUN_COMPLETE.json" if full_run else "SHARD_COMPLETE.json"
    lock_run(out, cfg, device, supplied_cfg, collection_ids)
    learning = out / "learning"
    if unit_complete(learning):
        if (out / marker_name).exists():
            result = json.loads((out / marker_name).read_text())
            if result["learning_complete_sha256"] != sha(learning / "COMPLETE.json") or result["run_lock_sha256"] != sha(out / "RUN_LOCK.json"):
                raise RuntimeError("Root completion record differs from verified learning or run identity.")
            emit(stage="run_resume_verified", out=str(out))
            return result
        # Recover an interruption between verified learning completion and the
        # final root marker. The collection checks below skip every completed fit.
        emit(stage="recover_root_completion", out=str(out))
    common_warmup(cfg, device, out)
    learning.mkdir(exist_ok=True)
    lc = cfg["learning"]
    for collection in collection_ids:
        cp = learning / f"collection_{collection:02d}"
        if unit_complete(cp):
            emit(stage="collection_resume_verified", collection=collection)
            continue
        cp.mkdir(exist_ok=True)
        dp = cp / "datasets"
        dp.mkdir(exist_ok=True)
        train = verify_or_create_dataset(dp / "train.npz", cfg, collection, "train", lc["train_n"], lc["train_graphs"])
        dev = verify_or_create_dataset(dp / "development.npz", cfg, collection, "development", lc["train_n"], lc["development_graphs"])
        required = [dp / "train.npz", dp / "development.npz", dp / "train.npz.sha256.json", dp / "development.npz.sha256.json"]
        rows = []
        frozen_init_choices = []
        for init in range(lc["initializations"]):
            ip = cp / "uc" / f"init_{init:02d}"
            ip.mkdir(parents=True, exist_ok=True)
            order = candidate_order(collection, init, cfg)
            order_path = ip / "CANDIDATE_ORDER.json"
            order_record = {"order": [{"objective": o, "weight": w} for o, w in order],
                            "counterbalance": "cyclic objective order, reversed every six paired units; lambda order alternates"}
            if order_path.exists() and json.loads(order_path.read_text()) != order_record:
                raise RuntimeError("Candidate counterbalanced order changed.")
            atomic_json(order_path, order_record)
            required.append(order_path)
            candidate_records = []
            initial_shared = []
            for objective, weight in order:
                path = ip / f"{objective}_lambda_{weight:g}"
                records = train_candidate(cfg, collection, init, objective, weight, train, dev, path, device, out)
                candidate_records.extend([{**record, "objective": objective} for record in records])
                required.append(path / "COMPLETE.json")
                initial_shared.append(json.loads((path / "DEVICE_PROOF.json").read_text())["shared_encoder_pair_sha256"])
            if len(set(initial_shared)) != 1:
                raise RuntimeError("Shared pair/encoder initialization mismatch across paired methods or lambda candidates.")
            choices = select_choices(ip, cfg, candidate_records)
            required.extend([ip / "CHOICES_BEFORE_TEST.json", ip / "SELECTION_FROZEN.json", ip / "ALL_DEVELOPMENT_CANDIDATES.json", ip / "SELECTION_COST.json"])
            frozen_init_choices.append((init, ip, choices))
        # Every initialization and all four systems' choices in this collection
        # are frozen and hashed before its first held-out test is generated.
        for init, ip, choices in frozen_init_choices:
            for n in lc["test_sizes"]:
                test_path = dp / f"test_n{n}.npz"
                td = verify_or_create_dataset(test_path, cfg, collection, "test", n, lc["test_graphs_per_size"])
                required.extend([test_path, test_path.with_suffix(test_path.suffix + ".sha256.json")])
                new_rows, test_files = evaluate_choices(ip, choices, td, cfg, device, out, collection, init, n)
                rows.extend(new_rows)
                required.extend(test_files)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            emit(stage="initialization_complete", collection=collection, initialization=init,
                 expected_initializations=lc["initializations"])
        write_rows(cp / "per_case_metrics.csv", rows)
        required.append(cp / "per_case_metrics.csv")
        finish_unit(cp, sorted(set(required)))
        emit(stage="collection_complete", collection=collection, completed=collection+1,
             expected=len(collection_ids), rows=len(rows))
    dataset_rows = [{"path": str(p.relative_to(out)), "sha256": sha(p)}
                    for p in sorted(learning.glob("collection_*/datasets/*.npz"))]
    write_rows(learning / "DATASET_HASHES.csv", dataset_rows)
    finish_unit(learning, [learning / "DATASET_HASHES.csv", *sorted(learning.glob("collection_*/COMPLETE.json"))])
    candidates = list(learning.glob("collection_*/uc/init_*/*_lambda_*/timing.json"))
    candidate_times = [json.loads(p.read_text()) for p in candidates]
    total_rows = sum(len(pd.read_csv(p)) for p in learning.glob("collection_*/per_case_metrics.csv"))
    selection_seconds = sum(json.loads(p.read_text())["charged_seconds_total"] for p in learning.glob("collection_*/uc/init_*/SELECTION_COST.json"))
    result = {"release": cfg["release"], "complete": True, "smoke": cfg.get("smoke", False),
              "pilot": cfg.get("pilot", False), "scientific_run": not cfg.get("smoke") and not cfg.get("pilot"),
              "collections": len(collection_ids), "collection_ids": collection_ids,
              "global_collections": lc["collections"], "full_global_run": full_run,
              "initializations": lc["initializations"],
              "candidate_count": len(candidates), "per_case_metric_rows": total_rows,
              "dataset_count": len(dataset_rows), "all_candidates_within_fixed_tolerance": all(t["budget_valid"] for t in candidate_times),
              "charged_candidate_seconds_total": sum(t["charged_seconds"] for t in candidate_times),
              "charged_final_selection_seconds_total": selection_seconds,
              "charged_complete_system_seconds_total": sum(t["charged_seconds"] for t in candidate_times) + selection_seconds,
              "allocated_candidate_seconds_total": sum(t["budget_seconds"] for t in candidate_times),
              "end_to_end_seconds_this_invocation": time.perf_counter()-overall,
              "learning_complete_sha256": sha(learning / "COMPLETE.json"),
              "run_lock_sha256": sha(out / "RUN_LOCK.json"),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "formal_inference": "Requires independent analysis to verify every candidate and system total against the fixed 2% tolerance; smoke/pilot never allow inference."}
    atomic_json(out / marker_name, result)
    emit(stage="run_complete" if full_run else "shard_training_complete", **{k: v for k, v in result.items() if k not in ("utc",)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collections", default="all", help="Comma-separated frozen collection IDs; all by default. Shards cannot enter inference separately.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cuda:0", "cpu"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Tiny isolated software QA; never scientific evidence.")
    mode.add_argument("--pilot", action="store_true", help="Tiny isolated CUDA timing trial; never confirmatory evidence.")
    args = parser.parse_args()
    supplied_cfg = json.loads(args.config.read_text())
    cfg = configure_mode(supplied_cfg, args.smoke, args.pilot)
    validate_config(cfg)
    device = choose_device(cfg, args.device)
    out = args.out.resolve()
    run(cfg, out, device, supplied_cfg, parse_collections(args.collections, cfg))


if __name__ == "__main__":
    main()
