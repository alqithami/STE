#!/usr/bin/env python3
"""Validate the fresh run and summarize the prospectively specified UC study.

Graphs and model initializations are repeated measurements inside collections.
Only the twelve independent collection means enter the two primary tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "frozen"))
from ste.common import unit_complete
from ste.metrics import holm, paired_inference, decide, set_metrics, select_threshold, thresholds
from ste.operators import hard_core

METRICS = ["f1", "exact", "precision", "recall", "target_size", "selected_size",
           "ap", "auc", "edge_prob_mse", "edge_brier", "edge_logloss", "hard_f1", "hard_exact", "majority_tie_pairs",
           "hard_unoriented_pairs", "reciprocal_rounding_pairs"]
GROUP = ["collection", "target", "n", "objective", "readout", "rule"]
DEFAULT_READOUTS = {name: ["direct" if name.endswith("direct") else "hard_uc"] for name in
                    ("pair-hard", "aux-direct", "relational-direct", "ste-hard", "aux-hard", "relational-hard")}
READY_STATUS = "STE_HARDUC_CONFIRM_V1_ANALYSIS_READY"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def effective_config(results, supplied):
    """The runner writes the effective smoke/full configuration at run start."""
    cfg_path = results / "CONFIG.json"
    if not cfg_path.is_file():
        raise RuntimeError("Missing runner CONFIG.json; do not infer the effective configuration")
    cfg = json.loads(cfg_path.read_text())
    original = json.loads(Path(supplied).read_text())
    comparable = {k: v for k, v in cfg.items() if k not in ("smoke", "pilot")}
    if not cfg.get("smoke", False) and not cfg.get("pilot", False) and comparable != original:
        raise RuntimeError("Full-run CONFIG.json differs from the supplied frozen configuration")
    frozen_path = results / "FROZEN_CONFIG.json"
    if not frozen_path.is_file():
        raise RuntimeError("Missing FROZEN_CONFIG.json")
    if json.loads(frozen_path.read_text()) != original:
        raise RuntimeError("FROZEN_CONFIG.json differs from the supplied prospective configuration")
    if cfg.get("pilot", False):
        raise RuntimeError("A timing pilot is not a completed experiment and must not enter result analysis")
    lock_path = results / "RUN_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("Missing source/configuration run lock")
    lock = json.loads(lock_path.read_text())
    from run_experiment import source_fingerprint
    if lock["config"] != cfg or lock["supplied_config"] != original or lock["source_sha256"] != source_fingerprint():
        raise RuntimeError("Analysis source or prospective configuration changed after launch")
    protocol = json.loads((results / "PROTOCOL_LOCK.json").read_text())
    if protocol != {key: lock[key] for key in ("config", "supplied_config", "source_sha256")}:
        raise RuntimeError("Global prospective protocol lock differs from run identity")
    return cfg


def ensure_complete(path):
    if not unit_complete(path):
        raise RuntimeError("Incomplete or unverified result unit: " + str(path))


def selection_audit(cfg, results):
    """Recompute every development selector, its deterministic winner and pairing.

    This checks the archived evidence, rather than trusting the reported best
    score. Held-out predictions are never inputs to this function.
    """
    from run_experiment import model_hash
    from ste.models import PairModel
    import torch
    lc = cfg["learning"]
    checked = pairing = 0
    for c in range(lc["collections"]):
        cp = results / "learning" / f"collection_{c:02d}"
        with np.load(cp / "datasets" / "development.npz", allow_pickle=False) as z:
            truth, ids, family = z["y_uc"].astype(bool), z["case_id"].astype(str), z["family"].astype(str)
        selective = (truth.sum(1) > 1) & (truth.sum(1) < lc["train_n"])
        if not selective.any():
            raise RuntimeError("No selective development cases: fixed selector undefined")
        for init in range(lc["initializations"]):
            ip = cp / "uc" / f"init_{init:02d}"
            choices_path = ip / "CHOICES_BEFORE_TEST.json"
            choices = json.loads(choices_path.read_text())
            freeze = json.loads((ip / "SELECTION_FROZEN.json").read_text())
            if freeze["choices_sha256"] != digest(choices_path) or freeze.get("test_used_for_selection") is not False:
                raise RuntimeError("Missing or inconsistent development-only selection freeze")
            candidates = []
            shared_hashes, initialization_seeds = set(), set()
            for objective in lc["objectives"]:
                mapping = lc["systems"][objective]
                for weight in ([0.] if objective == "pair-hard" else lc["core_weights"]):
                    path = ip / f"{objective}_lambda_{weight:g}"
                    ensure_complete(path)
                    proof = json.loads((path / "DEVICE_PROOF.json").read_text())
                    initialization_seeds.add(int(proof["initialization_seed"]))
                    shared_hashes.add(proof["shared_encoder_pair_sha256"])
                    initial = torch.load(path / "initial.pt", map_location="cpu", weights_only=True)
                    model = PairModel(lc["hidden"], mapping["training_objective"] == "relational_aux")
                    model.load_state_dict(initial["model"], strict=True)
                    if model_hash(model) != proof["model_state_sha256"] or model_hash(model, shared_only=True) != proof["shared_encoder_pair_sha256"]:
                        raise RuntimeError("Initial model bytes differ from device/pairing proof")
                    pairing += 1
                    if not cfg.get("smoke") and (not str(proof["training_count_device"]).startswith("cuda") or not all(str(d).startswith("cuda") for d in proof["model_parameter_devices"])):
                        raise RuntimeError("Production model/training data did not run on CUDA")
                    records = json.loads((path / "development.json").read_text())
                    expected_indices = set(range(1, len(cfg["budget"]["checkpoint_fractions"]) + 1))
                    if len(records) != len(expected_indices) or {int(r["checkpoint_index"]) for r in records} != expected_indices:
                        raise RuntimeError("Missing or duplicate development checkpoint records")
                    for row in records:
                        index = int(row["checkpoint_index"])
                        readout = mapping["readout"]
                        rule = "hard" if readout == "hard_uc" else "absolute"
                        with np.load(path / f"development_checkpoint_{index:02d}.npz", allow_pickle=False) as z:
                            prediction = {key: z[key] for key in z.files}
                        if not np.array_equal(prediction["y"].astype(bool), truth) or not np.array_equal(prediction["case_id"].astype(str), ids) or not np.array_equal(prediction["family"].astype(str), family) or not np.array_equal(prediction["selective_mask"].astype(bool), selective):
                            raise RuntimeError("Development predictions differ from frozen development split")
                        scores = prediction[readout]
                        if scores.shape != truth.shape or not np.isfinite(scores).all():
                            raise RuntimeError("Malformed development scores")
                        if rule == "hard":
                            decoded = np.array([hard_core(matrix > .5, "uc") for matrix in prediction["P"]])
                            if not np.array_equal(scores, decoded):
                                raise RuntimeError("Development hard UC is not frozen hard decoding of P")
                            threshold = .5  # Compatibility metadata, never a tuned membership cutoff.
                            value = float(np.mean([set_metrics(y, score.astype(bool))["f1"] for y, score in zip(truth[selective], scores[selective])]))
                        else:
                            threshold, value = select_threshold(scores[selective], truth[selective], thresholds(cfg), "absolute")
                        if row["readout"] != readout or row["rule"] != rule or not np.isclose(float(row["weight"]), weight, atol=1e-12, rtol=0) or not np.isclose(float(row["development_f1"]), value, atol=1e-12, rtol=0) or not np.isclose(float(row["threshold"]), threshold, atol=1e-12, rtol=0) or int(row["development_selective_count"]) != int(selective.sum()):
                            raise RuntimeError("Reported development choice does not reproduce from archived scores")
                        candidates.append({**row, "objective": objective})
                        checked += 1
            if len(shared_hashes) != 1 or len(initialization_seeds) != 1:
                raise RuntimeError("Complete systems do not share paired encoder/pair initialization")
            saved_all = json.loads((ip / "ALL_DEVELOPMENT_CANDIDATES.json").read_text())
            # Ordering is execution-order dependent, but candidate identities are not.
            canonical = lambda rows: sorted(json.dumps(r, sort_keys=True, allow_nan=False) for r in rows)
            if canonical(saved_all) != canonical(candidates):
                raise RuntimeError("Saved development candidate universe differs from every archived candidate")
            expected_keys = set()
            for objective in lc["objectives"]:
                readout = lc["systems"][objective]["readout"]
                rule = "hard" if readout == "hard_uc" else "absolute"
                key = f"{objective}/{readout}/{rule}"
                expected_keys.add(key)
                available = [r for r in candidates if r["objective"] == objective and r["readout"] == readout]
                winner = max(available, key=lambda r: (r["development_f1"], -r["checkpoint_index"], -r["weight"], -abs(r["threshold"]-.5), -r["threshold"]))
                if choices.get(key) != winner:
                    raise RuntimeError("Saved before-test choice differs from deterministic development-only winner: " + key)
            if set(choices) != expected_keys:
                raise RuntimeError("Unexpected before-test choice key")
    return checked, pairing


def budget_audit(cfg, results):
    lc = cfg["learning"]
    budget_cfg = cfg.get("budget", cfg.get("compute", {}))
    tolerance = float(budget_cfg.get("relative_tolerance", .02))
    if tolerance != .02:
        raise RuntimeError("The prospective fair-budget tolerance must remain exactly 0.02")
    records = []
    for c in range(lc["collections"]):
        cp = results / "learning" / f"collection_{c:02d}"
        for init in range(lc["initializations"]):
            ip = cp / "uc" / f"init_{init:02d}"
            for objective in lc["objectives"]:
                weights = [0.] if objective == "pair-hard" else lc["core_weights"]
                for weight in weights:
                    candidate = ip / f"{objective}_lambda_{weight:g}"
                    ensure_complete(candidate)
                    timing_path = candidate / "timing.json"
                    if not timing_path.is_file():
                        raise RuntimeError("Missing candidate timing: " + str(timing_path))
                    item = json.loads(timing_path.read_text())
                    required = ["budget_seconds", "charged_seconds", "last_quantum_seconds"]
                    for field in required:
                        if field not in item or not np.isfinite(float(item[field])) or float(item[field]) < 0:
                            raise RuntimeError(f"Missing/nonfinite timing field {field}: {timing_path}")
                    budget = float(item["budget_seconds"])
                    charged = float(item["charged_seconds"])
                    if budget <= 0:
                        raise RuntimeError("Candidate budget must be positive")
                    expected_budget = float(budget_cfg["seconds_per_candidate"])
                    if objective == "pair-hard":
                        expected_budget *= float(budget_cfg["pair_candidate_multiplier"])
                    if not np.isclose(budget, expected_budget, rtol=0, atol=1e-9):
                        raise RuntimeError("Measured candidate budget differs from the prospective configuration")
                    valid = (1 - tolerance) * budget <= charged <= (1 + tolerance) * budget
                    # The last work quantum explains excess time; it never changes the ±2% rule.
                    checkpoints = sorted(candidate.glob("checkpoint_*.pt"))
                    if not checkpoints:
                        checkpoints = sorted(candidate.glob("epoch_*.pt"))
                    expected_checkpoints = len(budget_cfg.get("checkpoint_fractions", [.025, .1, .25, .5, 1.]))
                    if len(checkpoints) != expected_checkpoints:
                        raise RuntimeError(f"Expected exactly {expected_checkpoints} retained checkpoints: " + str(candidate))
                    checkpoint_records = item.get("checkpoint_records", [])
                    if len(checkpoint_records) != expected_checkpoints:
                        raise RuntimeError("Missing checkpoint timing records")
                    no_skips = not any(record.get("no_training_since_previous_checkpoint", True) for record in checkpoint_records)
                    valid = valid and no_skips
                    records.append({"collection": c, "init": init, "objective": objective,
                                    "weight": weight, "budget_seconds": budget,
                                    "charged_seconds": charged,
                                    "relative_budget_difference": charged / budget - 1,
                                    "last_quantum_seconds": float(item["last_quantum_seconds"]),
                                    "overshoot_seconds": max(0., charged - budget),
                                    "candidate_budget_valid": bool(valid),
                                    "timing_file": str(timing_path.relative_to(results))})
    frame = pd.DataFrame(records)
    system = frame.groupby(["collection", "init", "objective"], as_index=False)[["budget_seconds", "charged_seconds"]].sum()
    system = system.rename(columns={"charged_seconds": "candidate_charged_seconds"})
    selection_costs = {}
    for c in range(lc["collections"]):
        for init in range(lc["initializations"]):
            cost_path = results / "learning" / f"collection_{c:02d}" / "uc" / f"init_{init:02d}" / "SELECTION_COST.json"
            if not cost_path.is_file():
                raise RuntimeError("Missing charged development-selection cost: " + str(cost_path))
            cost = json.loads(cost_path.read_text())
            if cost.get("complete") is not True or cost.get("included_in_complete_system_budget") is not True or set(cost.get("systems", {})) != set(lc["objectives"]):
                raise RuntimeError("Incomplete or mismatched complete-system selection-cost allocation")
            total = 0.
            shared_costs = []
            for objective in lc["objectives"]:
                row = cost["systems"][objective]
                values = [float(row[key]) for key in ("winner_scan_seconds", "shared_freeze_seconds", "charged_seconds")]
                if not np.isfinite(values).all() or min(values) < 0 or not np.isclose(values[0] + values[1], values[2], atol=1e-9, rtol=1e-9):
                    raise RuntimeError("Invalid charged development-selection cost components")
                shared_costs.append(values[1]); total += values[2]
                selection_costs[(c, init, objective)] = values[2]
            if not np.isclose(total, float(cost["charged_seconds_total"]), atol=1e-9, rtol=1e-9) or not np.allclose(shared_costs, shared_costs[0], atol=1e-9, rtol=0):
                raise RuntimeError("Development-selection common cost was not allocated equally or total differs")
    system["selection_charged_seconds"] = [selection_costs[(int(row.collection), int(row.init), row.objective)] for row in system.itertuples(index=False)]
    system["charged_seconds"] = system.candidate_charged_seconds + system.selection_charged_seconds
    system["relative_budget_difference"] = system.charged_seconds / system.budget_seconds - 1
    system["system_budget_valid"] = system.relative_budget_difference.abs() <= tolerance + 1e-12
    # All six complete systems receive the same prospective candidate-search budget.
    for _, d in system.groupby(["collection", "init"]):
        if len(d) != len(lc["objectives"]) or not np.allclose(d.budget_seconds, d.budget_seconds.iloc[0], rtol=0, atol=1e-9):
            raise RuntimeError("Unequal prospective system budgets")
    valid = bool(frame.candidate_budget_valid.all() and system.system_budget_valid.all())
    return frame, system, valid


def check_numeric(frame):
    controls = frame.objective.isin(["all", "none"]).to_numpy()
    diagnostic = {"edge_prob_mse", "edge_brier", "edge_logloss", "hard_f1", "hard_exact", "majority_tie_pairs", "hard_unoriented_pairs", "reciprocal_rounding_pairs"}
    for column in METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(float)
        if column in ("ap", "auc"):
            undefined = frame.core_class.isin(["full", "empty"]).to_numpy() | controls
            if np.any(~np.isfinite(values) & ~undefined) or np.any(np.isinf(values)):
                raise RuntimeError("Invalid ranking metric outside its undefined truth strata: " + column)
        elif not np.isfinite(values[~controls if column in diagnostic else np.ones(len(frame), bool)]).all() or np.isinf(values).any():
            raise RuntimeError("Nonfinite required metric: " + column)
        if column not in ("target_size", "selected_size", "edge_logloss", "majority_tie_pairs", "hard_unoriented_pairs", "reciprocal_rounding_pairs"):
            finite = values[np.isfinite(values)]
            if np.any(finite < -1e-7) or np.any(finite > 1 + 1e-7):
                raise RuntimeError("Metric outside [0,1]: " + column)
        elif np.any(values < 0):
            raise RuntimeError("Negative metric: " + column)


def validate_metrics(cfg, results):
    lc = cfg["learning"]
    readouts = lc.get("readouts", DEFAULT_READOUTS)
    frames = []
    prediction_cache = {}
    checks = 0
    for c in range(lc["collections"]):
        cp = results / "learning" / f"collection_{c:02d}"
        ensure_complete(cp)
        path = cp / "per_case_metrics.csv"
        frame = pd.read_csv(path)
        required = GROUP + ["init", "family", "case_id", "core_class", "threshold", "weight", "checkpoint_index"] + METRICS
        if not set(required).issubset(frame.columns):
            raise RuntimeError("Missing metric columns: " + repr(sorted(set(required) - set(frame.columns))))
        if frame.empty or frame.isna().any()[["case_id", "family", "core_class"]].any():
            raise RuntimeError("Empty or malformed metric table")
        if set(frame.collection.astype(int)) != {c} or set(frame.target) != {"uc"}:
            raise RuntimeError("Unexpected collection/target rows")
        if set(frame.init.astype(int)) != set(range(lc["initializations"])):
            raise RuntimeError("Missing initialization")
        if set(frame.n.astype(int)) != set(lc["test_sizes"]):
            raise RuntimeError("Unexpected or missing held-out size")
        allowed_systems = {(objective, readout, "hard" if readout == "hard_uc" else "absolute") for objective in lc["objectives"] for readout in readouts[objective]}
        allowed_systems.update({("all", "native", "native"), ("none", "native", "native")})
        actual_systems = set(zip(frame.objective, frame.readout, frame.rule))
        if actual_systems != allowed_systems:
            raise RuntimeError("Unexpected or missing prescribed complete system/control")
        if frame.duplicated(["collection", "init", "n", "case_id", "objective", "readout", "rule"]).any():
            raise RuntimeError("Duplicate per-case rows")
        check_numeric(frame)
        for n in lc["test_sizes"]:
            dataset = cp / "datasets" / f"test_n{n}.npz"
            if not dataset.is_file():
                raise RuntimeError("Missing raw held-out dataset: " + str(dataset))
            with np.load(dataset, allow_pickle=False) as z:
                ids = z["case_id"].astype(str)
                y = z["y_uc"].astype(bool)
                families = z["family"].astype(str)
                latent = z["P"]
            if len(ids) != lc["test_graphs_per_size"] or len(set(ids)) != len(ids) or y.shape != (len(ids), n):
                raise RuntimeError("Unexpected test dataset shape/coverage")
            idx = {value: i for i, value in enumerate(ids)}
            sizes = y.sum(1)
            classes = np.where(sizes == 1, "singleton", np.where(sizes == n, "full", np.where(sizes > 0, "selective_non_singleton", "empty")))
            for init in range(lc["initializations"]):
                ip = cp / "uc" / f"init_{init:02d}"
                choices_path = ip / "CHOICES_BEFORE_TEST.json"
                if not choices_path.is_file():
                    raise RuntimeError("Missing development-only choices")
                choices = json.loads(choices_path.read_text())
                for objective in lc["objectives"]:
                    for readout in readouts[objective]:
                        rule = "hard" if readout == "hard_uc" else "absolute"
                        key = f"{objective}/{readout}/{rule}"
                        if key not in choices:
                            raise RuntimeError("Missing prescribed development choice: " + key)
                        subset = frame[(frame.init == init) & (frame.n == n) & (frame.objective == objective) & (frame.readout == readout) & (frame.rule == rule)]
                        if len(subset) != len(ids) or set(subset.case_id.astype(str)) != set(ids):
                            raise RuntimeError(f"Missing held-out cases c{c} init{init} n{n} {key}")
                        choice = choices[key]
                        for field in ["threshold", "weight", "checkpoint_index"]:
                            values = subset[field].to_numpy(float)
                            if not np.isfinite(values).all() or not np.allclose(values, float(choice[field]), rtol=0, atol=1e-9):
                                raise RuntimeError("Reported test choice differs from the frozen before-test choice: " + field)
                        if "prediction_file" not in subset.columns:
                            raise RuntimeError("Missing prediction_file; archived predictions must permit score verification")
                        for row in subset.itertuples(index=False):
                            k = idx[str(row.case_id)]
                            if row.family != families[k] or row.core_class != classes[k] or row.target_size != sizes[k]:
                                raise RuntimeError("Truth metadata differs from the raw test dataset")
                            predpath = (results / str(row.prediction_file)).resolve()
                            if not predpath.is_relative_to(results.resolve()) or not predpath.is_file():
                                raise RuntimeError("Prediction path escapes or is missing")
                            if predpath not in prediction_cache:
                                with np.load(predpath, allow_pickle=False) as z:
                                    prediction_cache[predpath] = {field: z[field] for field in z.files}
                            prediction = prediction_cache[predpath]
                            predids = prediction["case_id"].astype(str)
                            if not np.array_equal(predids, ids) or not np.array_equal(prediction["y"].astype(bool), y):
                                raise RuntimeError("Predictions and raw test truth do not align")
                            scores = prediction[readout][k]
                            if scores.shape != (n,) or not np.isfinite(scores).all():
                                raise RuntimeError("Invalid saved membership scores")
                            selected = scores.astype(bool) if rule == "hard" else decide(scores, rule, float(row.threshold))
                            calculated = set_metrics(y[k], selected, scores)
                            for metric in ["f1", "exact", "precision", "recall", "target_size", "selected_size", "ap", "auc"]:
                                actual = float(getattr(row, metric))
                                value = float(calculated[metric])
                                if not (np.isnan(actual) and np.isnan(value)) and not np.isclose(actual, value, rtol=1e-6, atol=1e-7):
                                    raise RuntimeError("Saved scores disagree with reported metric: " + metric)
                            P = prediction["P"][k]
                            if P.shape != (n, n) or not np.isfinite(P).all() or np.any(P < 0) or np.any(P > 1):
                                raise RuntimeError("Invalid saved pair-probability matrix")
                            if not np.allclose(P + P.T, 1., rtol=0, atol=2e-7) or not np.allclose(np.diag(P), .5, rtol=0, atol=2e-7):
                                raise RuntimeError("Saved pair probabilities violate model reciprocity/diagonal convention")
                            ii, jj = np.triu_indices(n, 1)
                            q = latent[k, ii, jj]
                            prob_mse = np.mean((P[ii, jj] - q) ** 2)
                            brier = np.mean((P[ii, jj] - q) ** 2 + q * (1 - q))
                            if not np.isclose(float(row.edge_prob_mse), prob_mse, rtol=1e-5, atol=1e-7):
                                raise RuntimeError("Edge probability MSE differs from saved P and latent truth")
                            if not np.isclose(float(row.edge_brier), brier, rtol=1e-5, atol=1e-7):
                                raise RuntimeError("Expected edge Brier score differs from saved P and latent truth")
                            clipped = np.clip(P[ii, jj].astype(np.float64), 1e-6, 1 - 1e-6)
                            logloss = np.mean(-(q * np.log(clipped) + (1 - q) * np.log1p(-clipped)))
                            if not np.isclose(float(row.edge_logloss), logloss, rtol=1e-5, atol=1e-7):
                                raise RuntimeError("Expected edge log loss differs from saved P and latent truth")
                            ties = int(np.sum(P[ii, jj] == .5))
                            if float(row.majority_tie_pairs) != ties:
                                raise RuntimeError("Saved majority tie count disagrees with P")
                            adjacency = P > .5
                            unoriented = int(np.sum(~(adjacency | adjacency.T)[ii, jj]))
                            rounding = int(np.sum((P[ii, jj] == .5) != (P[jj, ii] == .5)))
                            if float(row.hard_unoriented_pairs) != unoriented or float(row.reciprocal_rounding_pairs) != rounding:
                                raise RuntimeError("Saved unoriented/reciprocal-rounding pair count disagrees with P")
                            decoded = hard_core(P > .5, "uc")
                            if not np.array_equal(decoded, prediction["hard_uc"][k]):
                                raise RuntimeError("Saved hard UC differs from frozen decoding of P")
                            hard_metrics = set_metrics(y[k], decoded)
                            for metric in ["f1", "exact"]:
                                if not np.isclose(float(getattr(row, "hard_" + metric)), hard_metrics[metric], rtol=1e-6, atol=1e-7):
                                    raise RuntimeError("Hard UC metric differs from saved P and truth")
                            checks += 1
                for control in ("all", "none"):
                    control_rows = frame[(frame.init == init) & (frame.n == n) & (frame.objective == control)]
                    if len(control_rows) != len(ids) or set(control_rows.case_id.astype(str)) != set(ids):
                        raise RuntimeError("Missing constant-control cases")
                    for row in control_rows.itertuples(index=False):
                        k = idx[str(row.case_id)]
                        expected = set_metrics(y[k], np.full(n, control == "all", bool))
                        if row.family != families[k] or row.core_class != classes[k]:
                            raise RuntimeError("Control truth metadata differs from raw test data")
                        for metric in ("f1", "exact", "precision", "recall", "target_size", "selected_size"):
                            if not np.isclose(float(getattr(row, metric)), float(expected[metric]), rtol=1e-6, atol=1e-7):
                                raise RuntimeError("Constant control metric differs from raw truth")
        frames.append(frame)
        prediction_cache.clear()
    full = pd.concat(frames, ignore_index=True)
    return full, checks


def collection_means(frame, extra=()):
    # Explicit two-stage averaging: model initializations never become independent units.
    within = frame.groupby(GROUP + list(extra) + ["init"], dropna=False)[METRICS].mean().reset_index()
    return within.groupby(GROUP + list(extra), dropna=False)[METRICS].mean().reset_index()


def summarize(units, extra=()):
    rows = []
    keys = GROUP[1:] + list(extra)
    for values, group in units.groupby(keys, dropna=False):
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        row["collections"] = int(len(group))
        for metric in METRICS:
            x = group[metric].dropna()
            row[metric] = float(x.mean()) if len(x) else np.nan
            row[metric + "_sd"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
            row[metric + "_sem"] = float(x.sem(ddof=1)) if len(x) > 1 else np.nan
            row[metric + "_defined_collections"] = int(len(x))
        rows.append(row)
    return pd.DataFrame(rows)


def primary_analysis(cfg, cores, formal_valid):
    n = int(cfg.get("primary_n", 24))
    endpoint = cores[(cores.n == n) & (cores.core_class == "selective_non_singleton")]
    comparisons = [("aux-direct", "direct"), ("relational-direct", "direct")]
    rows, differences = [], []
    expected = set(range(cfg["learning"]["collections"]))
    for right, readout in comparisons:
        left = endpoint[(endpoint.objective == "ste-hard") & (endpoint.readout == "hard_uc") & (endpoint.rule == "hard")].set_index("collection").f1
        other = endpoint[(endpoint.objective == right) & (endpoint.readout == readout) & (endpoint.rule == "absolute")].set_index("collection").f1
        if set(left.index) != expected or set(other.index) != expected:
            raise RuntimeError("Missing selective primary endpoint collection; never silently drop it")
        delta = (left.sort_index() - other.sort_index()).to_numpy(float)
        if not np.isfinite(delta).all():
            raise RuntimeError("Nonfinite primary collection differences")
        base = {"target": "uc", "n": n, "core_class": "selective_non_singleton", "rule": "hard-versus-absolute",
                "left": "ste-hard", "left_readout": "hard_uc", "right": right, "right_readout": readout,
                "left_rule": "hard", "right_rule": "absolute",
                "difference": float(delta.mean()), "units": len(delta), "positive_units": int((delta > 0).sum()),
                "formal_inference_valid": bool(formal_valid), "p": None, "p_holm": None,
                "ci_low": None, "ci_high": None}
        if formal_valid:
            base.update(paired_inference(delta))
        rows.append(base)
        differences.extend([{**{k: v for k, v in base.items() if k in ("target", "n", "core_class", "left", "left_readout", "right", "right_readout")},
                             "collection": c, "difference": float(d)} for c, d in enumerate(delta)])
    if formal_valid:
        for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
            row["p_holm"] = float(adjusted)
    return rows, differences


def analyze(results, config):
    results = Path(results).resolve()
    cfg = effective_config(results, config)
    smoke = bool(cfg.get("smoke", False))
    if not (results / "RUN_COMPLETE.json").is_file():
        raise RuntimeError("Missing RUN_COMPLETE.json; a launch or partial run cannot be analyzed as complete")
    complete = json.loads((results / "RUN_COMPLETE.json").read_text())
    ensure_complete(results / "learning")
    if complete["learning_complete_sha256"] != digest(results / "learning" / "COMPLETE.json") or complete["run_lock_sha256"] != digest(results / "RUN_LOCK.json"):
        raise RuntimeError("Root completion marker does not identify the verified full learning run")
    lock = json.loads((results / "RUN_LOCK.json").read_text())
    if lock["collection_ids"] != list(range(cfg["learning"]["collections"])):
        raise RuntimeError("Partial shards cannot enter global inference")
    if not smoke and (cfg["learning"]["collections"] != 12 or cfg["learning"]["initializations"] != 3):
        raise RuntimeError("The full prospective study requires all 12 collections and 3 initializations")
    development_checks, initialization_checks = selection_audit(cfg, results)
    frame, score_checks = validate_metrics(cfg, results)
    budget, systems, budget_valid = budget_audit(cfg, results)
    out = results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "PER_CASE_METRICS.csv", index=False)
    units = collection_means(frame)
    cores = collection_means(frame, ["core_class"])
    families = collection_means(frame, ["family"])
    units.to_csv(out / "COLLECTION_MEANS.csv", index=False)
    cores.to_csv(out / "CORE_CLASS_COLLECTION_MEANS.csv", index=False)
    families.to_csv(out / "FAMILY_COLLECTION_MEANS.csv", index=False)
    summarize(units).to_csv(out / "SUMMARY.csv", index=False)
    summarize(cores, ["core_class"]).to_csv(out / "CORE_CLASS_SUMMARY.csv", index=False)
    summarize(families, ["family"]).to_csv(out / "FAMILY_SUMMARY.csv", index=False)
    budget.to_csv(out / "CANDIDATE_BUDGET_AUDIT.csv", index=False)
    systems.to_csv(out / "SYSTEM_BUDGET_AUDIT.csv", index=False)
    formal_valid = not smoke and budget_valid
    primary, differences = primary_analysis(cfg, cores, formal_valid)
    pd.DataFrame(primary).to_csv(out / "PRIMARY_COMPARISONS.csv", index=False)
    pd.DataFrame(differences).to_csv(out / "PRIMARY_COLLECTION_DIFFERENCES.csv", index=False)
    invalid_reasons = (["software smoke: never scientific evidence"] if smoke else []) + ([] if budget_valid else ["one or more candidates or complete systems violate the fixed ±2% charged-time tolerance"])
    interpretation = {
        "smoke": smoke, "scientific_evidence": not smoke, "primary_inference_valid": formal_valid,
        "invalid_inference_reasons": invalid_reasons, "budget_valid": budget_valid,
        "primary_endpoint": "selective non-singleton UC F1 at n=24; STE hard UC versus ordinary direct and relational direct heads",
        "primary_multiplicity": "Holm across exactly two prespecified comparisons",
        "independent_units": "12 fresh independent data collections; average graphs and 3 initializations within each collection",
        "sign_flip_assumption": "paired independent collection differences are exchangeable under independent sign reversals under the null; not assumption-free",
        "interval_assumption": "95% t intervals assume independent collection differences and approximate normality",
        "budget_definition": "synchronized charged elapsed time for training, development prediction and readout selection, checkpoint/development serialization, and in-loop progress/bookkeeping; complete-system totals additionally include winner scanning and an equal share of common choice/development-universe freezing; initialization, data creation and held-out test evaluation reported separately",
        "budget_tolerance": "fixed ±2% for each candidate and each complete system; measured work quanta explain overshoot but never relax this rule",
        "selection": "checkpoint and lambda chosen for each system’s deployed readout on n=12 development selective non-singleton UC F1; only direct heads also select an absolute membership threshold; all choices freeze before held-out test evaluation",
        "secondary_only": ["n=48", "all-graph overlap", "exact recovery", "selected cardinality", "AP/AUROC", "families", "other core strata", "saved structural scores of auxiliary models as raw diagnostics; no separately selected auxiliary structural system", "latent pair-probability MSE, expected Bernoulli Brier and expected Bernoulli log loss", "auxiliary-trained hard-UC systems"],
        "hard_decode_ties": "Each directed edge uses its own stored float32 P>.5 comparison. hard_unoriented_pairs counts pairs with neither edge; reciprocal_rounding_pairs counts pairs where exactly one stored direction rounds to .5. majority_tie_pairs is the historical upper-triangle .5 count, not the number of unoriented pairs. No floating-point tie repair is applied.",
        "application_scope": "synthetic generator mix and archived finite tournament-core supervision motivation; this study does not establish human preference transfer",
        "theoretical_scope": "no general learnability, universal gradient or asymptotic superiority claim",
        "completed_not_superior": "completion does not imply an advantage; all paired differences and unfavorable outcomes are retained",
        "prior_run": "fresh seed and data; these are a separately frozen follow-up family, not extra endpoints merged into the previous Holm family",
        "saved_score_metric_checks": score_checks,
        "development_selection_checks": development_checks,
        "initialization_pairing_checks": initialization_checks,
    }
    atomic_json(out / "INTERPRETATION_CONSTRAINTS.json", interpretation)
    lines = ["STE Hard UC Confirmation v1 — measured results only", "",
             "SOFTWARE SMOKE; NOT SCIENTIFIC EVIDENCE" if smoke else "COMPLETE FRESH FOLLOW-UP; completion alone is not evidence of superiority",
             "Fair budget valid: " + str(budget_valid), "Primary inference valid: " + str(formal_valid), ""]
    for row in primary:
        text = f"STE hard UC minus {row['right']}: selective UC F1 mean difference {row['difference']:.6f}, n={row['n']}, collection units={row['units']}"
        if formal_valid:
            text += f", 95% t interval [{row['ci_low']:.6f}, {row['ci_high']:.6f}], two-sided sign-flip p={row['p']:.6g}, Holm p={row['p_holm']:.6g}"
        else:
            text += "; DESCRIPTIVE ONLY; formal p-values and confidence intervals suppressed"
        lines.append(text)
    lines += ["", *invalid_reasons, "", "Read every collection and candidate budget row before interpreting aggregates.",
              "No manuscript claim is updated automatically. Review limitations and raw predictions before making a submission claim."]
    (out / "READ_MEASURED_RESULTS.txt").write_text("\n".join(lines) + "\n")
    required = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "COMPLETE.json" and p.suffix != ".tmp")
    atomic_json(out / "COMPLETE.json", {"files": {str(p.relative_to(out)): digest(p) for p in required},
                "status": READY_STATUS, "smoke": smoke, "primary_inference_valid": formal_valid})
    print(json.dumps({"status": READY_STATUS, "smoke": smoke, "budget_valid": budget_valid,
                      "primary_inference_valid": formal_valid, "collections": cfg["learning"]["collections"],
                      "initializations": cfg["learning"]["initializations"], "saved_score_metric_checks": score_checks}), flush=True)
    return interpretation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.results, args.config)
