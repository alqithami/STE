"""Deterministic software QA; this script does not generate research results."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "frozen"))
import numpy as np
import torch
from ste.common import atomic_json, configure_torch
from ste.models import PairModel
from ste.operators import hard_core, hard_core_independent, soft_uc, numpy_reference
from ste.metrics import select_threshold, set_metrics, holm
from run_experiment import (SYSTEMS, add_edge_diagnostics, candidate_order, candidate_weights, configure_mode,
    decision_rule, choose_device, model_hash, parse_collections, predict, select_choices, select_development_readout, validate_config)
from merge import check_shard_coverage, compatible_runtime
import copy
import tempfile


def model_pair_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(("encoder.", "pair.")):
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(device):
    configure_torch(40407, device)
    count = 0
    for n in range(2, 6):
        i, j = np.triu_indices(n, 1)
        for bits in itertools.product([False, True], repeat=len(i)):
            a = np.zeros((n, n), dtype=bool)
            a[i, j] = bits; a[j, i] = ~np.asarray(bits)
            assert np.array_equal(hard_core(a, "uc"), hard_core_independent(a, "uc"))
            count += 1
    rng = np.random.default_rng(40407)
    parity = 0
    for n in [2, 3, 4, 6, 12]:
        z = rng.uniform(-.4, .4, (n, n)); p = .5 + (z-z.T)/2
        np.fill_diagonal(p, .5)
        t = torch.tensor(p, dtype=torch.double, device=device, requires_grad=True)
        actual = soft_uc(t)
        assert np.max(np.abs(actual.detach().cpu().numpy()-numpy_reference(p, "uc"))) < 1e-10
        perm = rng.permutation(n)
        assert torch.allclose(soft_uc(t[perm][:, perm]), actual[perm], atol=1e-10, rtol=1e-10)
        if n <= 4:
            assert torch.autograd.gradcheck(soft_uc, (t,), eps=1e-6, atol=2e-5, rtol=1e-3)
        # At finite temperature the neutral reference has a known closed form.
        if n >= 3:
            neutral = torch.full((n, n), .5, dtype=torch.double, device=device)
            assert torch.allclose(soft_uc(neutral), torch.full((n,), .625, dtype=torch.double, device=device), atol=1e-12)
        if device.type == "cuda":
            cpu = torch.tensor(p, dtype=torch.float32)
            assert torch.allclose(soft_uc(cpu), soft_uc(cpu.to(device)).cpu(), atol=5e-6, rtol=5e-6)
        parity += 1

    configure_torch(4242, device); plain = PairModel().to(device)
    configure_torch(4242, device); relational = PairModel(relational=True).to(device)
    assert model_pair_hash(plain) == model_pair_hash(relational)
    w = torch.tensor(rng.integers(0, 10, (2, 6, 6)), dtype=torch.float32, device=device)
    ties = torch.zeros_like(w)
    for model in [plain, relational]:
        p, direct = model(w, ties)
        assert torch.allclose(p+p.transpose(1, 2), torch.ones_like(p), atol=2e-7)
        perm = torch.tensor([2, 5, 0, 4, 1, 3], device=device)
        moved, dm = model(w[:, perm][:, :, perm], ties[:, perm][:, :, perm])
        assert torch.allclose(moved, p[:, perm][:, :, perm], atol=2e-6)
        assert torch.allclose(dm, direct[:, perm], atol=2e-6)

    # Calibration optimum verified independently by enumerating thresholds.
    y = np.array([[1,1,0,0], [0,1,1,0], [1,0,0,1]], bool)
    s = np.array([[.8,.7,.2,.1], [.3,.8,.6,.4], [.9,.2,.1,.7]])
    grid = [-1e-6,0,.25,.5,.75,1,1.000001]
    threshold, value = select_threshold(s, y, grid)
    expected = max(np.mean([set_metrics(a, b>=h)["f1"] for a,b in zip(y,s)]) for h in grid)
    assert abs(value-expected) < 1e-12
    assert np.allclose(holm([.01,.04]), [.02,.04])
    # Tied P uses explicitly incomplete adjacency; two independent oracles agree.
    a = np.array([[0,1,0], [0,0,1], [0,0,0]], bool)
    assert np.array_equal(hard_core(a, "uc"), hard_core_independent(a, "uc"))
    cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
    validate_config(cfg)
    assert cfg["seed"] != 2026100407
    try: choose_device(cfg, "cpu")
    except RuntimeError: pass
    else: raise AssertionError("Production silently fell back to CPU")
    assert parse_collections("0,2,4,6,8,10", cfg) == [0,2,4,6,8,10]
    for bad in ["0,0", "12", "-1", "", "0,no"]:
        try: parse_collections(bad, cfg)
        except ValueError: pass
        else: raise AssertionError("Malformed shard IDs accepted")
    assert check_shard_coverage([{"collection_ids": list(range(6))}, {"collection_ids": list(range(6,12))}], list(range(12))) == list(range(12))
    for bad in [[{"collection_ids": [0,1]}], [{"collection_ids": list(range(12))}, {"collection_ids": [0]}]]:
        try: check_shard_coverage(bad, list(range(12)))
        except RuntimeError: pass
        else: raise AssertionError("Partial/overlapping shards accepted for full inference")
    runtime = {"torch":"2.6.0+cu124", "numpy":"1.26.4", "scipy":"1.13.1", "pandas":"2.2.3", "cuda_runtime":"12.4", "gpu_name":"NVIDIA L40S", "gpu_capability":[8,9]}
    assert compatible_runtime(runtime, dict(runtime))
    assert not compatible_runtime(runtime, {**runtime, "gpu_name":"NVIDIA A100"})
    # Exactly 11 candidate fits and 220 seconds for each of six complete systems.
    order = candidate_order(0,0,cfg)
    assert len(order) == 11 and set(s for s,_ in order) == set(SYSTEMS)
    for system in SYSTEMS:
        allocation = sum(110 * (2 if w == 0 else 1) for w in candidate_weights(cfg, system))
        assert allocation == 220
    # Known binary hard sets are evaluated exactly; no threshold grid is consulted.
    hard = np.array([[1,1,0,0], [0,1,0,1], [1,0,0,1]], bool)
    altered = copy.deepcopy(cfg); altered["decisions"]["threshold_step"] = 2.0
    threshold, hard_value = select_development_readout({"hard_uc": hard}, y, altered, "hard_uc")
    assert threshold == .5 and abs(hard_value - (1 + .5 + 1)/3) < 1e-12
    try: select_development_readout({"hard_uc": hard.astype(float)}, y, cfg, "hard_uc")
    except RuntimeError: pass
    else: raise AssertionError("Continuous hard set accepted")
    # Every system starts from exactly the same paired relation parameters.
    shared = []
    for system, (training, _) in SYSTEMS.items():
        configure_torch(81818, device)
        paired = PairModel(cfg["learning"]["hidden"], training == "relational_aux").to(device)
        shared.append(model_hash(paired, shared_only=True))
    assert len(set(shared)) == 1
    # Repeated model ties are omitted in both directions, rather than arbitrarily oriented.
    tied = np.full((4,4), .5)
    adjacency = tied > .5
    assert not adjacency.any()
    assert np.array_equal(hard_core(adjacency, "uc"), hard_core_independent(adjacency, "uc"))
    # Adversarial float32 boundary: one stored direction ties, the reverse wins.
    rounded = np.array([[[.5,.5],[np.nextafter(np.float32(.5), np.float32(1)),.5]]], dtype=np.float32)
    diag = add_edge_diagnostics({"P":rounded}, {"P":rounded.astype(np.float64)})
    assert diag["majority_tie_pairs"].tolist() == [1]
    assert diag["hard_unoriented_pairs"].tolist() == [0]
    assert diag["reciprocal_rounding_pairs"].tolist() == [1]
    assert np.array_equal(diag["hard_uc"][0], hard_core_independent(rounded[0] > .5, "uc"))
    # Decoder-specific checkpoints are frozen before any test and tie-break independently.
    records = []
    for system, (_, readout) in SYSTEMS.items():
        for checkpoint in (1,2):
            records.append({"objective":system, "readout":readout, "rule":decision_rule(readout),
                            "development_f1": .6 if checkpoint == 1 else .8, "checkpoint_index":checkpoint,
                            "weight":0. if system == "pair-hard" else .1, "threshold":.5})
    with tempfile.TemporaryDirectory() as temp:
        choices = select_choices(Path(temp), cfg, records)
        assert len(choices) == 6 and all(x["checkpoint_index"] == 2 for x in choices.values())
        altered_records = copy.deepcopy(records); altered_records[0]["development_f1"] = 1
        try: select_choices(Path(temp), cfg, altered_records)
        except RuntimeError: pass
        else: raise AssertionError("Frozen choices were overwritten")
    return {"passed": True, "classification": "software QA only; no model-performance evidence",
            "device": str(device), "strict_uc_tournaments_checked": count,
            "numpy_torch_parity_sizes": parity, "finite_difference_sizes": 3,
            "paired_encoder_initialization": True, "model_reciprocity_and_equivariance": True,
            "threshold_enumeration": True, "holm_known_case": True,
            "cuda_formula_parity_checked": device.type == "cuda",
            "hard_selection_has_no_membership_cutoff": True, "fresh_root_seed": True,
            "six_system_pairing": True, "equal_system_allocations": True,
            "shards_disjoint_complete_and_matching_hardware": True,
            "fully_tied_pairs_omit_both_directions": True, "one_sided_float32_boundary_reported": True, "development_choice_freeze": True, "cpu_production_refused": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; no CPU fallback")
    result = run(torch.device("cuda:0" if args.device == "cuda" else "cpu"))
    if args.out:
        atomic_json(args.out, result)
    print(json.dumps(result, indent=2))
