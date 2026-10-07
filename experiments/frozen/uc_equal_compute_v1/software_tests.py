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
    return {"passed": True, "classification": "software QA only; no model-performance evidence",
            "device": str(device), "strict_uc_tournaments_checked": count,
            "numpy_torch_parity_sizes": parity, "finite_difference_sizes": 3,
            "paired_encoder_initialization": True, "model_reciprocity_and_equivariance": True,
            "threshold_enumeration": True, "holm_known_case": True,
            "cuda_formula_parity_checked": device.type == "cuda"}


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
