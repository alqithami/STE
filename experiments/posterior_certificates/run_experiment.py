#!/usr/bin/env python3
"""Exploratory, source-held-out human-profile benchmark; never launches remotely."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, betaincc

from stepc.data import prepare_data, observed_view, seed_for, sha256
from stepc.posterior import (exact_uc, jeffreys_orientation_probabilities,
                             jeffreys_pair_means, gfm_decision_from_joint, soft_uc_plugin,
                             mixture_orientation_probabilities)
from stepc.graphs import strict_partial_relation, hard_orientation_relation
from stepc.certificates import simultaneous_hoeffding_intervals, certificate_bounds

HERE = Path(__file__).resolve().parent
SOFTWARE_RELEASE = "STE-Posterior-Certificates-v2"
METHODS = ("empirical_bayes", "ordinary", "relational", "learned_edge", "posterior_mixture")
PRIMARY_FRACTION = .10
REPLAY_TOLERANCE = 1e-5


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as h:
        np.savez_compressed(h, **arrays)
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(h, keys); writer.writeheader(); writer.writerows(rows)


def source_hashes():
    paths = list(HERE.glob("*.py")) + list((HERE/"stepc").glob("*.py")) + list((HERE/"tests").glob("*.py"))
    paths += [p for p in (HERE/"config.json", HERE/"requirements.txt", HERE/"PROTOCOL.md",
                          HERE/"run_all.sh", HERE/"CORRECTION_NOTICE.md",
                          HERE/"FROZEN_V1_PARENT_MANIFEST.json") if p.exists()]
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths)}


def validate_future_output_path(out):
    """Refuse the historical v1 namespace; software v2 needs a new run root."""
    out = Path(out).expanduser().resolve()
    for part in out.parts:
        if part.lower() in {"ste-posterior-certificates-v1", "ste_posterior_certificates_v1"}:
            raise ValueError("V2 must not write to the frozen v1 result/source namespace")
    return out


def sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def setup_device(requested, smoke, audit=False):
    import torch
    if requested == "cpu" and not (smoke or audit):
        raise RuntimeError("Production and timing pilot require CUDA; CPU is software QA/audit only")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    device = torch.device(requested)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return device


def configure_seed(seed, device):
    import torch
    seed = int(seed) % (2**63-1)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@contextmanager
def exclusive_run(out):
    """Fresh directories only. Interrupted runs must be restarted at a new root."""
    out = validate_future_output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = [p.name for p in out.iterdir() if p.name not in ("run.log",)]
    if entries:
        raise RuntimeError("Existing output refused; use a new run directory: " + ", ".join(entries[:8]))
    lock = out/"RUNNING.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, (str(os.getpid())+"\n").encode()); os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def joint_from_memberships(memberships):
    Y = np.asarray(memberships, dtype=bool)
    if Y.ndim != 2 or len(Y) < 1:
        raise ValueError("Memberships must have shape draws by alternatives")
    draws, n = Y.shape
    sizes = Y.sum(1)
    joint = np.zeros((n, n+1))
    for s in range(n+1):
        joint[:, s] = (Y & (sizes[:, None] == s)).mean(0)
    cards = np.bincount(sizes, minlength=n+1) / draws
    gfm, expected = gfm_decision_from_joint(joint, cards)
    return {"marginals": Y.mean(0), "membership_size_joint": joint,
            "cardinality_probabilities": cards, "gfm": np.asarray(gfm, bool),
            "half": Y.mean(0) >= .5, "expected_f1": float(expected)}


def draw_memberships(q, draws, seed, components=None, weights=None, batch_size=256):
    """Common NumPy random stream; one shared component per whole tournament."""
    q = np.asarray(q, dtype=float)
    n = len(q); ij = np.triu_indices(n, 1)
    components = q[None] if components is None else np.asarray(components, dtype=float)
    weights = np.ones(1) if weights is None else np.asarray(weights, dtype=float)
    if components.shape[1:] != (n,n) or len(weights) != len(components):
        raise ValueError("Invalid mixture shapes")
    if not np.isfinite(components).all() or np.any((components < 0)|(components > 1)):
        raise ValueError("Invalid orientation probabilities")
    if not np.allclose(components+components.transpose(0,2,1), 1, atol=1e-6):
        raise ValueError("Nonreciprocal orientation probabilities")
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-6)):
        raise ValueError("Invalid mixture weights")
    if not np.allclose(q, mixture_orientation_probabilities(components, weights), atol=1e-12, rtol=0):
        raise ValueError("q must represent the normalized native upper-edge sampling law")
    # Separate streams preserve common edge uniforms across independent and
    # mixture methods without coupling an edge uniform to the component draw.
    rng = np.random.default_rng(int(seed))
    component_rng = np.random.default_rng(seed_for(int(seed),"global-components"))
    component_ids = component_rng.choice(len(weights), size=int(draws), p=weights/weights.sum())
    memberships = np.empty((draws,n), dtype=bool)
    for start in range(0, draws, batch_size):
        stop = min(start+batch_size, draws)
        p = components[component_ids[start:stop]][:, ij[0], ij[1]]
        edge = rng.random(p.shape) < p
        A = np.zeros((stop-start,n,n), bool)
        A[:,ij[0],ij[1]] = edge; A[:,ij[1],ij[0]] = ~edge
        memberships[start:stop] = exact_uc(A)
    return memberships, component_ids


def partial_relation_uc(adjacency):
    """Original covering definition on a strict, possibly incomplete relation.

    Unlike the two-step reachability theorem for complete tournaments, this
    preserves all vertices when no observed strict-majority edge is available.
    """
    A=strict_partial_relation(adjacency)
    if A.ndim != 2:
        raise ValueError("Partial relation UC expects one square adjacency matrix")
    missing=(A[None,:,:] & ~A[:,None,:]).any(-1)
    covers=A & ~missing
    return ~covers.any(0)


def set_metrics(y, prediction, score=None, q=None, fullA=None):
    y, prediction = np.asarray(y,bool), np.asarray(prediction,bool)
    tp = int((y & prediction).sum()); ps = int(prediction.sum()); ys = int(y.sum())
    values = {"f1": 2*tp/(ps+ys) if ps+ys else 1., "precision": tp/ps if ps else float(ys == 0),
              "recall": tp/ys if ys else 1., "exact_set": int(np.array_equal(y,prediction)),
              "predicted_cardinality": ps, "true_cardinality": ys,
              "cardinality_error": ps-ys,"absolute_cardinality_error":abs(ps-ys),
              "exact_cardinality":int(ps==ys)}
    if score is not None:
        values["uc_brier"] = float(np.mean((np.asarray(score)-y)**2))
    if q is not None and fullA is not None:
        ij = np.triu_indices(len(q),1)
        p = np.asarray(q)[ij]; labels = np.asarray(fullA,dtype=float)[ij]
        values["orientation_brier"] = float(np.mean((p-labels)**2))
        ece = 0.
        for lo in np.arange(0.,1.,.1):
            mask = (p>=lo)&((p<lo+.1) if lo<.9 else (p<=1.))
            if mask.any():
                ece += mask.mean()*abs(float(p[mask].mean()-labels[mask].mean()))
        values["orientation_ece_10bins"] = float(ece)
    return values


def dev_score(predictions, cases):
    grouped = {}
    for pred, case in zip(predictions, cases):
        grouped.setdefault(case["profile"], []).append(set_metrics(case["y"], pred)["f1"])
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def select_threshold(scores, cases, grid):
    records = [(dev_score([decide_scores(s,t) for s in scores], cases), float(t)) for t in grid]
    f1, t = max(records, key=lambda x:(x[0], -abs(x[1]-.5), -x[1]))
    return t, f1


def decide_scores(scores, threshold):
    scores=np.asarray(scores)
    if threshold==0.:
        return np.ones(scores.shape,bool)
    if threshold==1.:
        return np.zeros(scores.shape,bool)
    return scores>=threshold


def beta_q(W, alpha):
    W = np.asarray(W,float)
    if not np.isfinite(alpha) or alpha<=0 or not np.isfinite(W).all() or np.any(W<0):
        raise ValueError("Invalid Beta prior/counts")
    i,j = np.triu_indices(len(W),1)
    a,b=alpha+W[i,j],alpha+W[j,i]
    small=betaincc(np.minimum(a,b),np.maximum(a,b),.5)
    small=np.where(a==b,.5,small)
    q=np.full(W.shape,.5)
    q[i,j]=np.where(a<=b,small,1-small)
    q[j,i]=np.where(a<=b,1-small,small)
    return q


def fit_empirical_bayes(train):
    """Symmetric Beta scalar ML using full decisive train-source counts only."""
    unique = {c["profile"]: c for c in train}
    sources = {}
    for case in unique.values():
        W = np.asarray(case["fullW"],float); i,j = np.triu_indices(len(W),1)
        sources.setdefault(case["source"],[]).append((W[i,j],W[j,i]))
    def objective(log_alpha):
        a = np.exp(log_alpha)
        # A composite working likelihood, balanced source -> profile -> pair;
        # repeated nested subsamples are not treated as independent cohorts.
        return -float(np.mean([np.mean([np.mean(betaln(x+a,y+a)-betaln(a,a))
            for x,y in profiles]) for profiles in sources.values()]))
    result = minimize_scalar(objective, bounds=(-8.,8.), method="bounded", options={"xatol":1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Empirical Bayes scalar fit failed")
    return {"alpha":float(np.exp(result.x)), "log_marginal_likelihood":float(-result.fun),
            "training_profiles": sorted(unique), "fit_target":"full decisive counts; no UC labels",
            "weighting":"equal source means of profile means of unordered-pair log marginals",
            "likelihood_scope":"composite working likelihood; no cross-pair or resample independence claim"}


def atomic_torch_save(path, data):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("wb") as h:
        torch.save(data,h); h.flush(); os.fsync(h.fileno())
    os.replace(tmp,path)


def evaluate_development(model, cases, cfg, device, seed, grid):
    from stepc.learning import predict_observed
    scores=[]; masks=[]
    for case in cases:
        result = predict_observed(model, observed_view(case), device)
        if "scores" in result:
            scores.append(result["scores"])
        else:
            memberships,_ = draw_memberships(result["q"], cfg["posterior"]["development_draws"],
                seed_for(seed,"development",case["case_id"]), result["components"],result["weights"])
            masks.append(joint_from_memberships(memberships)["gfm"])
    if scores:
        threshold,f1 = select_threshold(scores,cases,grid)
        return {"threshold":threshold,"development_f1":f1,"selection_decoder":"direct_fixed_grid"}
    return {"threshold":None,"development_f1":dev_score(masks,cases),"selection_decoder":"native_gfm"}


def fit_method(method, train, dev, cfg, path, device, fold, init, allocation):
    """Every fit, dev check and snapshot is charged to the method's allocation."""
    import torch
    from stepc.learning import PredictiveModel, supervised_loss, model_hash, FEATURE_VERSION
    path.mkdir(parents=True, exist_ok=True)
    sync(device); start=time.perf_counter()
    seed=seed_for(cfg["seed"],"paired-initialization",fold,init)
    dev=[c for c in dev if np.isclose(c["fraction"],PRIMARY_FRACTION)]
    if not dev:
        raise ValueError("Development source has no fixed 10% cases")
    grid=cfg["learning"]["threshold_grid"]
    records=[]; proof=[]
    if method == "empirical_bayes":
        params=fit_empirical_bayes(train)
        ck=path/"selected.json"; atomic_json(ck,params)
        masks=[]
        for case in dev:
            q=beta_q(case["W"],params["alpha"])
            members,_=draw_memberships(q,cfg["posterior"]["development_draws"],seed_for(seed,"development",case["case_id"]))
            masks.append(joint_from_memberships(members)["gfm"])
        records.append({"method":method,"weight":0.,"checkpoint":str(ck.relative_to(path.parents[2])),
                        "checkpoint_sha256":sha256(ck),"development_f1":dev_score(masks,dev),
                        "threshold":None,"selection_decoder":"native_gfm","alpha":params["alpha"],"step":0})
        proof.append({"optimizer":"scipy bounded scalar ML","label_access":"fullW only", "finite":True,
                      "early_native_convergence":True,"paired_initialization_seed":seed})
    else:
        lambdas=cfg["learning"]["lambdas"] if method=="posterior_mixture" else [0.]
        for weight in lambdas:
            configure_seed(seed,device)
            candidate_start=time.perf_counter()
            candidate_budget=allocation/len(lambdas)
            model=PredictiveModel(method,cfg["learning"]["hidden"]).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning"]["learning_rate"])
            initial_hash=model_hash(model)
            generator=torch.Generator(device=device).manual_seed(seed % (2**63-1))
            order_rng=np.random.default_rng(seed_for(seed,"training-order"))
            steps=0; finite_grad=True; snapshots=0; losses=[]; checkpoint_times=[.20,.45,.70]
            order=order_rng.permutation(len(train)); position=0
            def checkpoint():
                nonlocal snapshots
                sync(device)
                ck=path/f"lambda_{weight:g}_checkpoint_{snapshots+1}.pt"
                atomic_torch_save(ck,{"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "kind":method,"hidden":model.hidden,"feature_version":FEATURE_VERSION,"seed":seed,
                    "step":steps,"weight":weight,"state_hash":model_hash(model)})
                development=evaluate_development(model,dev,cfg,device,seed,grid)
                records.append({"method":method,"weight":float(weight),"checkpoint":str(ck.relative_to(path.parents[2])),
                    "checkpoint_sha256":sha256(ck),"step":steps,"snapshot":snapshots+1,**development})
                snapshots+=1
            # Reserve at least 15% for final dev evaluation and serialization.
            while snapshots<3:
                elapsed=time.perf_counter()-candidate_start
                if elapsed >= checkpoint_times[snapshots]*candidate_budget and steps:
                    checkpoint()
                    if time.perf_counter()-start>2*allocation:
                        raise RuntimeError("Noninterruptible development work exceeded 2x allocation")
                    continue
                if elapsed >= .85*candidate_budget:
                    checkpoint(); continue
                if position==len(order):
                    order=order_rng.permutation(len(train));position=0
                case=train[int(order[position])];position+=1
                optimizer.zero_grad(set_to_none=True)
                loss,values=supervised_loss(model,case,device,float(weight),cfg["posterior"]["training_draws"],generator)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                finite_grad=finite_grad and bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
                if not finite_grad:
                    raise RuntimeError("Nonfinite or absent training gradients")
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("Nonfinite clipped gradient")
                optimizer.step(); steps+=1
                if steps<=4 or steps%100==0:
                    losses.append({"step":steps,"loss":float(loss.detach()),**values})
            sync(device)
            changed=model_hash(model)!=initial_hash
            if not steps or not changed:
                raise RuntimeError("Training failed to change model parameters")
            opt_devices=sorted({str(v.device) for s in optimizer.state.values() for k,v in s.items() if torch.is_tensor(v) and k!="step"})
            if device.type=="cuda" and opt_devices != [str(device)]:
                raise RuntimeError("Optimizer state is not on the selected CUDA device")
            proof.append({"paired_initialization_seed":seed,"weight":float(weight),"initial_state_hash":initial_hash,
                "final_state_hash":model_hash(model),"parameters_changed":changed,"finite_gradients":finite_grad,
                "steps":steps,"model_devices":sorted({str(p.device) for p in model.parameters()}),
                "training_feature_device":str(device),"training_label_device":str(device),"optimizer_devices":opt_devices,
                "candidate_allocated_seconds":candidate_budget,"candidate_charged_seconds":time.perf_counter()-candidate_start,
                "loss_samples":losses})
    chosen=max(records,key=lambda r:(r["development_f1"],-r.get("snapshot",0),-r["weight"]))
    atomic_json(path/"DEVELOPMENT_CANDIDATES.json",records)
    atomic_json(path/"DEVICE_AND_GRADIENT_PROOF.json",proof)
    atomic_json(path/"SELECTED_BEFORE_TEST.json",chosen)
    sync(device);elapsed=time.perf_counter()-start
    timing={"method":method,"allocated_seconds":allocation,"charged_seconds":elapsed,
            "includes":"initialization, fitting, all development checks, checkpoints and logs",
            "equal_allocation_valid":elapsed<=allocation*1.02,
            "policy":"upper allocation; early native convergence permitted; 2% fixed overrun gate",
            "candidate_count":len(proof)}
    atomic_json(path/"TIMING.json",timing)
    return chosen,timing


def load_predictor(choice,out,device):
    if choice["method"]=="empirical_bayes":
        return json.loads((out/choice["checkpoint"]).read_text())
    import torch
    from stepc.learning import PredictiveModel, FEATURE_VERSION, model_hash
    state=torch.load(out/choice["checkpoint"],map_location="cpu",weights_only=True)
    if state["feature_version"] != FEATURE_VERSION:
        raise RuntimeError("Checkpoint feature version differs")
    model=PredictiveModel(state["kind"],state["hidden"]).to(device)
    model.load_state_dict(state["model"])
    if model_hash(model)!=state["state_hash"]:
        raise RuntimeError("Checkpoint tensor hash mismatch")
    return model


def select_count_plugin(dev,cfg):
    cases=[c for c in dev if np.isclose(c["fraction"],PRIMARY_FRACTION)]
    scores=[soft_uc_plugin(jeffreys_pair_means(c["W"]),tau=.035,gamma=.035) for c in cases]
    threshold,f1=select_threshold(scores,cases,cfg["learning"]["threshold_grid"])
    return {"method":"count_soft_plugin","threshold":threshold,"development_f1":f1,
            "tau":.035,"gamma":.035,"role":"secondary development-calibrated plugin"}


def selective_row(case, delta):
    intervals=simultaneous_hoeffding_intervals(case["W"],delta,
        sampling_model="finite_cohort",fixed_nonadaptive=True)
    bounds=certificate_bounds(intervals)
    y=np.asarray(case["y"],bool); inner=np.asarray(bounds.inner,bool);outer=np.asarray(bounds.outer,bool)
    decided=inner|~outer
    return {"delta":float(delta),"certified_fraction":float(decided.mean()),
        "inner_cardinality":int(inner.sum()),"outer_cardinality":int(outer.sum()),
        "abstention_fraction":float((~decided).mean()),
        "false_inclusions":int((inner&~y).sum()),"false_exclusions":int((~outer&y).sum()),
        "selective_accuracy":float((inner[decided]==y[decided]).mean()) if decided.any() else None,
        "coverage_guarantee":False,"guarantee_label":"finite-cohort descriptive interval diagnostic"},intervals,bounds


def case_metadata(case,fold,init,method,readout):
    return {"fold":int(fold),"initialization":int(init),"source":case["source"],"profile":case["profile"],
            "case_id":case["case_id"],"fraction":float(case["fraction"]),"replicate":int(case["replicate"]),
            "n":int(case["n"]),"method":method,"readout":readout}


def evaluate_tests(cases, choices, cfg, out, device, fold, init, rows, selective, records):
    from stepc.learning import predict_observed
    base=out/f"fold_{fold:02d}"/f"init_{init:02d}"
    predictors={k:load_predictor(v,out,device) for k,v in choices.items() if k in METHODS}
    for case in cases:
        view=observed_view(case)
        # No full counts, orientations, or labels are handed to any predictor.
        cid=case["case_id"]; shared_seed=seed_for(cfg["seed"],"test-draws",fold,init,cid)
        raw=base/"predictions"/cid
        for method in (*METHODS,"count_jeffreys","count_raw","count_soft_plugin"):
            sync(device);inference_started=time.perf_counter()
            prediction={}; masks={}; q=None; scores=None
            if method=="count_raw":
                # Preserve the raw strict observed-majority relation; unresolved ties stay absent.
                A=np.asarray(view["W"])>np.asarray(view["W"]).T
                masks["raw_majority_uc"]=partial_relation_uc(A)
                prediction["adjacency"]=A
            elif method=="count_soft_plugin":
                P=jeffreys_pair_means(view["W"])
                scores=soft_uc_plugin(P,tau=.035,gamma=.035)
                prediction.update(P=P,scores=scores)
                masks["dev_calibrated_secondary"]=decide_scores(scores,choices[method]["threshold"])
            else:
                if method=="count_jeffreys":
                    q=jeffreys_orientation_probabilities(view["W"])
                    prediction={"q":q,"components":q[None],"weights":np.ones(1)}
                elif method=="empirical_bayes":
                    q=beta_q(view["W"],predictors[method]["alpha"])
                    prediction={"q":q,"components":q[None],"weights":np.ones(1)}
                else:
                    prediction=predict_observed(predictors[method],view,device)
                    q=prediction.get("q")
                if q is None:
                    scores=prediction["scores"]
                    masks["dev_calibrated_direct"]=decide_scores(scores,choices[method]["threshold"])
                else:
                    memberships, component_ids=draw_memberships(q,cfg["posterior"]["test_draws"],shared_seed,
                        prediction["components"],prediction["weights"])
                    joint=joint_from_memberships(memberships)
                    scores=joint["marginals"]
                    masks={"gfm":joint["gfm"],"half":joint["half"],
                        "hard_q_diagnostic":partial_relation_uc(hard_orientation_relation(q))}
                    prediction.update(memberships=memberships,component_ids=component_ids,draw_seed=np.array(shared_seed,dtype=np.uint64),
                        membership_size_joint=joint["membership_size_joint"],cardinality_probabilities=joint["cardinality_probabilities"],
                        marginals=scores,gfm_expected_f1=np.array(joint["expected_f1"]))
            sync(device);inference_seconds=time.perf_counter()-inference_started
            arrays={**prediction,"W":np.asarray(view["W"]),"T":np.asarray(view["T"]),
                "y":np.asarray(case["y"],bool),"fullA":np.asarray(case["fullA"],bool)}
            arrays.update({"decision_"+k:np.asarray(v,bool) for k,v in masks.items()})
            target=raw/(method+".npz");atomic_npz(target,**arrays)
            records.append({**case_metadata(case,fold,init,method,"raw"),"path":str(target.relative_to(out)),
                "choice_key":method,"choice_path":str((base/"CHOICES_BEFORE_TEST.json").relative_to(out)),
                "draw_seed":shared_seed if q is not None else None,"draws":cfg["posterior"]["test_draws"] if q is not None else 0,
                "inference_seconds":inference_seconds})
            for readout,mask in masks.items():
                rows.append({**case_metadata(case,fold,init,method,readout),
                    "inference_seconds":inference_seconds,
                    **set_metrics(case["y"],mask,scores,q,case["fullA"])})
        # Each delta has its own row. These are descriptive finite-cohort diagnostics.
        for delta in cfg["certificates"]["deltas"]:
            values,intervals,bounds=selective_row(case,delta)
            selective.append({**case_metadata(case,fold,init,"selective_certificates","bounds"),**values})
            atomic_npz(raw/f"certificate_delta_{delta:g}.npz",known=intervals.known,lower=intervals.lower,
                upper=intervals.upper,inner=bounds.inner,outer=bounds.outer,y=case["y"],W=case["W"])


def descriptive_summary(rows, fraction=PRIMARY_FRACTION, cohort="all"):
    """Average resamples and initializations per profile, then six sources equally."""
    grouped={}
    for row in rows:
        if not np.isclose(float(row["fraction"]),fraction):
            continue
        target_size=int(row["true_cardinality"])
        if cohort=="singleton" and target_size!=1:
            continue
        if cohort=="selective_non_singleton" and target_size==1:
            continue
        key=(row["method"],row["readout"],row["source"],row["profile"])
        grouped.setdefault(key,[]).append(row)
    metric_keys=("f1","precision","recall","exact_set","predicted_cardinality","true_cardinality","cardinality_error",
        "absolute_cardinality_error","exact_cardinality","orientation_brier","orientation_ece_10bins","uc_brier","inference_seconds")
    profile_rows=[]
    for (method,readout,source,profile), values in grouped.items():
        profile_rows.append({"method":method,"readout":readout,"source":source,"profile":profile,
            "cohort":"singleton" if int(values[0]["true_cardinality"])==1 else "selective_non_singleton",
            **{k:float(np.mean([float(v[k]) for v in values if k in v and v[k] != ""])) for k in metric_keys if any(k in v and v[k] != "" for v in values)}})
    sources={}
    for row in profile_rows:
        sources.setdefault((row["method"],row["readout"],row["source"]),[]).append(row)
    source_rows=[]
    for (method,readout,source),values in sources.items():
        source_rows.append({"method":method,"readout":readout,"source":source,
            **{k:float(np.mean([v[k] for v in values if k in v])) for k in metric_keys if any(k in v for v in values)}})
    systems={}
    for row in source_rows:
        systems.setdefault((row["method"],row["readout"]),[]).append(row)
    summary=[]
    for (method,readout),values in systems.items():
        summary.append({"method":method,"readout":readout,"sources":len(values),"fraction":fraction,"cohort":cohort,
            "aggregation":"resamples/init per profile; profiles per source; sources equally",
            "inference":"descriptive only; no p-values, population generalization or superiority claim",
            **{k:float(np.mean([v[k] for v in values if k in v])) for k in metric_keys if any(k in v for v in values)}})
    return profile_rows,source_rows,summary


def normalized_config(config,smoke,pilot):
    cfg=json.loads(json.dumps(config));cfg.setdefault("seed",20261007)
    if cfg.get("release") != SOFTWARE_RELEASE:
        raise ValueError("Config release must match corrected v2 software; v1 replay needs frozen v1 source")
    lc=cfg.setdefault("learning",{})
    lc.setdefault("hidden",32);lc.setdefault("learning_rate",.001)
    lc.setdefault("initializations",3);lc.setdefault("lambdas",[.1,1.])
    lc.setdefault("threshold_grid",[0.]+[round(x,2) for x in np.arange(.05,.951,.05)]+[1.])
    lc.setdefault("methods",list(METHODS))
    if lc["methods"] != list(METHODS):
        raise ValueError("The five specified methods must remain fixed")
    budget=cfg.setdefault("budget",{})
    budget.setdefault("production_seconds",120.);budget.setdefault("pilot_seconds",30.);budget.setdefault("smoke_seconds",2.)
    budget.setdefault("checkpoint_fractions",[.20,.45,.70]);budget.setdefault("reserve_fraction",.15);budget.setdefault("relative_overrun_gate",.02)
    pc=cfg.setdefault("posterior",{})
    pc.setdefault("development_draws",512);pc.setdefault("test_draws",4096);pc.setdefault("training_draws",64)
    cfg.setdefault("certificates",{}).setdefault("deltas",[.01,.05,.10])
    cfg["mode"]="smoke" if smoke else "pilot" if pilot else "production"
    if smoke or pilot:
        lc["initializations"]=1
    if smoke:
        cfg.setdefault("data",{})["resamples"]=1;cfg["data"]["replicates"]=1
        pc["development_draws"]=32;pc["test_draws"]=64;pc["training_draws"]=8
    if not smoke:
        if lc["lambdas"] != [.1,1.] or pc["training_draws"]!=64 or pc["test_draws"]!=4096:
            raise ValueError("Prospective lambda/train/test draw choices changed")
        if budget["production_seconds"]!=120. or budget["pilot_seconds"]!=30.:
            raise ValueError("Prospective upper allocations must be 120s production / 30s pilot")
        if cfg["seed"]!=20261007 or sorted(cfg["data"]["fractions"])!=[.05,.1,.25] or cfg["data"]["resamples"]!=8 or cfg["data"].get("replicates",8)!=8:
            raise ValueError("Fixed seed, all three fractions and eight resamples required")
        if pc["development_draws"]!=512 or lc["initializations"]!=(1 if pilot else 3):
            raise ValueError("Production needs three paired initializations; pilot one; dev needs 512 draws")
        expected_grid=[0.]+[round(x,2) for x in np.arange(.05,.951,.05)]+[1.]
        if lc["threshold_grid"]!=expected_grid:
            raise ValueError("Fixed threshold grid with all/none endpoints required")
        if budget["checkpoint_fractions"]!=[.20,.45,.70] or budget["reserve_fraction"]!=.15 or budget["relative_overrun_gate"]!=.02:
            raise ValueError("Fixed checkpoint/reserve/2% allocation gate changed")
    return cfg


def run(config,out,device,smoke=False,pilot=False,skip_audit=False):
    import torch, scipy
    out=validate_future_output_path(out);cfg=normalized_config(config,smoke,pilot)
    runtime={"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__,"scipy":scipy.__version__,
        "cuda":torch.version.cuda,"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "tf32":False,"mode":cfg["mode"],"classification":"software QA" if smoke else "timing pilot" if pilot else "exploratory source-held-out benchmark on previously observed cohort"}
    with exclusive_run(out):
        started=time.perf_counter()
        atomic_json(out/"RUN_LOCK.json",{"config":cfg,"source_sha256":source_hashes(),"runtime":runtime})
        atomic_json(out/"CONFIG.json",cfg);atomic_json(out/"ENVIRONMENT.json",runtime)
        data_started=time.perf_counter();data=prepare_data(HERE,cfg,out/"data")
        cases=data["cases"];folds=data["folds"][:2] if smoke else data["folds"]
        atomic_json(out/"FOLDS.json",folds)
        timing={"data_preparation_seconds":time.perf_counter()-data_started,"methods":[],"test_seconds":0.,
            "count_baseline_development_seconds":0.,"common_choices_serialization_seconds":0.}
        rows=[];selective=[];records=[]
        allocation=cfg["budget"][cfg["mode"]+"_seconds"]
        for fold in folds:
            f=int(fold.get("fold",fold.get("fold_id")))
            devsource=fold.get("development_source",fold.get("dev_source"))
            train=[c for c in cases if c["source"] in fold["train_sources"]]
            dev=[c for c in cases if c["source"]==devsource]
            test=[c for c in cases if c["source"]==fold["test_source"]]
            for init in range(cfg["learning"]["initializations"]):
                base=out/f"fold_{f:02d}"/f"init_{init:02d}";base.mkdir(parents=True,exist_ok=True)
                choices={}
                # Rotate ordering without altering paired seed or allocations.
                shift=(f+init)%len(METHODS);order=METHODS[shift:]+METHODS[:shift]
                for method in order:
                    choice,clock=fit_method(method,train,dev,cfg,base/method,device,f,init,allocation)
                    choices[method]=choice;timing["methods"].append({"fold":f,"initialization":init,**clock})
                    print(json.dumps({"stage":"fit","fold":f,"init":init,"method":method,"seconds":clock["charged_seconds"]}),flush=True)
                count_start=time.perf_counter()
                choices["count_soft_plugin"]=select_count_plugin(dev,cfg)
                timing["count_baseline_development_seconds"]+=time.perf_counter()-count_start
                common_start=time.perf_counter()
                atomic_json(base/"CHOICES_BEFORE_TEST.json",{"choices":choices,"selection_sources":[devsource],
                    "training_sources":fold["train_sources"],"test_source":fold["test_source"],"selection_fraction":PRIMARY_FRACTION,
                    "test_predictions_not_started":True,"created_unix_ns":time.time_ns()})
                timing["common_choices_serialization_seconds"]+=time.perf_counter()-common_start
                test_start=time.perf_counter()
                evaluate_tests(test,choices,cfg,out,device,f,init,rows,selective,records)
                sync(device);timing["test_seconds"]+=time.perf_counter()-test_start
        write_csv(out/"per_case_metrics.csv",rows);write_csv(out/"selective_metrics.csv",selective)
        profile_rows,source_rows,summary=descriptive_summary(rows)
        write_csv(out/"per_profile_10pct.csv",profile_rows);write_csv(out/"per_source_10pct.csv",source_rows)
        write_csv(out/"descriptive_summary_10pct.csv",summary)
        for fraction in (.05,.25):
            _,_,secondary=descriptive_summary(rows,fraction=fraction)
            write_csv(out/f"descriptive_summary_{int(100*fraction)}pct.csv",secondary)
        for cohort in ("singleton","selective_non_singleton"):
            _,_,subset=descriptive_summary(rows,cohort=cohort)
            write_csv(out/f"descriptive_summary_10pct_{cohort}.csv",subset)
        atomic_json(out/"PREDICTION_INDEX.json",records)
        timing["fit_and_test_wall_seconds"]=time.perf_counter()-started
        timing["equal_allocation_valid"]=all(x["equal_allocation_valid"] for x in timing["methods"])
        atomic_json(out/"TIMING.json",timing)
        expected={"folds":len(folds),"initializations":cfg["learning"]["initializations"],
            "prediction_files":len(records),"metric_rows":len(rows),"selective_rows":len(selective),
            "test_cases_per_fold":{str(f.get("fold",f.get("fold_id"))):sum(c["source"]==f["test_source"] for c in cases) for f in folds},
            "methods":list(METHODS),"all_test_methods":[*METHODS,"count_jeffreys","count_raw","count_soft_plugin"]}
        atomic_json(out/"EXPECTED_COVERAGE.json",expected)
        files=[p for p in out.rglob("*") if p.is_file() and p.name not in ("RUNNING.lock","run.log","FIT_COMPLETE.json","COMPLETE.json","AUDIT.json","FAILED.json")]
        atomic_json(out/"FIT_COMPLETE.json",{"status":"FIT_COMPLETE_AWAITING_AUDIT","hashes":{str(p.relative_to(out)):sha256(p) for p in sorted(files)},"equal_allocation_valid":timing["equal_allocation_valid"]})
        if not skip_audit:
            from audit_results import audit
            audit(out,device=str(device))
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=HERE/"config.json")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    modes=p.add_mutually_exclusive_group();modes.add_argument("--smoke",action="store_true");modes.add_argument("--pilot",action="store_true")
    p.add_argument("--audit-only",action="store_true");p.add_argument("--skip-audit",action="store_true")
    p.add_argument("--replay-report",type=Path,help="New external report for --audit-only; preserves audit markers")
    args=p.parse_args()
    device=setup_device(args.device,args.smoke,audit=args.audit_only)
    if args.audit_only:
        from audit_results import audit
        audit(args.out,device=str(device),replay_report=args.replay_report);return
    if args.replay_report is not None:
        p.error("--replay-report requires --audit-only")
    cfg=json.loads(args.config.read_text())
    previously_locked=(args.out/"RUN_LOCK.json").exists()
    try:
        run(cfg,args.out,device,args.smoke,args.pilot,args.skip_audit)
    except Exception as exc:
        if not previously_locked and args.out.exists() and (args.out/"RUN_LOCK.json").exists() and not (args.out/"FIT_COMPLETE.json").exists():
            atomic_json(args.out/"FAILED.json",{"error":str(exc),"restart_policy":"new output directory required; partial states never resumed"})
        raise


if __name__=="__main__":
    main()
