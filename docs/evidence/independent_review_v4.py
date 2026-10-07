#!/usr/bin/env python3
"""Read-only external audit of STE Posterior Certificates completed output.

This script never invokes the original metric/summary/audit functions and never
modifies the experiment source or result tree. Only the immutable neural model
architecture is reused for selected-checkpoint replay on the recorded device. It writes a
new JSON report outside both trees and refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zipfile

# The architecture import is read-only even when the source directory is writable.
sys.dont_write_bytecode = True
# Match the immutable runner's deterministic CUDA BLAS setup before Torch loads.
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import numpy as np
from scipy.special import betaincc, expit, logsumexp

SOURCES = ("00007", "00019", "00021", "00022", "00067", "00073")
FITTED = ("empirical_bayes", "ordinary", "relational", "learned_edge", "posterior_mixture")
METHODS = (*FITTED, "count_jeffreys", "count_raw", "count_soft_plugin")
NATIVE = {"empirical_bayes":"gfm", "ordinary":"dev_calibrated_direct",
          "relational":"dev_calibrated_direct", "learned_edge":"gfm",
          "posterior_mixture":"gfm", "count_jeffreys":"gfm",
          "count_raw":"raw_majority_uc", "count_soft_plugin":"dev_calibrated_secondary"}
EXCLUDED = {"RUNNING.lock", "run.log", "FIT_COMPLETE.json", "COMPLETE.json", "AUDIT.json", "FAILED.json"}
METRICS = ("f1", "precision", "recall", "exact_set", "predicted_cardinality", "true_cardinality",
           "cardinality_error", "absolute_cardinality_error", "exact_cardinality", "uc_brier",
           "orientation_brier", "orientation_ece_10bins", "inference_seconds")
SELECTIVE = {"00007-00000068.soi", "00007-00000086.soi", "00019-00000004.toi"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jread(path):
    return json.loads(Path(path).read_text())


def cread(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def aread(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def check_close(actual, expected, context, tolerance=1e-10):
    require(np.asarray(actual).shape == np.asarray(expected).shape, "Shape mismatch: " + context)
    require(np.isfinite(np.asarray(actual, dtype=float)).all(), "Nonfinite: " + context)
    difference = float(np.max(np.abs(np.asarray(actual, dtype=float)-np.asarray(expected, dtype=float)))) if np.asarray(actual).size else 0.
    require(difference <= tolerance, "Numerical mismatch: " + context + " " + str(difference))
    return difference


def uc_cover(A):
    """Set-inclusion covering oracle, including incomplete relations."""
    A = np.asarray(A, dtype=bool)
    outgoing = [set(np.flatnonzero(A[i])) - {i} for i in range(len(A))]
    return np.array([not any(v in outgoing[u] and outgoing[v] <= outgoing[u]
                             for u in range(len(A)) if u != v) for v in range(len(A))])


def recorded_general_cover(A):
    """Reproduce recorded generalized cover arithmetic, including self-cover.

    This is for archival record verification ONLY. Reflexive/bidirectional
    graphs do not satisfy the intended irreflexive tournament readout contract.
    A self-loop covers its own vertex and can therefore produce an empty set.
    """
    A = np.asarray(A,dtype=bool)
    outgoing = [set(np.flatnonzero(A[i])) for i in range(len(A))]
    return np.array([not any(v in outgoing[u] and outgoing[v] <= outgoing[u]
                            for u in range(len(A))) for v in range(len(A))])


def parse_raw(path):
    """Independent PrefLib rank-group parser; absent ranks remain unobserved."""
    metadata, records = {}, []
    for text in Path(path).read_text(encoding="utf-8-sig").splitlines():
        text = text.strip()
        if not text:
            continue
        if text.startswith("#"):
            if ":" in text:
                key, value = text[1:].split(":", 1)
                metadata[key.strip()] = value.strip()
            continue
        count, ranking = text.split(":", 1)
        # Split at commas outside braces, preserving tied ranks as one group.
        groups, token, depth = [], "", 0
        for character in ranking + ",":
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            if character == "," and depth == 0:
                groups.append([int(v.strip()) for v in token.strip().strip("{}").split(",")])
                token = ""
            else:
                token += character
        require(depth == 0 and int(count) > 0, "Malformed raw ballot")
        records.append((int(count), groups))
    ids = sorted(int(k.rsplit(" ", 1)[1]) for k in metadata if k.startswith("ALTERNATIVE NAME "))
    require(len(ids) == int(metadata["NUMBER ALTERNATIVES"]), "Alternative count mismatch")
    position = {value:index for index, value in enumerate(ids)}
    ranks = np.full((len(records),len(ids)), -1, dtype=np.int64)
    multiplicity = np.array([count for count, _ in records], dtype=np.int64)
    for row, (_, groups) in enumerate(records):
        for rank, group in enumerate(groups):
            for value in group:
                require(value in position and ranks[row,position[value]] == -1, "Repeated/unknown alternative")
                ranks[row,position[value]] = rank
    require(multiplicity.sum() == int(metadata["NUMBER VOTERS"]), "Raw voter count mismatch")
    return ranks, multiplicity, np.array(ids)


def raw_counts(ranks, multiplicity):
    n = ranks.shape[1]
    wins, ties = np.zeros((n,n),dtype=np.int64), np.zeros((n,n),dtype=np.int64)
    for i in range(n):
        for j in range(i+1,n):
            observed = (ranks[:,i] >= 0) & (ranks[:,j] >= 0)
            wins[i,j] = np.dot(multiplicity, observed & (ranks[:,i] < ranks[:,j]))
            wins[j,i] = np.dot(multiplicity, observed & (ranks[:,j] < ranks[:,i]))
            ties[i,j] = ties[j,i] = np.dot(multiplicity, observed & (ranks[:,i] == ranks[:,j]))
    return wins, ties


def beta_orientation(W, alpha):
    q = np.full(W.shape, .5)
    for i in range(len(W)):
        for j in range(i+1,len(W)):
            a, b = W[i,j] + alpha, W[j,i] + alpha
            # Evaluate the smaller tail then reflect to preserve small values.
            if a <= b:
                q[i,j] = betaincc(a,b,.5)
                q[j,i] = 1-q[i,j]
            else:
                q[j,i] = betaincc(b,a,.5)
                q[i,j] = 1-q[j,i]
    return q


def plugin_score(W):
    P = (W+.5)/(W+W.T+1.)
    n = len(P)
    if n == 1:
        return P, np.ones(1)
    D = expit((P-.5)/.035)
    np.fill_diagonal(D,0.)
    result = []
    for v in range(n):
        cover = []
        for u in range(n):
            if u == v:
                continue
            witness = np.array([D[v,w]*(1-D[u,w]) for w in range(n) if w != v and w != u])
            softmax = .035*(logsumexp(witness/.035)-math.log(len(witness))) if len(witness) else 0.
            cover.append(D[u,v]*(1-softmax))
        result.append(1-.035*(logsumexp(np.array(cover)/.035)-math.log(n-1)))
    return P, np.array(result)


def threshold_decision(scores, threshold):
    if threshold == 0.:
        return np.ones(len(scores),dtype=bool)
    if threshold == 1.:
        return np.zeros(len(scores),dtype=bool)
    return np.asarray(scores) >= threshold


def sample_joint(membership):
    """Independent membership-cardinality sufficient statistics and GFM search."""
    Y = np.asarray(membership,dtype=bool)
    size = Y.sum(1)
    joint = np.column_stack([(Y*(size[:,None] == s)).mean(0) for s in range(Y.shape[1]+1)])
    probabilities = np.bincount(size,minlength=Y.shape[1]+1)/len(Y)
    best_utility, best_mask = float(probabilities[0]), np.zeros(Y.shape[1],dtype=bool)
    for k in range(1,Y.shape[1]+1):
        utility = (joint[:,1:]* (2/(k+np.arange(1,Y.shape[1]+1)))).sum(1)
        # Original established GFM implementation breaks ties by lower index.
        order = np.argsort(-utility,kind="stable")
        total = float(utility[order[:k]].sum())
        # Preserve the declared earlier-cardinality tie policy under roundoff.
        roundoff = 8*np.finfo(np.float64).eps*max(1.,abs(total),abs(best_utility))
        if total > best_utility + roundoff:
            best_utility = total
            best_mask[:] = False
            best_mask[order[:k]] = True
    return Y.mean(0), joint, probabilities, best_mask, best_utility


def compute_metrics(y, decision, scores=None, q=None, fullA=None):
    y, decision = np.asarray(y,dtype=bool), np.asarray(decision,dtype=bool)
    intersection, selected, target = int(np.count_nonzero(y & decision)), int(decision.sum()), int(y.sum())
    result = {"f1":2*intersection/(selected+target) if selected+target else 1.,
              "precision":intersection/selected if selected else float(target == 0),
              "recall":intersection/target if target else 1., "exact_set":int(np.array_equal(y,decision)),
              "predicted_cardinality":selected, "true_cardinality":target,
              "cardinality_error":selected-target,"absolute_cardinality_error":abs(selected-target),
              "exact_cardinality":int(selected == target)}
    if scores is not None:
        squared = np.square(np.asarray(scores)-y)
        # Saved native-head scores are float32. Preserve the recorded reduction
        # and division precision instead of NumPy 1.26's scalar / int promotion.
        numerator = np.add.reduce(squared, dtype=squared.dtype)
        denominator = np.asarray(len(y), dtype=squared.dtype)
        result["uc_brier"] = float(np.divide(numerator, denominator))
    if q is not None:
        i,j = np.triu_indices(len(q),k=1)
        values, truth = q[i,j], fullA[i,j].astype(float)
        result["orientation_brier"] = float(np.square(values-truth).sum()/len(values))
        ece = 0.
        # Follow the saved reporting bin boundaries, independently computed.
        for low in np.arange(0.,1.,.1):
            mask = (values >= low) & ((values < low+.1) if low < .9 else (values <= 1.))
            if mask.any():
                ece += mask.sum()/len(values)*abs(float(values[mask].mean()-truth[mask].mean()))
        result["orientation_ece_10bins"] = float(ece)
    return result


def aggregate(rows, fraction=.1, cohort="all"):
    """Explicit three-level averaging, independent of original summaries."""
    groups = collections.defaultdict(list)
    for row in rows:
        if abs(row["fraction"]-fraction) > 1e-12:
            continue
        if cohort == "singleton" and row["true_cardinality"] != 1:
            continue
        if cohort == "selective_non_singleton" and row["true_cardinality"] == 1:
            continue
        groups[(row["method"],row["readout"],row["source"],row["profile"])].append(row)
    profile = []
    for (method,readout,source,name), values in sorted(groups.items()):
        record = {"method":method,"readout":readout,"source":source,"profile":name,
                  "cohort":"singleton" if values[0]["true_cardinality"] == 1 else "selective_non_singleton"}
        record.update({metric:float(np.mean([v[metric] for v in values if metric in v]))
                       for metric in METRICS if any(metric in v for v in values)})
        profile.append(record)
    groups = collections.defaultdict(list)
    for row in profile:
        groups[(row["method"],row["readout"],row["source"])].append(row)
    source = []
    for (method,readout,name),values in sorted(groups.items()):
        record = {"method":method,"readout":readout,"source":name}
        record.update({metric:float(np.mean([v[metric] for v in values if metric in v]))
                       for metric in METRICS if any(metric in v for v in values)})
        source.append(record)
    groups = collections.defaultdict(list)
    for row in source:
        groups[(row["method"],row["readout"])].append(row)
    summary = []
    for (method,readout),values in sorted(groups.items()):
        record = {"method":method,"readout":readout,"fraction":fraction,"cohort":cohort,"sources":len(values)}
        record.update({metric:float(np.mean([v[metric] for v in values if metric in v]))
                       for metric in METRICS if any(metric in v for v in values)})
        summary.append(record)
    return profile,source,summary


def comparison(saved_rows, rebuilt_rows, keys, context):
    saved = {tuple(row[k] for k in keys):row for row in saved_rows}
    require(len(saved) == len(saved_rows) == len(rebuilt_rows), context+" coverage mismatch")
    max_difference = 0.
    for row in rebuilt_rows:
        key = tuple(row[k] for k in keys)
        require(key in saved, context+" missing row "+str(key))
        for metric in METRICS:
            if metric in row:
                max_difference = max(max_difference,check_close(float(saved[key][metric]),row[metric],context+" "+str(key)+" "+metric))
    return max_difference


def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()


def verify_archive(path,out,full):
    sidecar=path.with_suffix(path.suffix+".sha256")
    require(sidecar.exists(),"Missing archive sidecar "+str(path))
    expected=sidecar.read_text().split()[0]
    require(digest(path)==expected,"Archive SHA-256 mismatch "+str(path))
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None,"Archive CRC mismatch")
        names=archive.namelist()
        require(len(names)==len(set(names)),"Duplicate ZIP entries")
        manifest=json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        hashes=manifest["payload_sha256"]
        require(set(names)==set(hashes)|{"ARCHIVE_MANIFEST.json"},"ZIP manifest coverage mismatch")
        for name,wanted in hashes.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,"ZIP unsafe path")
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1<<20),b""):
                    h.update(block)
            require(h.hexdigest()==wanted,"ZIP payload mismatch "+name)
        if full:
            for local in out.rglob("*"):
                if not local.is_file() or local.name in {"RUNNING.lock","run.log"} or local.suffix in {".zip",".sha256"}:
                    continue
                name="results/"+local.relative_to(out).as_posix()
                require(name in hashes and hashes[name]==digest(local),"FULL archive differs from original "+name)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":expected,"payload_files":len(hashes),"full_original_results_match":bool(full)}


def certificate_rebuild(W,delta):
    n=len(W);lower=np.zeros((n,n));upper=np.ones((n,n));np.fill_diagonal(lower,.5);np.fill_diagonal(upper,.5)
    for i in range(n):
        for j in range(i+1,n):
            m=W[i,j]+W[j,i]
            if not m:
                continue
            radius=math.sqrt(math.log(n*(n-1)/delta)/(2*m))
            lo=max(0.,W[i,j]/m-radius);hi=min(1.,W[i,j]/m+radius)
            lower[i,j],upper[i,j]=lo,hi;lower[j,i],upper[j,i]=1-hi,1-lo
    known=lower>.5;np.fill_diagonal(known,False)
    paths=known.astype(np.int64)@known.astype(np.int64)
    reachable=known|(paths>0);np.fill_diagonal(reachable,True)
    inner=reachable.all(1)
    outer=np.ones(n,dtype=bool)
    for v in range(n):
        for u in range(n):
            if u!=v and known[u,v] and all(known[w,v] or known[u,w] for w in range(n) if w not in (u,v)):
                outer[v]=False;break
    return lower,upper,known,inner,outer


def native(rows):
    return [r for r in rows if r["readout"]==NATIVE[r["method"]]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--archives",type=Path,nargs="*")
    parser.add_argument("--replay-device",choices=("cpu","cuda"),default="cuda",
                        help="Selected-state replay device; strict absolute 1e-5 gates remain unchanged")
    parser.add_argument("--skip-forward-replay",action="store_true",help="Produces partial audit only; never full verification")
    args=parser.parse_args();start=time.perf_counter()
    out,source,report=args.out.resolve(),args.source.resolve(),args.report.resolve()
    require(not report.exists(),"Report already exists; choose a new external report")
    require(not report.is_relative_to(out) and not report.is_relative_to(source),"Report must be outside immutable source/results")
    complete,audit,fit,lock=jread(out/"COMPLETE.json"),jread(out/"AUDIT.json"),jread(out/"FIT_COMPLETE.json"),jread(out/"RUN_LOCK.json")
    cfg=lock["config"]
    require(complete["status"]=="COMPLETE" and audit["status"]=="AUDIT_COMPLETE","Study not completed/audited")
    require(fit["status"]=="FIT_COMPLETE_AWAITING_AUDIT","Unexpected fitting marker")
    require(complete["AUDIT_sha256"]==digest(out/"AUDIT.json") and complete["FIT_COMPLETE_sha256"]==digest(out/"FIT_COMPLETE.json"),"Audit chain marker hash mismatch")
    for key,value in audit.items():
        if key!="status":
            require(complete[key]==value,"Completion marker disagreement "+key)
    for key in ("coverage_verified","source_hashes_verified","data_hashes_verified","checkpoint_hashes_verified","selected_replay_verified"):
        require(complete.get(key) is True,"Missing prior audit gate "+key)
    actual={p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name not in EXCLUDED}
    require(actual==set(fit["hashes"]),"Result file list differs from hashed payload")
    for name,wanted in fit["hashes"].items():
        require(digest(out/name)==wanted,"Output hash mismatch "+name)
    for name,wanted in lock["source_sha256"].items():
        require(digest(source/name)==wanted,"Frozen source mismatch "+name)
    require(jread(out/"CONFIG.json")==cfg,"Config differs from run lock")
    mode=cfg["mode"];production=mode=="production"
    if production:
        require(cfg["seed"]==20261007 and cfg["learning"]["initializations"]==3 and cfg["data"]["resamples"]==8,"Fixed production protocol mismatch")
        require(cfg["posterior"]["test_draws"]==4096 and cfg["data"]["fractions"]==[.05,.1,.25],"Production sampling/MC mismatch")
    manifest=jread(out/"data/DATA_PROFILE_MANIFEST.json")
    samples=cread(out/"data/DATA_COUNTS_MANIFEST.csv")
    cases={r["case_id"]:r for r in samples}
    require(len(cases)==len(samples)==manifest["case_count"],"Duplicate/count-mismatch data case")
    require(len(manifest["profiles"])==36,"Original 36-profile cohort absent")
    data,raw_count_checks={},0
    for profile in manifest["profiles"]:
        name=profile["profile"];path=source/"reference/data/profiles"/name
        require(digest(path)==profile["raw_sha256"],"Original raw profile hash mismatch")
        ranks,mult,ids=parse_raw(path)
        require(digest(out/"data"/profile["counts_npz"])==profile["counts_npz_sha256"],"Counts NPZ hash mismatch")
        arr=aread(out/"data"/profile["counts_npz"])
        require(np.array_equal(arr["ranks"],ranks) and np.array_equal(arr["original_multiplicity"],mult) and np.array_equal(arr["alternative_ids"],ids),"Saved ranks/multiplicities differ from raw ballots")
        W,T=raw_counts(ranks,mult);require(np.array_equal(W,arr["fullW"]) and np.array_equal(T,arr["fullT"]),"Full counts differ from raw ballots")
        require(np.array_equal(W>W.T,arr["fullA"]) and np.array_equal(uc_cover(W>W.T),arr["y"]),"Independent full-reference UC mismatch")
        cells=[r for r in samples if r["profile"]==name]
        require({(float(r["fraction"]),int(r["replicate"])) for r in cells}=={(float(f),r) for f in cfg["data"]["fractions"] for r in range(cfg["data"]["resamples"])},"Per-profile fraction/resample coverage mismatch")
        for cell in cells:
            index=int(cell["array_index"]);used=arr["sampled_multiplicity"][index]
            require(np.all(used>=0) and np.all(used<=mult),"Sample not subset of actual finite ballots")
            W,T=raw_counts(ranks,used)
            require(np.array_equal(W,arr["W"][index]) and np.array_equal(T,arr["T"][index]),"Observed counts differ from saved sampled ballots")
            require(used.sum()==int(cell["voters_sampled"])==max(1,int(np.floor(float(cell["fraction"])*mult.sum()))),"Sample size/fraction mismatch")
            raw_count_checks+=1
        for replicate in range(cfg["data"]["resamples"]):
            ordered=sorted((r for r in cells if int(r["replicate"])==replicate),key=lambda r:float(r["fraction"]))
            for left,right in zip(ordered,ordered[1:]):
                require(np.all(arr["sampled_multiplicity"][int(left["array_index"])]<=arr["sampled_multiplicity"][int(right["array_index"])]),"Fractions not nested within resample")
        data[name]=arr
    require({p["profile"] for p in manifest["profiles"] if int(p["uc_size"])>1}==SELECTIVE,"Selective-profile identity differs")
    folds=jread(out/"FOLDS.json")
    require(len(folds)==(2 if mode=="smoke" else 6),"Whole-source fold coverage mismatch")
    for index,fold in enumerate(folds):
        require(fold["test_source"]==SOURCES[index] and fold.get("dev_source",fold.get("development_source"))==SOURCES[(index+1)%6],"Frozen source order/dev split mismatch")
        require(set(fold["train_sources"])==set(SOURCES)-{SOURCES[index],SOURCES[(index+1)%6]},"Training-source isolation mismatch")
    prediction_index=jread(out/"PREDICTION_INDEX.json")
    expected={(i,init,cid,method) for i,fold in enumerate(folds) for init in range(cfg["learning"]["initializations"])
              for cid,case in cases.items() if case["source"]==fold["test_source"] for method in METHODS}
    indexed={(int(r["fold"]),int(r["initialization"]),r["case_id"],r["method"]) for r in prediction_index}
    require(len(indexed)==len(prediction_index) and indexed==expected,"Exact prediction cross-product coverage mismatch")
    saved_metrics=cread(out/"per_case_metrics.csv")
    saved={ (int(r["fold"]),int(r["initialization"]),r["case_id"],r["method"],r["readout"]):r for r in saved_metrics}
    require(len(saved)==len(saved_metrics),"Duplicate metric row")
    choices,models,checkpoints,proofs={},{},{},[]
    replay_count=0;forward_max=0.;metric_max=0.;rows=[];readout_keys=set();joint_count=0
    forward_max_by_output={};forward_device_evidence={}
    invalid_secondary=[];posthoc_diagonal_rows=[];posthoc_sampling_law_rows=[]
    q_range_excursions=[]
    if not args.skip_forward_replay:
        sys.path.insert(0,str(source))
        import torch
        from stepc.learning import PredictiveModel,FEATURE_VERSION
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if args.replay_device=="cuda":
            require(torch.cuda.is_available(),"CUDA requested for selected-state replay but unavailable")
            replay_device=torch.device("cuda:0")
            require(torch.cuda.get_device_name(replay_device)==lock["runtime"]["gpu"],"Replay GPU differs from frozen run device")
        else:
            replay_device=torch.device("cpu")
        forward_device_evidence={"requested":args.replay_device,"actual":str(replay_device),
            "gpu":torch.cuda.get_device_name(replay_device) if replay_device.type=="cuda" else None,
            "torch":str(torch.__version__),"cuda":torch.version.cuda,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"absolute_tolerance_all_outputs":1e-5}
    for number,record in enumerate(prediction_index):
        f,init,cid,method=int(record["fold"]),int(record["initialization"]),record["case_id"],record["method"]
        context=f"record={number} fold={f} init={init} case={cid} method={method}"
        meta=cases[cid];arr=data[meta["profile"]];at=int(meta["array_index"]);raw=aread(out/record["path"])
        require(np.array_equal(raw["W"],arr["W"][at]) and np.array_equal(raw["T"],arr["T"][at]),"Raw prediction observed input mismatch")
        require(np.array_equal(raw["y"],arr["y"]) and np.array_equal(raw["fullA"],arr["fullA"]),"Prediction/reference mismatch")
        choicepath=record["choice_path"]
        if choicepath not in choices:
            selected=jread(out/choicepath);choices[choicepath]=selected["choices"]
            require(selected["selection_sources"]==[SOURCES[(f+1)%6]] and set(selected["training_sources"])==set(folds[f]["train_sources"]),"Saved choices violate source isolation")
            require(selected["test_source"]==SOURCES[f] and selected["selection_fraction"]==.1 and selected["test_predictions_not_started"],"Selection not frozen before testing")
            for kind in FITTED:
                choice=selected["choices"][kind]
                require(digest(out/choice["checkpoint"])==choice["checkpoint_sha256"],"Selected checkpoint hash mismatch")
                candidates=jread((out/choicepath).parent/kind/"DEVELOPMENT_CANDIDATES.json")
                require(choice==max(candidates,key=lambda c:(c["development_f1"],-c.get("snapshot",0),-c["weight"])),"Checkpoint not deterministic development-selected candidate")
                for proof in jread((out/choicepath).parent/kind/"DEVICE_AND_GRADIENT_PROOF.json"):
                    proofs.append({"fold":f,"initialization":init,"method":kind,**proof})
        require((out/record["path"]).stat().st_mtime_ns >= (out/choicepath).stat().st_mtime_ns,"Prediction filesystem chronology precedes frozen choices")
        choice=choices[choicepath].get(method)
        scores,q=raw.get("scores"),raw.get("q")
        decisions={}
        if method in FITTED:
            modelkey=(choicepath,method)
            if modelkey not in checkpoints:
                checkpoints[modelkey]={"checkpoint":choice["checkpoint"],"sha256":choice["checkpoint_sha256"],"step":choice.get("step",0),"weight":choice["weight"]}
            if method=="empirical_bayes":
                alpha=jread(out/choice["checkpoint"])["alpha"]
                check_close(raw["q"],beta_orientation(raw["W"],alpha),"Independent empirical-Bayes orientation replay",1e-12)
                replay_count+=1
            elif not args.skip_forward_replay:
                if modelkey not in models:
                    state=torch.load(out/choice["checkpoint"],map_location="cpu",weights_only=True)
                    require(state["feature_version"]==FEATURE_VERSION and state["kind"]==method,"Checkpoint model/feature identity mismatch")
                    model=PredictiveModel(state["kind"],state["hidden"]).to(replay_device);model.load_state_dict(state["model"]);model.eval()
                    require(state_hash(model)==state["state_hash"],"Independent tensor state-hash mismatch")
                    models[modelkey]=model
                with torch.no_grad():
                    replay=models[modelkey](torch.tensor(raw["W"],dtype=torch.float32,device=replay_device),torch.tensor(raw["T"],dtype=torch.float32,device=replay_device))
                for name,value in replay.items():
                    require(name in raw,"Saved prediction missing forward output "+name)
                    difference=check_close(value.detach().cpu().numpy(),raw[name],context+" selected "+str(replay_device)+" forward "+name,1e-5)
                    forward_max=max(forward_max,difference)
                    forward_max_by_output[name]=max(forward_max_by_output.get(name,0.),difference)
                replay_count+=1
        elif method=="count_jeffreys":
            check_close(raw["q"],beta_orientation(raw["W"],.5),"Independent Jeffreys replay",1e-12)
        elif method=="count_raw":
            require(np.array_equal(raw["adjacency"],raw["W"]>raw["W"].T),"Raw majority graph mismatch")
            decisions["raw_majority_uc"]=uc_cover(raw["adjacency"])
        elif method=="count_soft_plugin":
            P,scores=plugin_score(raw["W"])
            check_close(P,raw["P"],"Independent plug-in mean",1e-12);check_close(scores,raw["scores"],"Independent plug-in score",1e-12)
            decisions["dev_calibrated_secondary"]=threshold_decision(scores,choice["threshold"])
        if q is not None:
            adjacency=np.asarray(q)>.5
            self_loops=np.flatnonzero(np.diag(adjacency))
            bidirectional=np.argwhere(np.triu(adjacency & adjacency.T,k=1))
            outside=np.argwhere((q<0)|(q>1))
            require(np.isfinite(q).all(),context+" nonfinite raw orientation marginal")
            if len(outside):
                q_range_excursions.append({"fold":f,"initialization":init,"case_id":cid,"method":method,
                    "count":len(outside),"minimum":float(q.min()),"maximum":float(q.max()),
                    "scope":"raw marginal rounding range excursion; native draws use valid component probabilities with normalized weights"})
            require(raw["memberships"].shape==(cfg["posterior"]["test_draws"],len(q)),"Joint draw coverage mismatch")
            require(record["draws"]==cfg["posterior"]["test_draws"] and record["draw_seed"]==int(raw["draw_seed"]),"Recorded seed/draw count mismatch")
            check_close(q,(raw["weights"][:,None,None]*raw["components"]).sum(0),"Global mixture orientation mean",1e-6)
            scores,joint,cards,gfm,utility=sample_joint(raw["memberships"])
            check_close(scores,raw["marginals"],"Saved joint membership marginal")
            check_close(joint,raw["membership_size_joint"],"Saved membership/cardinality joint")
            check_close(cards,raw["cardinality_probabilities"],"Saved cardinality law")
            check_close(utility,float(raw["gfm_expected_f1"]),"Independent empirical GFM utility")
            # Keep the actual recorded secondary mask and its metrics intact.
            # Source generalized-cover arithmetic allows self-cover; verifying
            # that recorded calculation does NOT validate its tournament meaning.
            recorded_hard=recorded_general_cover(adjacency)
            decisions={"gfm":gfm,"half":scores>=.5,"hard_q_diagnostic":recorded_hard}
            require(np.isfinite(raw["components"]).all() and np.all((raw["components"]>=0)&(raw["components"]<=1)),
                    context+" invalid component probability in native sampling law")
            require(np.isfinite(raw["weights"]).all() and np.all(raw["weights"]>=0) and raw["weights"].sum()>0,
                    context+" invalid weights in native sampling law")
            if len(self_loops) or len(bidirectional) or len(outside):
                corrected_adjacency=adjacency.copy();np.fill_diagonal(corrected_adjacency,False)
                diagonal_corrected=uc_cover(corrected_adjacency)
                # A second, separately labelled post hoc reconstruction uses the
                # actual sampled law: normalized weights, upper-edge components,
                # exact reciprocal lower edges, and an explicitly zero diagonal.
                weights=np.asarray(raw["weights"],dtype=np.float64)
                weights=weights/weights.sum()
                marginal=(np.asarray(raw["components"],dtype=np.float64)*weights[:,None,None]).sum(0)
                law_adjacency=np.zeros(q.shape,dtype=bool)
                ai,aj=np.triu_indices(len(q),k=1)
                law_adjacency[ai,aj]=marginal[ai,aj]>.5
                law_adjacency[aj,ai]=marginal[ai,aj]<.5
                law_corrected=uc_cover(law_adjacency)
                invalid_secondary.append({"fold":f,"initialization":init,"case_id":cid,"method":method,
                    "source":meta["source"],"profile":meta["profile"],"fraction":float(meta["fraction"]),
                    "self_loop_vertices":self_loops.tolist(),"bidirectional_pairs":bidirectional.tolist(),
                    "raw_probability_outside_unit_interval":outside.tolist(),
                    "weight_sum_float64":float(np.asarray(raw["weights"],dtype=np.float64).sum()),
                    "q_diagonal":np.diag(q).tolist(),"q_minimum":float(q.min()),"q_maximum":float(q.max()),
                    "recorded_decision_vertices":np.flatnonzero(recorded_hard).tolist(),
                    "posthoc_diagonal_cleared_vertices":np.flatnonzero(diagonal_corrected).tolist(),
                    "posthoc_diagonal_cleared_still_bidirectional":bool(len(bidirectional)),
                    "posthoc_sampling_law_vertices":np.flatnonzero(law_corrected).tolist(),
                    "recorded_differs_from_diagonal_cleared":bool(np.any(recorded_hard!=diagonal_corrected)),
                    "recorded_differs_from_sampling_law":bool(np.any(recorded_hard!=law_corrected)),
                    "validity":"INVALID_SECONDARY_DIAGNOSTIC: intended irreflexive/probability relation contract violated; archived calculation and bytes preserved"})
                for correction,mask,container in (("posthoc_hard_q_diagonal_cleared",diagonal_corrected,posthoc_diagonal_rows),
                                                  ("posthoc_hard_q_normalized_sampling_law",law_corrected,posthoc_sampling_law_rows)):
                    container.append({"fold":f,"initialization":init,"case_id":cid,"method":method,"readout":correction,
                        "source":meta["source"],"profile":meta["profile"],"fraction":float(meta["fraction"]),
                        "replicate":int(meta["replicate"]),"n":int(meta["n"]),
                        **compute_metrics(raw["y"],mask,scores,q,raw["fullA"]),
                        "scope":"POST_HOC_SECONDARY_DIAGNOSTIC_ONLY; affected cases only; no replacement/promotion of native endpoint"})
            joint_count+=1
        elif method in ("ordinary","relational"):
            decisions["dev_calibrated_direct"]=threshold_decision(scores,choice["threshold"])
        require({k for k in raw if k.startswith("decision_")}=={"decision_"+r for r in decisions},"Saved decoder coverage mismatch")
        for readout,decision in decisions.items():
            require(np.array_equal(decision,raw["decision_"+readout]),context+" readout="+readout+" independent decoder mismatch")
            key=(f,init,cid,method,readout);require(key in saved,"Saved metric missing "+str(key));readout_keys.add(key)
            computed=compute_metrics(raw["y"],decision,scores,q,raw["fullA"])
            computed["inference_seconds"]=float(record["inference_seconds"])
            require(math.isfinite(computed["inference_seconds"]) and computed["inference_seconds"]>=0,"Invalid inference timing")
            for name,value in computed.items():
                metric_max=max(metric_max,check_close(float(saved[key][name]),value,context+" readout="+readout+" independent per-case "+name))
            rows.append({"fold":f,"initialization":init,"case_id":cid,"method":method,"readout":readout,
                         "source":meta["source"],"profile":meta["profile"],"fraction":float(meta["fraction"]),
                         "replicate":int(meta["replicate"]),"n":int(meta["n"]),**computed})
        if (number+1)%1000==0:
            print(json.dumps({"external_review_predictions":number+1,"total":len(prediction_index),"selected_forward_replays":replay_count,"replay_device":args.replay_device}),flush=True)
    require(readout_keys==set(saved),"Metric keys differ from exact rederived decoder coverage")
    for proof in proofs:
        if proof["method"]=="empirical_bayes":
            require(proof["finite"] and proof["early_native_convergence"],"Empirical-Bayes convergence proof absent")
        else:
            require(proof["finite_gradients"] and proof["parameters_changed"] and proof["steps"]>0 and proof["initial_state_hash"]!=proof["final_state_hash"],"Gradient updates/parameter-change evidence invalid")
            if mode in ("production","pilot"):
                require(proof["model_devices"]==["cuda:0"] and proof["optimizer_devices"]==["cuda:0"] and proof["training_feature_device"]=="cuda:0" and proof["training_label_device"]=="cuda:0","Fitted tensors/optimizer not CUDA")
    profile,source_rows,main_summary=aggregate(rows)
    aggregate_max=max(comparison(cread(out/"per_profile_10pct.csv"),profile,("method","readout","source","profile"),"Profile summaries"),
                      comparison(cread(out/"per_source_10pct.csv"),source_rows,("method","readout","source"),"Source summaries"),
                      comparison(cread(out/"descriptive_summary_10pct.csv"),main_summary,("method","readout"),"Main source-macro summaries"))
    secondary=[];cohorts=[]
    for fraction in (.05,.25):
        _,_,summary=aggregate(rows,fraction=fraction)
        aggregate_max=max(aggregate_max,comparison(cread(out/f"descriptive_summary_{int(100*fraction)}pct.csv"),summary,("method","readout"),"Fraction summaries"))
        secondary.extend(native(summary))
    for cohort in ("singleton","selective_non_singleton"):
        _,_,summary=aggregate(rows,cohort=cohort)
        aggregate_max=max(aggregate_max,comparison(cread(out/f"descriptive_summary_10pct_{cohort}.csv"),summary,("method","readout"),"Cohort summaries"))
        cohorts.extend(native(summary))
    certificate_rows=cread(out/"selective_metrics.csv")
    expected_cert={(f,i,cid,float(delta)) for f,i,cid,_ in expected for delta in cfg["certificates"]["deltas"]}
    cert_keys=set();cert_values=[];certificate_max=0.
    for row in certificate_rows:
        f,i,cid,delta=int(row["fold"]),int(row["initialization"]),row["case_id"],float(row["delta"])
        key=(f,i,cid,delta);require(key not in cert_keys,"Duplicate certificate row");cert_keys.add(key)
        meta=cases[cid];arr=data[meta["profile"]];at=int(meta["array_index"]);W,y=arr["W"][at],arr["y"]
        lo,hi,known,inner,outer=certificate_rebuild(W,delta)
        path=out/f"fold_{f:02d}"/f"init_{i:02d}"/"predictions"/cid/f"certificate_delta_{delta:g}.npz"
        raw=aread(path)
        for name,value in (("lower",lo),("upper",hi),("known",known),("inner",inner),("outer",outer)):
            certificate_max=max(certificate_max,check_close(raw[name],value,"Independent certificate "+name,1e-12))
        decided=inner|~outer
        rebuilt={"certified_fraction":float(decided.mean()),"inner_cardinality":int(inner.sum()),"outer_cardinality":int(outer.sum()),
                 "abstention_fraction":float((~decided).mean()),"false_inclusions":int((inner&~y).sum()),"false_exclusions":int((~outer&y).sum()),
                 "selective_accuracy":float((inner[decided]==y[decided]).mean()) if decided.any() else None}
        require(row["coverage_guarantee"]=="False","Invalid frequentist human-profile guarantee")
        for name,value in rebuilt.items():
            if value is None:
                require(row[name]=="","Undecided certificate accuracy must be blank")
            else:
                certificate_max=max(certificate_max,check_close(float(row[name]),value,"Certificate metric "+name))
        cert_values.append({"source":meta["source"],"profile":meta["profile"],"fraction":float(meta["fraction"]),"delta":delta,**rebuilt})
    require(cert_keys==expected_cert,"Certificate cross-product coverage mismatch")
    timing=jread(out/"TIMING.json")
    budget_valid=all(t["charged_seconds"]<=t["allocated_seconds"]*1.02 for t in timing["methods"])
    require(budget_valid==timing["equal_allocation_valid"]==fit["equal_allocation_valid"]==complete["EQUAL_ALLOCATION_VALID"],"Budget validity classification mismatch")
    require(len(timing["methods"])==len(folds)*cfg["learning"]["initializations"]*5,"Fit timing coverage mismatch")
    timing_summary=[]
    for method in FITTED:
        clocks=[t for t in timing["methods"] if t["method"]==method]
        evidence=[p for p in proofs if p["method"]==method]
        timing_summary.append({"method":method,"fits":len(clocks),"allocation_seconds_total":sum(t["allocated_seconds"] for t in clocks),
                               "charged_seconds_total":sum(t["charged_seconds"] for t in clocks),
                               "charged_seconds_min":min(t["charged_seconds"] for t in clocks),"charged_seconds_max":max(t["charged_seconds"] for t in clocks),
                               "gradient_candidates":len(evidence) if method!="empirical_bayes" else 0,
                               "optimizer_steps_total":sum(p.get("steps",0) for p in evidence),
                               "selected_steps_min":min(c["step"] for (cp,m),c in checkpoints.items() if m==method),
                               "selected_steps_max":max(c["step"] for (cp,m),c in checkpoints.items() if m==method)})
    inference_costs=[]
    for method in METHODS:
        records=[r for r in prediction_index if r["method"]==method]
        values=np.array([r["inference_seconds"] for r in records],dtype=float)
        inference_costs.append({"method":method,"test_predictions":len(records),"inference_seconds_total":float(values.sum()),
                                "seconds_mean":float(values.mean()),"seconds_max":float(values.max()),"draws_per_posterior_test":cfg["posterior"]["test_draws"] if method in ("empirical_bayes","count_jeffreys","learned_edge","posterior_mixture") else 0})
    native_sources=native(source_rows);lookup={(r["method"],r["source"]):r for r in native_sources}
    contrasts=[]
    for method in ("posterior_mixture","learned_edge","ordinary","relational"):
        for baseline in ("count_raw","count_jeffreys","empirical_bayes"):
            differences=[{"source":s,"f1_difference":lookup[(method,s)]["f1"]-lookup[(baseline,s)]["f1"],
                          "exact_set_difference":lookup[(method,s)]["exact_set"]-lookup[(baseline,s)]["exact_set"]}
                         for s in SOURCES if (method,s) in lookup]
            contrasts.append({"method":method,"baseline":baseline,"source_macro_f1_difference":float(np.mean([d["f1_difference"] for d in differences])),
                              "sources_with_positive_difference":sum(d["f1_difference"]>0 for d in differences),"source_differences":differences,
                              "scope":"descriptive contrasts only; no p-values or population/superiority inference"})
    archives=args.archives if args.archives is not None else sorted(out.parent.glob("STE_Posterior_Certificates_v1_"+out.name+"_*.zip"))
    require(bool(archives),"No completed study archives available for external verification")
    archive_records=[verify_archive(p.resolve(),out,p.name.endswith("_FULL.zip")) for p in archives]
    require(any(p.name.endswith("_FULL.zip") for p in archives) and any(p.name.endswith("_REVIEW.zip") for p in archives),"FULL/REVIEW archive pair absent")
    certificate_summary=[]
    for delta in cfg["certificates"]["deltas"]:
        for fraction in cfg["data"]["fractions"]:
            chosen=[r for r in cert_values if r["delta"]==delta and r["fraction"]==fraction]
            grouped=collections.defaultdict(list)
            for r in chosen:
                grouped[(r["source"],r["profile"])].append(r)
            pg=[]
            for (s,p),rs in grouped.items():
                pg.append({"source":s,"profile":p,**{k:float(np.mean([v[k] for v in rs])) for k in ("certified_fraction","abstention_fraction","inner_cardinality","outer_cardinality","false_inclusions","false_exclusions")}})
            sg=[]
            for s in dict.fromkeys(r["source"] for r in pg):
                rs=[r for r in pg if r["source"]==s]
                sg.append({k:float(np.mean([v[k] for v in rs])) for k in ("certified_fraction","abstention_fraction","inner_cardinality","outer_cardinality","false_inclusions","false_exclusions")})
            certificate_summary.append({"delta":delta,"fraction":fraction,"scope":"finite-cohort diagnostic, no coverage guarantee",
                                        **{k:float(np.mean([v[k] for v in sg])) for k in sg[0]}})
    expected_manifest=jread(out/"EXPECTED_COVERAGE.json")
    require(expected_manifest["prediction_files"]==len(prediction_index) and expected_manifest["metric_rows"]==len(rows) and expected_manifest["selective_rows"]==len(certificate_rows),"Original expected coverage count disagreement")
    review_status=("PRIMARY_REVIEW_COMPLETE_WITH_INVALID_SECONDARY_DIAGNOSTIC" if invalid_secondary else "INDEPENDENT_REVIEW_COMPLETE")
    if args.skip_forward_replay:
        review_status="PARTIAL_REVIEW_NO_NEURAL_REPLAY"
    invalid_counts=[]
    for method in METHODS:
        affected=[r for r in invalid_secondary if r["method"]==method]
        if affected:
            invalid_counts.append({"method":method,"affected_cases":len(affected),
                "self_loop_cases":sum(bool(r["self_loop_vertices"]) for r in affected),
                "bidirectional_cases":sum(bool(r["bidirectional_pairs"]) for r in affected),
                "probability_range_excursion_cases":sum(bool(r["raw_probability_outside_unit_interval"]) for r in affected),
                "recorded_mask_changed_by_diagonal_clear":sum(r["recorded_differs_from_diagonal_cleared"] for r in affected),
                "recorded_mask_changed_by_sampling_law":sum(r["recorded_differs_from_sampling_law"] for r in affected)})
    invalid_methods={r["method"] for r in invalid_secondary}
    for summary in main_summary:
        if summary["readout"]=="hard_q_diagnostic":
            summary["endpoint_validity"]=("INVALID_SECONDARY_DIAGNOSTIC" if summary["method"] in invalid_methods else "valid_irreflexive_secondary_relation")
            summary["scope"]="Archived arithmetic reproduced; no affected diagnostic is promoted as a primary/native result."
    results={"status":review_status,
             "classification":complete["classification"],"mode":mode,"results_root":str(out),"immutable_source":str(source),
             "review_script_sha256":digest(Path(__file__)),"seconds":time.perf_counter()-start,
             "runtime":lock["runtime"],"coverage":{"profiles":36,"singleton_profiles":33,"selective_profiles":3,"sources":len(folds),
             "initializations":cfg["learning"]["initializations"],"resamples":cfg["data"]["resamples"],"fractions":cfg["data"]["fractions"],
             "data_cases":len(cases),"raw_ballot_count_reconstructions":raw_count_checks,"prediction_files":len(prediction_index),
             "metric_rows":len(rows),"certificate_rows":len(certificate_rows),"joint_statistics_reconstructions":joint_count,
             "selected_forward_replays":replay_count,"selected_checkpoint_files":len(checkpoints)},
             "verification":{"audit_chain":True,"exact_payload_filelist":True,"result_hashes":len(fit["hashes"]),"source_hashes":len(lock["source_sha256"]),
             "raw_ballot_counts_and_truth":True,"exact_fold_method_case_init_coverage":True,"selected_checkpoint_hashes":True,
             "native_decoder_and_joint_reconstruction":True,"recorded_secondary_arithmetic_reproduced":True,
             "all_secondary_diagnostics_structurally_valid":not bool(invalid_secondary),
             "selected_neural_forward_replay":not args.skip_forward_replay,
             "cpu_neural_selected_replay":not args.skip_forward_replay and args.replay_device=="cpu",
             "cuda_neural_selected_replay":not args.skip_forward_replay and args.replay_device=="cuda",
             "per_case_metrics_max_abs_error":metric_max,"profile_source_summary_max_abs_error":aggregate_max,
             "selected_forward_max_abs_error":forward_max,"forward_max_abs_error_by_output":forward_max_by_output,
             "selected_forward_device_evidence":forward_device_evidence,"certificate_max_abs_error":certificate_max,"equal_allocation_valid":budget_valid},
             "precision_note":"Membership Brier reductions/divisions retain the saved score dtype (float32 native heads, float64 sampled marginals); all metric comparisons retain 1e-10 tolerance. No result values or frozen code were changed.",
             "invalid_hard_q_secondary_diagnostic":{"affected_cases":len(invalid_secondary),"counts_by_method":invalid_counts,
                 "cases":invalid_secondary,"raw_q_range_excursion_cases":q_range_excursions,
                 "record_integrity":"Saved hard_q_diagnostic decisions/metrics verified using independent generalized cover including self-cover, solely to verify the actual recorded calculation. Those affected rows are scientifically invalid as intended irreflexive tournament readouts.",
                 "native_endpoint_scope":"Native GFM uses sampled strict tournaments with zero diagonal; half/direct/raw/plugin endpoints are checked separately. Invalid secondary rows do not replace or alter native results.",
                 "posthoc_scope":"Corrections below are external diagnostic reconstructions on affected cases only, not frozen outputs, not a new primary endpoint, not outcome-dependent promotion.",
                 "posthoc_diagonal_cleared_case_metrics":posthoc_diagonal_rows,
                 "posthoc_normalized_sampling_law_case_metrics":posthoc_sampling_law_rows},
             "main_native_10pct":native(main_summary),"main_all_readouts_10pct":main_summary,
             "native_sources_10pct":native_sources,"native_profiles_10pct":native(profile),
             "three_selective_profile_native_10pct":[r for r in native(profile) if r["profile"] in SELECTIVE],
             "singleton_selective_native_10pct":cohorts,"secondary_fraction_native":secondary,
             "descriptive_source_contrasts":contrasts,"training_time_and_updates":timing_summary,
             "held_out_inference_costs":inference_costs,"timing":{"fit_and_test_wall_seconds":timing["fit_and_test_wall_seconds"],
             "test_seconds":timing["test_seconds"],"data_preparation_seconds":timing["data_preparation_seconds"]},
             "certificate_descriptive_source_macro":certificate_summary,"archives":archive_records,
             "limits":["Previously observed cohort; all results exploratory/descriptive.","Six whole-source folds reuse sources in different roles.",
                       "33 of 36 references are singleton; only three selective profiles across two sources.",
                       "No p-values, general superiority or population/calibration guarantee.",
                       "Neural selected-state replay verifies saved inference on the explicitly recorded replay device, not independent retraining.",
                       "Preserved prior CPU replay attempt stopped on an unbounded learned-edge logit absolute difference of 1.0013580322265625e-5, just above the fixed 1e-5 gate. Current replay uses the requested device and does not relax that gate or claim a complete CPU replay when using CUDA.",
                       "Model architecture reused only for replay; metrics, summaries, raw ballot parser, certificates and archive validation independently implemented.",
                       "Original audit replayed every saved tournament draw; this external review independently reconstructs saved joint draws/statistics and decisions.",
                       "Affected hard_q_diagnostic rows violate the irreflexive relation contract through rounding-created self-loops/bidirectional edges; native endpoints are preserved and post hoc corrections are not promoted."]}
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("x") as stream:
        json.dump(results,stream,indent=2,allow_nan=False)
    print(json.dumps({"status":results["status"],"report":str(report),"seconds":results["seconds"],"coverage":results["coverage"],
                      "verification":results["verification"],"main_native_10pct":results["main_native_10pct"]}),flush=True)


if __name__=="__main__":
    main()
