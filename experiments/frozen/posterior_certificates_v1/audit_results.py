#!/usr/bin/env python3
"""Fail-closed archive audit, including selected-state CPU/GPU feature replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np

from stepc.data import sha256, parse_profile, ballot_counts
from stepc.posterior import exact_uc, jeffreys_orientation_probabilities, jeffreys_pair_means, soft_uc_plugin


def require(condition, message):
    if not condition:
        raise RuntimeError("Audit failed: " + message)


def read_csv(path):
    with Path(path).open(newline="") as h:
        return list(csv.DictReader(h))


def metric_key(row):
    return (int(row["fold"]),int(row["initialization"]),row["case_id"],row["method"],row["readout"])


def compare_metrics(saved, recomputed, context, tolerance=1e-10):
    for name,value in recomputed.items():
        require(name in saved,context+": missing metric "+name)
        if value is None:
            require(saved[name] in ("",None),context+": expected empty "+name)
        elif isinstance(value,(int,float,np.integer,np.floating)):
            require(abs(float(saved[name])-float(value))<=tolerance,context+": metric mismatch "+name)
        else:
            require(saved[name]==value,context+": field mismatch "+name)


def audit_data(out):
    """Check raw-profile hashes and every saved count against exact ballots."""
    from stepc.posterior import exact_uc
    data=out/"data"
    manifest=json.loads((data/"DATA_PROFILE_MANIFEST.json").read_text())
    samples=read_csv(data/"DATA_COUNTS_MANIFEST.csv")
    bycase={row["case_id"]:row for row in samples}
    require(len(bycase)==len(samples)==manifest["case_count"],"unique dataset case coverage")
    files={}
    for row in manifest["profiles"]:
        raw=Path(__file__).resolve().parent/"reference"/"data"/"profiles"/row["profile"]
        require(raw.is_file() and sha256(raw)==row["raw_sha256"],"raw profile hash "+row["profile"])
        parsed=parse_profile(raw)
        path=data/row["counts_npz"]
        require(path.is_file() and sha256(path)==row["counts_npz_sha256"],"profile count archive hash")
        arrays={k:v for k,v in np.load(path,allow_pickle=False).items()}
        W,T=ballot_counts(parsed,np.asarray(parsed["multiplicity"]))
        require(np.array_equal(W,arrays["fullW"]) and np.array_equal(T,arrays["fullT"]),"full counts match original ballots")
        require(np.array_equal(W>W.T,arrays["fullA"]) and np.array_equal(exact_uc(W>W.T),arrays["y"]),"reference majority/UC labels")
        for sample in (r for r in samples if r["profile"]==row["profile"]):
            i=int(sample["array_index"])
            mult=arrays["sampled_multiplicity"][i]
            require(np.all(mult>=0) and np.all(mult<=parsed["multiplicity"]),"finite-cohort sample multiplicities")
            sw,st=ballot_counts(parsed,mult)
            require(np.array_equal(sw,arrays["W"][i]) and np.array_equal(st,arrays["T"][i]),"observed counts match exact sampled ballots")
            require(np.isclose(float(sample["fraction"]),arrays["fraction"][i]),"fraction identity")
            require(int(sample["replicate"])==int(arrays["resample"][i]),"resample identity")
        files[row["profile"]]=arrays
    require(len(files)==36,"complete original 36-profile cohort")
    cfg=json.loads((out/"CONFIG.json").read_text())
    expected_cells={(float(f),r) for f in cfg["data"]["fractions"] for r in range(cfg["data"]["resamples"])}
    for profile in files:
        rows=[r for r in samples if r["profile"]==profile]
        require(len(rows)==len(expected_cells) and {(float(r["fraction"]),int(r["replicate"])) for r in rows}==expected_cells,
                "all prescribed fractions and resamples per profile")
    return bycase,files


def audit(out,device="cpu"):
    from run_experiment import (atomic_json, source_hashes, set_metrics, joint_from_memberships,
        draw_memberships, partial_relation_uc, beta_q, load_predictor, selective_row,
        descriptive_summary, METHODS, REPLAY_TOLERANCE, setup_device, decide_scores)
    from stepc.learning import predict_observed
    started=time.perf_counter();out=Path(out).resolve()
    fit=json.loads((out/"FIT_COMPLETE.json").read_text())
    lock=json.loads((out/"RUN_LOCK.json").read_text());cfg=lock["config"]
    require(lock["source_sha256"]==source_hashes(),"runner/kernel/model/config source hashes changed")
    for relative,digest in fit["hashes"].items():
        path=out/relative
        require(path.is_file() and sha256(path)==digest,"artifact hash "+relative)
    required={str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name not in
              ("RUNNING.lock","run.log","FIT_COMPLETE.json","COMPLETE.json","AUDIT.json","FAILED.json")}
    require(required==set(fit["hashes"]),"unhashed or missing output artifact")
    replay_device=setup_device(device,smoke=cfg["mode"]=="smoke",audit=True)
    cases,files=audit_data(out)
    folds=json.loads((out/"FOLDS.json").read_text());expected=json.loads((out/"EXPECTED_COVERAGE.json").read_text())
    from stepc.data import source_folds
    intended=source_folds()[:2] if cfg["mode"]=="smoke" else source_folds()
    require(len(folds)==len(intended),"two QA/six study source folds")
    for actual,target in zip(folds,intended):
        require(actual["test_source"]==target["test_source"] and actual["train_sources"]==target["train_sources"] and
                actual.get("development_source",actual.get("dev_source"))==target.get("development_source",target.get("dev_source")),
                "fixed original cyclic source folds")
    index=json.loads((out/"PREDICTION_INDEX.json").read_text())
    metrics=read_csv(out/"per_case_metrics.csv");savedmetrics={metric_key(r):r for r in metrics}
    require(len(savedmetrics)==len(metrics)==expected["metric_rows"],"metric row uniqueness/coverage")
    require(len(index)==expected["prediction_files"],"prediction file count")
    init_count=cfg["learning"]["initializations"]
    expected_pairs={(int(f.get("fold",f.get("fold_id"))),init,cid,method)
        for f in folds for init in range(init_count) for cid,c in cases.items() if c["source"]==f["test_source"]
        for method in (*METHODS,"count_jeffreys","count_raw","count_soft_plugin")}
    actual_pairs={(int(r["fold"]),int(r["initialization"]),r["case_id"],r["method"]) for r in index}
    require(len(actual_pairs)==len(index) and actual_pairs==expected_pairs,"all held-out cases, fractions, resamples, initializations and methods")
    choices_cache={};predictors={};recomputed_rows=[];draw_checks=0;replay_checks=0
    for record in index:
        f,init,cid,method=int(record["fold"]),int(record["initialization"]),record["case_id"],record["method"]
        sample=cases[cid];arrays=files[sample["profile"]];i=int(sample["array_index"])
        path=out/record["path"]
        raw={k:v for k,v in np.load(path,allow_pickle=False).items()}
        require(np.array_equal(raw["W"],arrays["W"][i]) and np.array_equal(raw["T"],arrays["T"][i]),"prediction input counts "+str(path))
        require(np.array_equal(raw["y"],arrays["y"]) and np.array_equal(raw["fullA"],arrays["fullA"]),"prediction evaluation truth")
        choicepath=record["choice_path"]
        if choicepath not in choices_cache:
            choicefile=out/choicepath;chosen=json.loads(choicefile.read_text());choices_cache[choicepath]=chosen
            fold=next(v for v in folds if int(v.get("fold",v.get("fold_id")))==f)
            require(chosen["selection_sources"]==[fold.get("development_source",fold.get("dev_source"))],"development-source isolation")
            require(chosen["training_sources"]==fold["train_sources"] and chosen["test_source"]==fold["test_source"],"source isolation")
            require(chosen["test_predictions_not_started"] and np.isclose(chosen["selection_fraction"],.1),"choices frozen at prescribed development fraction")
            for kind,choice in chosen["choices"].items():
                if kind in METHODS:
                    require(sha256(out/choice["checkpoint"])==choice["checkpoint_sha256"],"selected checkpoint hash")
                    candidates=json.loads((choicefile.parent/kind/"DEVELOPMENT_CANDIDATES.json").read_text())
                    require(choice in candidates,"selected candidate was saved before testing")
                    selected=max(candidates,key=lambda r:(r["development_f1"],-r.get("snapshot",0),-r["weight"]))
                    require(choice==selected,"development-only deterministic candidate selection")
                    proofs=json.loads((choicefile.parent/kind/"DEVICE_AND_GRADIENT_PROOF.json").read_text())
                    if kind!="empirical_bayes":
                        require(all(p["finite_gradients"] and p["parameters_changed"] and p["steps"]>0 for p in proofs),"finite gradient and weight-change proof")
        choices=choices_cache[choicepath]["choices"]
        require(path.stat().st_mtime_ns >= (out/choicepath).stat().st_mtime_ns,"choice file precedes held-out prediction")
        view={"W":raw["W"],"T":raw["T"]} # reference arrays never enter replay predictor
        q=None;scores=None;predictions={}
        if method in METHODS:
            key=(choicepath,method)
            if key not in predictors:
                predictors[key]=load_predictor(choices[method],out,replay_device)
            if method=="empirical_bayes":
                replay={"q":beta_q(view["W"],predictors[key]["alpha"])}
            else:
                replay=predict_observed(predictors[key],view,replay_device)
            for name in ("q","components","weights","scores","logits","logweights"):
                if name in replay:
                    require(name in raw and np.allclose(replay[name],raw[name],atol=REPLAY_TOLERANCE,rtol=0),"selected forward replay "+name+" "+str(path))
            replay_checks+=1
        elif method=="count_jeffreys":
            require(np.allclose(raw["q"],jeffreys_orientation_probabilities(view["W"]),atol=1e-12,rtol=0),"Jeffreys orientation q differs from matchup means")
        elif method=="count_raw":
            require(np.array_equal(raw["adjacency"],view["W"]>view["W"].T),"strict observed majority relation")
            predictions={"raw_majority_uc":partial_relation_uc(raw["adjacency"])}
        elif method=="count_soft_plugin":
            P=jeffreys_pair_means(view["W"]);scores=soft_uc_plugin(P,tau=.035,gamma=.035)
            require(np.allclose(P,raw["P"],atol=1e-12) and np.allclose(scores,raw["scores"],atol=1e-12),"soft plugin raw scores")
            predictions={"dev_calibrated_secondary":decide_scores(scores,choices[method]["threshold"])}
        if "q" in raw:
            q=raw["q"]
            require(int(raw["draw_seed"])==record["draw_seed"],"saved draw seed")
            require(len(raw["memberships"])==cfg["posterior"]["test_draws"]==record["draws"],"prescribed native draw count")
            # Replay raw saved q, not newly rounded CPU q: membership bits can
            # change near uniform thresholds even when CPU/GPU agree at 1e-5.
            members,components=draw_memberships(q,record["draws"],record["draw_seed"],raw["components"],raw["weights"])
            require(np.array_equal(members,raw["memberships"]) and np.array_equal(components,raw["component_ids"]),"joint draw/whole-tournament component replay")
            require(np.allclose(q,(raw["weights"][:,None,None]*raw["components"]).sum(0),atol=1e-6),"mixture marginal orientation probabilities")
            joint=joint_from_memberships(members)
            require(np.allclose(joint["membership_size_joint"],raw["membership_size_joint"],atol=1e-12),"membership/cardinality raw joint")
            require(np.allclose(joint["cardinality_probabilities"],raw["cardinality_probabilities"],atol=1e-12),"raw cardinality law")
            require(abs(joint["expected_f1"]-float(raw["gfm_expected_f1"]))<1e-12,"GFM expected F1")
            scores=joint["marginals"]
            require(np.allclose(scores,raw["marginals"],atol=1e-12),"saved marginal means")
            predictions={"gfm":joint["gfm"],"half":joint["half"],"hard_q_diagnostic":partial_relation_uc(q>.5)}
            draw_checks+=1
        elif method in ("ordinary","relational"):
            scores=raw["scores"];predictions={"dev_calibrated_direct":decide_scores(scores,choices[method]["threshold"])}
        require(set(k for k in raw if k.startswith("decision_"))=={"decision_"+r for r in predictions},"decoder output coverage")
        for readout,pred in predictions.items():
            require(np.array_equal(pred,raw["decision_"+readout]),"raw native decoder recomputation "+readout)
            key=(f,init,cid,method,readout)
            require(key in savedmetrics,"missing per-case metric")
            values=set_metrics(raw["y"],pred,scores,q,raw["fullA"])
            compare_metrics(savedmetrics[key],values,str(key))
            measured=float(record["inference_seconds"])
            require(np.isfinite(measured) and measured>=0 and float(savedmetrics[key]["inference_seconds"])==measured,
                    "finite consistent per-method/readout inference time")
            recomputed_rows.append({**savedmetrics[key],"fraction":float(sample["fraction"]),**values})
    require(len(recomputed_rows)==len(metrics),"all metric rows recomputed")
    selective=read_csv(out/"selective_metrics.csv")
    require(len(selective)==expected["selective_rows"]==len(expected_pairs)//8*len(cfg["certificates"]["deltas"]),"all three selective interval rows")
    selective_keys=set()
    for saved in selective:
        f,init,cid=int(saved["fold"]),int(saved["initialization"]),saved["case_id"]
        delta=float(saved["delta"]);key=(f,init,cid,delta)
        require(key not in selective_keys,"duplicate selective row");selective_keys.add(key)
        require(delta in cfg["certificates"]["deltas"],"prescribed selective delta")
        sample=cases[cid];arrays=files[sample["profile"]];i=int(sample["array_index"])
        values,intervals,bounds=selective_row({"W":arrays["W"][i],"y":arrays["y"]},delta)
        require(saved["coverage_guarantee"]=="False","no finite-cohort confidence guarantee")
        compare_metrics(saved,{k:v for k,v in values.items() if k!="coverage_guarantee"},str(key))
        path=out/f"fold_{f:02d}"/f"init_{init:02d}"/"predictions"/cid/f"certificate_delta_{delta:g}.npz"
        raw=np.load(path,allow_pickle=False)
        for name,array in (("known",intervals.known),("lower",intervals.lower),("upper",intervals.upper),("inner",bounds.inner),("outer",bounds.outer)):
            require(np.array_equal(array,raw[name]),"selective logical/interval replay "+name)
    expected_selective={(f,i,c,delta) for f,i,c,_ in expected_pairs for delta in cfg["certificates"]["deltas"]}
    require(selective_keys==expected_selective,"exact selective fold/init/case/delta coverage")
    profile_rows,source_rows,summary=descriptive_summary(recomputed_rows)
    for table,computed,keys in (("per_profile_10pct.csv",profile_rows,("method","readout","source","profile")),
                               ("per_source_10pct.csv",source_rows,("method","readout","source"))):
        saved={tuple(r[k] for k in keys):r for r in read_csv(out/table)}
        require(len(saved)==len(computed),"profile/source table coverage")
        for row in computed:
            compare_metrics(saved[tuple(row[k] for k in keys)],row,"profile/source aggregation")
    savedsummary={(r["method"],r["readout"]):r for r in read_csv(out/"descriptive_summary_10pct.csv")}
    require(len(summary)==len(savedsummary),"summary decoder coverage")
    for row in summary:
        compare_metrics(savedsummary[(row["method"],row["readout"])],row,"source macro aggregation")
    for fraction,cohort,name in ((.05,"all","descriptive_summary_5pct.csv"),(.25,"all","descriptive_summary_25pct.csv"),
        (.1,"singleton","descriptive_summary_10pct_singleton.csv"),(.1,"selective_non_singleton","descriptive_summary_10pct_selective_non_singleton.csv")):
        _,_,expected_summary=descriptive_summary(recomputed_rows,fraction,cohort)
        saved={(r["method"],r["readout"]):r for r in read_csv(out/name)}
        require(len(saved)==len(expected_summary),"secondary/cohort summary coverage")
        for row in expected_summary:
            compare_metrics(saved[(row["method"],row["readout"])],row,"secondary/cohort macro aggregation")
    timing=json.loads((out/"TIMING.json").read_text())
    budget_valid=all(float(t["charged_seconds"])<=1.02*float(t["allocated_seconds"]) for t in timing["methods"])
    require(budget_valid==timing["equal_allocation_valid"]==fit["equal_allocation_valid"],"immutable 2% budget validity gate")
    report={"status":"AUDIT_COMPLETE","hash_checks":len(fit["hashes"]),"selected_forward_replays":replay_checks,
        "selected_forward_tolerance":REPLAY_TOLERANCE,"replay_device":str(replay_device),"joint_draw_replays":draw_checks,
        "recomputed_metric_rows":len(recomputed_rows),"recomputed_selective_rows":len(selective),
        "seconds":time.perf_counter()-started,"coverage_verified":True,"source_hashes_verified":True,
        "data_hashes_verified":True,"checkpoint_hashes_verified":True,"selected_replay_verified":True,
        "EQUAL_ALLOCATION_VALID":budget_valid,"mode":cfg["mode"],
        "classification":"software_QA" if cfg["mode"]=="smoke" else "timing_pilot" if cfg["mode"]=="pilot" else
            "exploratory_budget_deviation" if not budget_valid else "exploratory_descriptive_benchmark",
        "inference":"no formal p-values or superiority/population/calibration claims"}
    atomic_json(out/"AUDIT.json",report)
    atomic_json(out/"COMPLETE.json",{**report,"status":"COMPLETE","FIT_COMPLETE_sha256":sha256(out/"FIT_COMPLETE.json"),
        "AUDIT_sha256":sha256(out/"AUDIT.json")})
    print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);p.add_argument("--device",default="cpu")
    args=p.parse_args();audit(args.out,args.device)


if __name__=="__main__":
    main()
