from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, atomic_npz, emit, seed_for, sha, write_rows, finish_unit, unit_complete
from .data import parse_profile,ballot_counts,nested_subsamples
from .operators import hard_core,hard_core_independent
from .baselines import count_methods
from .metrics import select_threshold,thresholds,decide,set_metrics

def inventory_profiles(code,cfg,out):
    manifest=pd.read_csv(code/"data/PREFLIB_INPUT_MANIFEST.csv",dtype={"source":str})
    if cfg.get("smoke"):
        # One complete candidate from each of three sources, and one partial candidate.
        screening=[]
        for _,entry in manifest.iterrows():
            prof=parse_profile(code/entry.relative_path); W,T=ballot_counts(prof,prof["multiplicity"])
            ii,jj=np.triu_indices(prof["n"],1); y=hard_core(W>W.T,"uc")
            tc=hard_core(W>W.T,"tc")
            if prof["n"]<=40 and 1<y.sum()<prof["n"] and 1<tc.sum()<prof["n"] and np.all((W+W.T)[ii,jj]>0) and np.all(W[ii,jj]!=W[jj,ii]):
                screening.append(entry)
        chosen=[]; seen=set()
        for entry in screening:
            if entry.source not in seen: chosen.append(entry.to_dict()); seen.add(entry.source)
            if len(chosen)==3: break
        if len(chosen)<3: raise RuntimeError("Software test needs three complete real sources")
        manifest=pd.DataFrame(chosen)
    rows=[]
    for _,entry in manifest.iterrows():
        path=code/entry.relative_path
        if sha(path)!=entry.sha256: raise RuntimeError("Bundled raw input checksum failed: "+str(path))
        prof=parse_profile(path); W,T=ballot_counts(prof,prof["multiplicity"]); n=prof["n"]; ii,jj=np.triu_indices(n,1)
        m=W+W.T; strict=bool(np.all(m[ii,jj]>0) and np.all(W[ii,jj]!=W[jj,ii]))
        row=entry.to_dict(); row.update(n=n,voters=prof["voters"],strict_complete=strict,
                  decisive_pair_coverage=float(np.mean(m[ii,jj]>0)),observation_coverage=float(np.mean((m+T)[ii,jj]>0)),
                  exact_majority_ties=int(np.sum((m[ii,jj]>0)&(W[ii,jj]==W[jj,ii]))),tie_only_pairs=int(np.sum((m[ii,jj]==0)&(T[ii,jj]>0))))
        for target in ("uc","tc"):
            y=hard_core(W>W.T,target)
            if not np.array_equal(y,hard_core_independent(W>W.T,target)): raise RuntimeError("Real reference-core disagreement")
            row[target+"_size"]=int(y.sum()); row[target+"_selective"]=bool(1<y.sum()<n)
        rows.append(row)
    inv=pd.DataFrame(rows).sort_values(["source","file"]).reset_index(drop=True)
    # Exact decisive/tied pair-count duplicates are retained as reconstruction inputs, but
    # only one is admitted to the primary analysis. Dependence is always clustered by source.
    seen={}; duplicates=[]
    for i,row in inv.iterrows():
        p=parse_profile(code/row.relative_path); W,T=ballot_counts(p,p["multiplicity"])
        import hashlib
        digest=hashlib.sha256(W.tobytes()+T.tobytes()).hexdigest(); key=(row.source,digest)
        duplicates.append(key in seen); seen[key]=row.file
    inv["duplicate_pair_counts_in_source"]=duplicates
    inv.to_csv(out/"REFERENCE_INVENTORY_BEFORE_METHODS.csv",index=False)
    clean=inv[inv.strict_complete & inv.uc_selective & ~inv.duplicate_pair_counts_in_source]
    passed=len(clean)>=cfg["real"]["min_primary_profiles"] and clean.source.nunique()>=cfg["real"]["min_primary_sources"]
    gate={"passed":bool(passed),"primary_profiles":len(clean),"primary_sources":int(clean.source.nunique()),
          "definition":"all decisive pairs observed; no exact empirical majority ties; 1 < UC size < n; no exact within-source count duplicate",
          "no_method_performance_used":True,"source_folds":sorted(inv.source.unique().tolist())}
    atomic_json(out/"ELIGIBILITY_GATE.json",gate)
    if not passed: raise RuntimeError("Prespecified genuine real-data eligibility gate failed; inspect ELIGIBILITY_GATE.json")
    atomic_json(out/"SOURCE_FOLDS_BEFORE_METHODS.json",{
      "folds":[{"test_source":s,"development_sources":[t for t in sorted(inv.source.unique()) if t!=s]} for s in sorted(inv.source.unique())],
      "primary_weighting":"average resamples within profile, profiles within source, then sources equally",
      "target":"finite full-profile empirical majority core; reconstruction, not labeled population-core accuracy"})
    return inv

def run_profile(code,cfg,row,out,device):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if unit_complete(out): return
    profile=parse_profile(code/row.relative_path); n=profile["n"]
    full_W,full_T=ballot_counts(profile,profile["multiplicity"])
    ys={t:hard_core(full_W>full_W.T,t) for t in ("uc","tc")}
    cases=[]; score_arrays={}; fixed_arrays={}; joint_arrays={"uc":[],"tc":[]}; Ws=[]; Ts=[]; ballots=[]; diagnostics=[]
    rc=cfg["real"]
    for replicate in range(rc["replicates"]):
        rng=np.random.default_rng(seed_for(cfg["seed"],"real-voter-subsample",row.file,replicate))
        subs=nested_subsamples(profile,rc["fractions"],rng)
        for fraction,(baseW,baseT,multiplicity) in subs.items():
            settings=[0.]+(rc["additional_missingness"] if fraction==rc["primary_fraction"] else [])
            for missing in settings:
                W,T=baseW.copy(),baseT.copy()
                if missing:
                    rr=np.random.default_rng(seed_for(cfg["seed"],"real-edge-mask",row.file,replicate,fraction,missing))
                    ii,jj=np.triu_indices(n,1); drop=rr.random(len(ii))<missing
                    W[ii[drop],jj[drop]]=0; W[jj[drop],ii[drop]]=0; T[ii[drop],jj[drop]]=0; T[jj[drop],ii[drop]]=0
                case_id=f"{row.file}/r{replicate:02d}/f{fraction:g}/m{missing:g}"
                scores,fixed,joint,meta=count_methods(W,T,cfg,device,seed_for(cfg["seed"],"posterior",case_id))
                idx=len(cases); cases.append({"index":idx,"case_id":case_id,"replicate":replicate,"fraction":fraction,"missing":missing,
                  "voters_sampled":int(multiplicity.sum()),"n":n,"source":row.source,"file":row.file,
                  "relation_class":"strict_complete" if row.strict_complete else "complete_with_majority_ties" if row.decisive_pair_coverage==1 else "partial_decisive_relation",
                  "strict_complete":bool(row.strict_complete),"duplicate":bool(row.duplicate_pair_counts_in_source),
                  "uc_selective":bool(row.uc_selective),"tc_selective":bool(row.tc_selective)})
                Ws.append(W); Ts.append(T); ballots.append(multiplicity); diagnostics.append({"case_id":case_id,**meta})
                for target in ("uc","tc"):
                    for method,s in scores[target].items(): score_arrays.setdefault(target+"__"+method,[]).append(s)
                    for method,pred in fixed[target].items(): fixed_arrays.setdefault(target+"__"+method,[]).append(pred)
                    joint_arrays[target].append(joint[target])
    atomic_npz(out/"scores.npz",**{k:np.asarray(v) for k,v in score_arrays.items()},y_uc=ys["uc"],y_tc=ys["tc"])
    atomic_npz(out/"fixed_sets.npz",**{k:np.asarray(v) for k,v in fixed_arrays.items()})
    atomic_npz(out/"observations.npz",W=np.array(Ws),T=np.array(Ts),ballot_multiplicity=np.array(ballots),full_W=full_W,full_T=full_T,
               original_multiplicity=profile["multiplicity"],ranks=profile["ranks"])
    atomic_npz(out/"posterior_joint_samples.npz",**{k:np.array(v) for k,v in joint_arrays.items()})
    write_rows(out/"cases.csv",cases); write_rows(out/"baseline_diagnostics.csv",diagnostics)
    finish_unit(out,[out/name for name in ["scores.npz","fixed_sets.npz","observations.npz","posterior_joint_samples.npz","cases.csv","baseline_diagnostics.csv"]])

def analyze_real(cfg,inv,out):
    rc=cfg["real"]; cache={}; rows=[]; choices=[]
    for _,row in inv.iterrows():
        p=out/"profiles"/row.file
        with np.load(p/"scores.npz") as z: scores={k:z[k] for k in z.files}
        with np.load(p/"fixed_sets.npz") as z: fixed={k:z[k] for k in z.files}
        cases=pd.read_csv(p/"cases.csv",dtype={"source":str})
        cache[row.file]=(scores,fixed,cases)
    for source in sorted(inv.source.unique()):
        for target in ("uc","tc"):
            development=inv[(inv.source!=source)&inv.strict_complete&inv[target+"_selective"]&~inv.duplicate_pair_counts_in_source]
            if not len(development): raise RuntimeError("No independent calibration profiles for "+source+"/"+target)
            for method in ("ste_lse","post","copeland","smooth_copeland","winrate","btl","hodge","rank_centrality"):
                xs=[]; ys=[]; groups=[]; profile_names=[]
                for _,dev in development.iterrows():
                    s,_,cases=cache[dev.file]
                    indices=cases[cases.missing==0]["index"].to_numpy()
                    for i in indices:
                        xs.append(s[target+"__"+method][i]); ys.append(s["y_"+target]); groups.append(dev.source); profile_names.append(dev.file)
                counts={g:groups.count(g) for g in set(groups)}; weights=[1/counts[g] for g in groups]
                for rule in cfg["decisions"]["rules"]:
                    h,dev_f1=select_threshold(xs,ys,thresholds(cfg),rule,weights)
                    choices.append({"heldout_source":source,"target":target,"method":method,"rule":rule,"threshold":h,
                                    "development_source_macro_f1":dev_f1,"development_sources":";".join(sorted(set(groups))),
                                    "development_profile_ids":";".join(sorted(set(profile_names))),"includes_test_source":False})
                    for _,test in inv[inv.source==source].iterrows():
                        s,_,cases=cache[test.file]
                        for case in cases.to_dict("records"):
                            score=s[target+"__"+method][case["index"]]
                            rows.append({**case,"target":target,"method":method,"rule":rule,"threshold":h,
                                         **set_metrics(s["y_"+target],decide(score,rule,h),score)})
            for _,test in inv[inv.source==source].iterrows():
                s,fixed,cases=cache[test.file]
                for method in ("hard","post_half","post_gfm","all","none"):
                    for case in cases.to_dict("records"):
                        rows.append({**case,"target":target,"method":method,"rule":"native","threshold":0,
                                     **set_metrics(s["y_"+target],fixed[target+"__"+method][case["index"]])})
    write_rows(out/"CALIBRATION_CHOICES.csv",choices); write_rows(out/"per_case_metrics.csv",rows)
    df=pd.DataFrame(rows)
    group=["source","file","target","method","rule","fraction","missing","strict_complete","duplicate","core_class","relation_class"]
    profiles=df.groupby(group,dropna=False)[["f1","exact","precision","recall","ap","auc","selected_size","target_size"]].mean().reset_index()
    profiles.to_csv(out/"profile_means.csv",index=False)
    sources=profiles.groupby([g for g in group if g!="file"],dropna=False)[["f1","exact","ap","auc","selected_size","target_size"]].mean().reset_index()
    sources.to_csv(out/"source_means.csv",index=False)
    primary=profiles[(profiles.target=="uc")&(profiles.fraction==rc["primary_fraction"])&(profiles.missing==0)&profiles.strict_complete&~profiles.duplicate&(profiles.core_class=="selective_non_singleton")]
    primary_sources=primary.groupby(["source","method","rule"])[["f1","exact","ap","auc"]].mean().reset_index()
    primary_sources.to_csv(out/"PRIMARY_SOURCE_MACRO.csv",index=False)
    primary_sources.groupby(["method","rule"])[["f1","exact","ap","auc"]].mean().reset_index().to_csv(out/"PRIMARY_REAL_SUMMARY.csv",index=False)
    atomic_json(out/"REAL_ANALYSIS_LIMITS.json",{
      "primary_is_source_macro":True,"fractions_are_nested_not_independent":True,"resamples_are_not_independent_profiles":True,
      "small_number_of_primary_source_series":int(primary_sources.source.nunique()),
      "uncertainty":"report each source and descriptive source-level differences; no real-data superiority claim is automatically generated",
      "selection_is_conditional_on_finite_empirical_core_class":True,"population_core_accuracy_is_not_identified":True})

def run_real(code,cfg,out,device):
    code=Path(code); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    inv=inventory_profiles(code,cfg,out)
    for i,(_,row) in enumerate(inv.iterrows(),1):
        run_profile(code,cfg,row,out/"profiles"/row.file,device)
        emit(stage="real_profile_complete",complete=i,expected=len(inv),file=row.file,source=row.source,device=str(device))
    analyze_real(cfg,inv,out)
    finish_unit(out,[out/name for name in ["REFERENCE_INVENTORY_BEFORE_METHODS.csv","ELIGIBILITY_GATE.json","SOURCE_FOLDS_BEFORE_METHODS.json",
                "CALIBRATION_CHOICES.csv","per_case_metrics.csv","PRIMARY_REAL_SUMMARY.csv","PRIMARY_SOURCE_MACRO.csv"]]+[p/"COMPLETE.json" for p in sorted((out/"profiles").iterdir())])
