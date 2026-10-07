from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import atomic_json,write_rows,finish_unit
from .metrics import paired_inference,holm

def run_analysis(cfg,run):
    run=Path(run); out=run/"analysis"; out.mkdir(exist_ok=True)
    # Average initializations within each collection. These are not new data collections.
    chunks=[]; family_chunks=[]; core_chunks=[]
    group=["collection","target","n","objective","readout","rule"]
    for p in sorted((run/"learning").glob("collection_*/per_case_metrics.csv")):
        d=pd.read_csv(p)
        chunks.append(d.groupby(group)[["f1","exact","ap","auc","selected_size","target_size"]].mean().reset_index())
        family_chunks.append(d.groupby(group+["family"])[["f1","exact","ap","auc"]].mean().reset_index())
        core_chunks.append(d.groupby(group+["core_class"])[["f1","exact","ap","auc"]].mean().reset_index())
    units=pd.concat(chunks,ignore_index=True); units.to_csv(out/"LEARNING_COLLECTION_MEANS.csv",index=False)
    units.groupby(group[1:])[["f1","exact","ap","auc","selected_size","target_size"]].agg(["mean","std"]).to_csv(out/"LEARNING_SUMMARY.csv")
    pd.concat(family_chunks,ignore_index=True).to_csv(out/"LEARNING_FAMILY_COLLECTION_MEANS.csv",index=False)
    pd.concat(core_chunks,ignore_index=True).to_csv(out/"LEARNING_CORE_CLASS_COLLECTION_MEANS.csv",index=False)
    primary=[]; raw=[]
    for comparison in cfg["primary_comparisons"]:
        d=units[(units.target==comparison["target"])&(units.n==comparison["n"])&(units.readout==comparison["readout"])&(units.rule==comparison["rule"])]
        pivot=d.pivot(index="collection",columns="objective",values="f1")
        if len(pivot)!=cfg["learning"]["collections"]: raise RuntimeError("Missing primary endpoint collections")
        diff=pivot[comparison["left"]]-pivot[comparison["right"]]
        stats=paired_inference(diff.to_numpy())
        primary.append({**comparison,**stats,"scope":"prespecified generator mix, architecture, budget, calibration; no universal or size-growth claim"})
        raw.extend([{**comparison,"collection":int(c),"difference":float(v)} for c,v in diff.items()])
    adjusted=holm([r["p"] if r["p"] is not None else 1. for r in primary])
    for row,p in zip(primary,adjusted): row["p_holm"]=float(p)
    write_rows(out/"PRIMARY_COMPARISONS.csv",primary); write_rows(out/"PRIMARY_COLLECTION_DIFFERENCES.csv",raw)
    count_chunks=[]; count_families=[]
    cg=["collection","target","n","method","rule","diagnostic"]
    for p in sorted((run/"counts").glob("collection_*/per_case_metrics.csv")):
        d=pd.read_csv(p)
        count_chunks.append(d.groupby(cg)[["f1","exact","ap","auc","selected_size","target_size"]].mean().reset_index())
        count_families.append(d.groupby(cg+["family","core_class"])[["f1","exact","ap","auc"]].mean().reset_index())
    cu=pd.concat(count_chunks,ignore_index=True); cu.to_csv(out/"COUNT_COLLECTION_MEANS.csv",index=False)
    cu.groupby(cg[1:])[["f1","exact","ap","auc","selected_size","target_size"]].agg(["mean","std"]).to_csv(out/"COUNT_SUMMARY.csv")
    pd.concat(count_families,ignore_index=True).to_csv(out/"COUNT_FAMILY_CORE_CLASS_MEANS.csv",index=False)
    # A source-level descriptive interval, not a confident inference with hundreds of independent elections.
    real=pd.read_csv(run/"real/PRIMARY_SOURCE_MACRO.csv",dtype={"source":str})
    real_comparisons=[]
    base=real[(real.method=="ste_lse")&(real.rule=="absolute")].set_index("source").f1
    for method,rule in [("post","absolute"),("post_gfm","native"),("copeland","absolute"),("btl","absolute"),("hard","native")]:
        competitor=real[(real.method==method)&(real.rule==rule)].set_index("source").f1
        common=base.index.intersection(competitor.index)
        delta=base.loc[common]-competitor.loc[common]
        stat=paired_inference(delta)
        stat.pop("p",None)
        real_comparisons.append({"left":"ste_lse/absolute","right":method+"/"+rule,**stat,
                                 "classification":"descriptive source-level interval; few source series"})
    write_rows(out/"REAL_DESCRIPTIVE_DIFFERENCES.csv",real_comparisons)
    atomic_json(out/"INTERPRETATION_CONSTRAINTS.json",{
      "smoke":cfg.get("smoke",False),"reuse_previous_result_tables":False,
      "primary_multiplicity":"Holm across exactly three learned endpoints",
      "sign_flip_assumption":"paired collection differences exchangeable under independent sign reversals under the null; not assumption-free",
      "t_interval_assumption":"independent collection means; approximate normality of their differences",
      "family_and_core_class_interactions":"descriptive, not confirmatory superiority",
      "real_target":"reconstruction of full finite-profile empirical cores, conditional on eligibility",
      "counts_reuse_learning_observations":"yes; an ablation on the same fresh observations, not additional independent evidence",
      "oracle_cardinality":"only explicitly marked diagnostic rows; excluded from automatic method comparisons",
      "proof_and_manuscript_revision":"outside this computational package; no theory claim revised automatically"})
    lines=["STE RunPod v1: measured results only", "", "Classification: "+("SOFTWARE SMOKE; NOT SCIENTIFIC EVIDENCE" if cfg.get("smoke") else "fresh GPU learning and operator computation; CPU truth checks/statistics and scalar baselines"), "",
           "Prespecified learned comparisons (STE minus equal-supervision auxiliary; common structural readout):"]
    for row in primary:
        lines.append(f"{row['target']} n={row['n']}: delta F1={row['difference']:.6f}; collection-level interval [{row['ci_low']}, {row['ci_high']}]; Holm p={row['p_holm']:.6f}")
    lines += ["", "Consult per-collection, per-family, and core-class tables. No growing-with-size conclusion is prespecified.",
              "Real data are finite-profile reconstruction with leave-source-out calibration. Read all source rows before interpreting its macro average.",
              "GFM optimizes expected F1 under the specified finite posterior sample distribution; this does not establish population-core truth.",
              "A completed experiment is not evidence of superiority. This report does not update manuscript claims."]
    (out/"READ_MEASURED_RESULTS.txt").write_text("\n".join(lines)+"\n")
    make_figures(cfg,run,units,real)
    finish_unit(out,[p for p in sorted(out.rglob("*")) if p.is_file() and p.name!="COMPLETE.json"])

def make_figures(cfg,run,units,real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    out=run/"analysis/figures"; out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(8,3.2),sharey=True)
    for panel,target in zip(ax,("uc","tc")):
        for objective,color,marker in [("ste","#2368a0","o"),("aux","#222222","s"),("pair","#777777","^")]:
            d=units[(units.target==target)&(units.objective==objective)&(units.readout=="structural")&(units.rule=="absolute")]
            mean=d.groupby("n").f1.mean(); se=d.groupby("n").f1.sem().fillna(0)
            panel.errorbar(mean.index,mean,yerr=se,marker=marker,color=color,label=objective,capsize=3)
        panel.set(title=target.upper(),xlabel="Test alternatives",ylim=(0,1)); panel.grid(alpha=.15)
    ax[0].set_ylabel("F1 (mean ± collection SEM)"); ax[1].legend(frameon=False); fig.tight_layout()
    fig.savefig(out/"learning.pdf"); fig.savefig(out/"learning.png",dpi=180); plt.close(fig)
    selected=real[((real.method=="ste_lse")&(real.rule=="absolute"))|((real.method=="post")&(real.rule=="absolute"))|((real.method=="post_gfm")&(real.rule=="native"))]
    pivot=selected.pivot(index="source",columns="method",values="f1")
    fig,ax=plt.subplots(figsize=(6.5,3.2))
    for method,color,marker in [("ste_lse","#2368a0","o"),("post","#222222","s"),("post_gfm","#777777","^")]:
        ax.plot(range(len(pivot)),pivot[method],marker=marker,color=color,label=method)
    ax.set_xticks(range(len(pivot)),pivot.index); ax.set(xlabel="Held-out PrefLib source series",ylabel="UC reconstruction F1",ylim=(0,1)); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/"real_sources.pdf"); fig.savefig(out/"real_sources.png",dpi=180); plt.close(fig)
