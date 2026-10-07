from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import atomic_json,sha,unit_complete

def validate_run(cfg,run):
    run=Path(run); lc=cfg["learning"]
    for stage in ("learning","counts","real","diagnostics","benchmark","analysis"):
        if not unit_complete(run/stage): raise RuntimeError("Incomplete stage: "+stage)
    metadata=json.loads((run/"ENVIRONMENT.json").read_text())
    if not cfg.get("smoke") and (metadata["device"]!="cuda:0" or metadata["gpu"] is None): raise RuntimeError("No production CUDA evidence")
    collections=sorted((run/"learning").glob("collection_*"))
    if len(collections)!=lc["collections"]: raise RuntimeError("Collection count mismatch")
    expected_per_init=len(lc["test_sizes"])*lc["test_graphs_per_size"]*(18+2)
    seen=set(); cases=0
    for cp in collections:
        d=pd.read_csv(cp/"per_case_metrics.csv",usecols=["collection","init","target","n","case_id","objective","readout","rule","f1","exact"])
        expected=expected_per_init*lc["initializations"]*len(lc["targets"])
        if len(d)!=expected: raise RuntimeError(f"Learning result rows {len(d)} != {expected}: {cp}")
        if d[["collection","init","target","n","case_id","objective","readout","rule"]].duplicated().any(): raise RuntimeError("Duplicate learned evaluation row")
        if not np.isfinite(d.f1).all() or not np.isfinite(d.exact).all(): raise RuntimeError("Nonfinite learned metrics")
        for p in sorted((cp/"datasets").glob("*.npz")):
            data=np.load(p)
            ids=set(data["case_id"].tolist())
            if len(ids)!=len(data["W"]) or ids&seen: raise RuntimeError("Duplicate or overlapping graph identities")
            seen|=ids
            if p.name.startswith("test"): cases+=len(ids)
        histories=list(cp.glob("*/init_*/*_lambda_*/history.json"))
        expected_models=len(lc["targets"])*lc["initializations"]*(1+3*len(lc["core_weights"]))
        if len(histories)!=expected_models: raise RuntimeError("Missing candidate training histories")
        for p in histories:
            history=json.loads(p.read_text())
            if [r["epoch"] for r in history]!=list(range(1,lc["epochs"]+1)): raise RuntimeError("Incomplete or duplicated training epochs: "+str(p))
            proof=json.loads((p.parent/"DEVICE_PROOF.json").read_text())
            if not cfg.get("smoke") and proof["model_parameter_devices"]!=["cuda:0"]: raise RuntimeError("CPU model in production study")
    if cases!=lc["collections"]*len(lc["test_sizes"])*lc["test_graphs_per_size"]: raise RuntimeError("Test graph count mismatch")
    for cp in sorted((run/"counts").glob("collection_*")):
        d=pd.read_csv(cp/"per_case_metrics.csv",usecols=["case_id","target","method","rule","f1"])
        expected=len(lc["test_sizes"])*lc["test_graphs_per_size"]*2*32
        if len(d)!=expected or d[["case_id","target","method","rule"]].duplicated().any(): raise RuntimeError("Incomplete or duplicate count-control results")
    real=pd.read_csv(run/"real/REFERENCE_INVENTORY_BEFORE_METHODS.csv",dtype={"source":str})
    expected_cases=cfg["real"]["replicates"]*(len(cfg["real"]["fractions"])+len(cfg["real"]["additional_missingness"]))
    for row in real.to_dict("records"):
        p=run/"real/profiles"/row["file"]
        if not unit_complete(p) or len(pd.read_csv(p/"cases.csv"))!=expected_cases: raise RuntimeError("Incomplete real profile")
    choices=pd.read_csv(run/"real/CALIBRATION_CHOICES.csv",dtype={"heldout_source":str,"development_sources":str})
    if choices.includes_test_source.any(): raise RuntimeError("Held-out source leakage")
    for row in choices.to_dict("records"):
        if row["heldout_source"] in row["development_sources"].split(";"): raise RuntimeError("Held-out source leakage in listed folds")
    diagnostics=pd.read_csv(run/"diagnostics/OPERATOR_DIAGNOSTICS.csv")
    expected_diag=0
    import math
    for n in cfg["diagnostics"]["sizes"]:
        depths=set([1,math.ceil(math.log2(n)),math.ceil(n/2),n-1])
        expected_diag+=len(cfg["diagnostics"]["families"])*cfg["diagnostics"]["independent_graphs"]*len(cfg["diagnostics"]["temperatures"])*(1+2*len(depths))
    if len(diagnostics)!=expected_diag or not diagnostics.finite.all(): raise RuntimeError("Incomplete operator diagnostics")
    if len(pd.read_csv(run/"benchmark/TRAINING_COST.csv"))!=len(cfg["benchmark"]["sizes"])*6: raise RuntimeError("Incomplete GPU training cost measurements")
    validated={"smoke":cfg.get("smoke",False),"planned_collections":lc["collections"],"initializations_per_collection":lc["initializations"],
               "unique_test_graphs":cases,"real_profiles":len(real),"real_observation_cases":len(real)*expected_cases,
               "operator_diagnostic_cells":len(diagnostics),"all_required_stages_verified":True,"device":metadata["device"],
               "prior_result_tables_imported":False,"interpretation":"software completeness and integrity; no automatic superiority conclusion"}
    atomic_json(run/"VALIDATION.json",validated)
    return validated
