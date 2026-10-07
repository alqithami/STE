from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import atomic_json, atomic_npz, configure_torch, emit, seed_for, write_rows, finish_unit, unit_complete, sha
from .data import make_split
from .models import PairModel,pair_loss
from .operators import soft_core
from .metrics import select_threshold, thresholds, decide, set_metrics

@torch.no_grad()
def predict(model,data,target,lc,device):
    model.eval(); result={"structural":[],"direct":[]}
    for start in range(0,len(data["W"]),lc["eval_batch_size"]):
        W=torch.as_tensor(data["W"][start:start+lc["eval_batch_size"]],dtype=torch.float32,device=device)
        T=torch.as_tensor(data["T"][start:start+lc["eval_batch_size"]],dtype=torch.float32,device=device)
        P,direct=model(W,T); structural=soft_core(P,target,lc["temperature"])
        if not torch.isfinite(structural).all(): raise RuntimeError("Nonfinite evaluation output")
        result["structural"].append(structural.cpu().numpy()); result["direct"].append(direct.cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}

def train_candidate(cfg,collection,init,target,objective,weight,train,dev,path,device):
    lc=cfg["learning"]; path=Path(path); path.mkdir(parents=True,exist_ok=True)
    if unit_complete(path): return json.loads((path/"development.json").read_text())
    seed=seed_for(cfg["seed"],"initialization",collection,init)
    configure_torch(seed,device)
    model=PairModel(lc["hidden"],objective=="relational_aux").to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lc["learning_rate"])
    history=[]; selection=[]; latest=path/"latest.pt"; first_epoch=1
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=True)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); first_epoch=state["epoch"]+1
        history=[x for x in json.loads((path/"history.json").read_text()) if x["epoch"]<first_epoch]
        selection=[x for x in json.loads((path/"development.json").read_text()) if x["epoch"]<first_epoch]
    W=torch.as_tensor(train["W"],dtype=torch.float32,device=device)
    T=torch.as_tensor(train["T"],dtype=torch.float32,device=device)
    Y=torch.as_tensor(train["y_"+target],dtype=torch.float32,device=device)
    if device.type=="cuda" and (not W.is_cuda or any(not p.is_cuda for p in model.parameters())): raise RuntimeError("GPU placement failure")
    proof=path/"DEVICE_PROOF.json"
    if not proof.exists():
        atomic_json(proof,{"model_parameter_devices":sorted({str(p.device) for p in model.parameters()}),
          "training_count_device":str(W.device),"training_tie_device":str(T.device),"training_label_device":str(Y.device),
          "target":target,"objective":objective,"weight":weight,"parameters":sum(p.numel() for p in model.parameters()),
          "initialization_seed":seed,"collection":collection,"init":init,"smoke":cfg.get("smoke",False)})
    start_time=time.perf_counter()
    for epoch in range(first_epoch,lc["epochs"]+1):
        model.train(); totals=[]
        order=np.random.default_rng(seed_for(cfg["seed"],"batch-order",collection,init,epoch)).permutation(len(W))
        for start in range(0,len(W),lc["batch_size"]):
            idx=torch.as_tensor(order[start:start+lc["batch_size"]],device=device); opt.zero_grad(set_to_none=True)
            P,direct=model(W[idx],T[idx]); pair=pair_loss(P,W[idx]); core=torch.zeros((),device=device)
            if objective in ("aux","relational_aux"):
                core=F.binary_cross_entropy(direct.clamp(1e-6,1-1e-6),Y[idx])
            elif objective=="ste":
                score=soft_core(P,target,lc["temperature"],checkpoint=lc["checkpoint_tc"])
                core=F.binary_cross_entropy(score.clamp(1e-6,1-1e-6),Y[idx])
            loss=pair+weight*core
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),lc["gradient_clip"])
            if not torch.isfinite(norm): raise RuntimeError("Nonfinite training gradient")
            opt.step(); totals.append([float(pair.detach()),float(core.detach()),float(loss.detach()),float(norm)])
        mean=np.mean(totals,axis=0)
        history.append({"epoch":epoch,"pair_loss":float(mean[0]),"core_loss":float(mean[1]),"total_loss":float(mean[2]),
                        "gradient_norm_pre_clip":float(mean[3]),"elapsed_seconds_this_launch":time.perf_counter()-start_time})
        if epoch in lc["checkpoints"]:
            scores=predict(model,dev,target,lc,device); state_path=path/f"epoch_{epoch:03d}.pt"
            torch.save({"model":model.state_dict(),"epoch":epoch,"objective":objective,"target":target,"weight":weight},state_path)
            readouts=["structural"]+(["direct"] if objective in ("aux","relational_aux") else [])
            atomic_npz(path/f"development_epoch_{epoch:03d}.npz",y=dev["y_"+target],case_id=dev["case_id"],**{k:scores[k] for k in readouts})
            for readout in readouts:
                for rule in cfg["decisions"]["rules"]:
                    h,f1=select_threshold(scores[readout],dev["y_"+target],thresholds(cfg),rule)
                    selection.append({"epoch":epoch,"weight":weight,"readout":readout,"rule":rule,"threshold":h,"development_f1":f1,"checkpoint":str(state_path)})
            atomic_json(path/"history.json",history); atomic_json(path/"development.json",selection)
            tmp=path/"latest.tmp"; torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch},tmp); tmp.replace(latest)
            emit(stage="training",collection=collection,initialization=init,target=target,objective=objective,weight=weight,
                 epoch=epoch,epochs=lc["epochs"],device=str(device),seconds=time.perf_counter()-start_time)
    if device.type=="cuda":
        optimizer_devices=sorted({str(v.device) for s in opt.state.values() for k,v in s.items() if torch.is_tensor(v) and k!="step"})
        if optimizer_devices!=[str(device)]: raise RuntimeError("Optimizer state not on selected CUDA device")
        atomic_json(path/"OPTIMIZER_DEVICE.json",{"devices":optimizer_devices})
    finish_unit(path,[p for p in sorted(path.iterdir()) if p.is_file() and p.name!="COMPLETE.json" and p.suffix!=".tmp"])
    return selection

def run_learning(cfg,out,device):
    lc=cfg["learning"]; out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for collection in range(lc["collections"]):
        cp=out/f"collection_{collection:02d}"; cp.mkdir(exist_ok=True)
        if unit_complete(cp): emit(stage="learning_resume_verified",collection=collection); continue
        dp=cp/"datasets"; n=lc["train_n"]
        train=make_split(dp/"train.npz",cfg,collection,"train",n,lc["train_graphs"])
        dev=make_split(dp/"development.npz",cfg,collection,"development",n,lc["development_graphs"])
        # Every checkpoint/lambda/threshold choice for this collection is fixed before generating its tests.
        rows=[]; required=[dp/"train.npz",dp/"development.npz"]
        for target in lc["targets"]:
            for init in range(lc["initializations"]):
                ip=cp/target/f"init_{init:02d}"; ip.mkdir(parents=True,exist_ok=True)
                choices={}; candidate_records=[]
                for objective in lc["objectives"]:
                    candidates=[]
                    for weight in ([0.] if objective=="pair" else lc["core_weights"]):
                        p=ip/f"{objective}_lambda_{weight:g}"
                        records=train_candidate(cfg,collection,init,target,objective,weight,train,dev,p,device)
                        candidates.extend(records)
                        candidate_records.extend([{**x,"objective":objective} for x in records])
                        required.append(p/"COMPLETE.json")
                    for readout in ["structural"]+(["direct"] if objective in ("aux","relational_aux") else []):
                        for rule in cfg["decisions"]["rules"]:
                            available=[x for x in candidates if x["readout"]==readout and x["rule"]==rule]
                            choice=max(available,key=lambda x:(x["development_f1"],-x["epoch"],-x["weight"],-abs(x["threshold"]-.5),-x["threshold"]))
                            choices[f"{objective}/{readout}/{rule}"]={**choice,"objective":objective}
                atomic_json(ip/"CHOICES_BEFORE_TEST.json",choices)
                atomic_json(ip/"ALL_DEVELOPMENT_CANDIDATES.json",candidate_records)
                required.append(ip/"CHOICES_BEFORE_TEST.json")
                loaded={}
                for size in lc["test_sizes"]:
                    td=make_split(dp/f"test_n{size}.npz",cfg,collection,"test",size,lc["test_graphs_per_size"])
                    if dp/f"test_n{size}.npz" not in required: required.append(dp/f"test_n{size}.npz")
                    for key,choice in choices.items():
                        ck=choice["checkpoint"]
                        if ck not in loaded:
                            model=PairModel(lc["hidden"],choice["objective"]=="relational_aux").to(device)
                            model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)["model"])
                            loaded[ck]=model
                        model=loaded[ck]
                        # Cache both readouts for each selected model; rules never alter scores.
                        pred_path=ip/"predictions"/f"{choice['objective']}_l{choice['weight']:g}_e{choice['epoch']}_n{size}.npz"
                        if pred_path.exists():
                            pr=np.load(pred_path); scores=pr[choice["readout"]]
                        else:
                            prediction=predict(model,td,target,lc,device)
                            atomic_npz(pred_path,**prediction,y=td["y_"+target],family=td["family"],case_id=td["case_id"])
                            scores=prediction[choice["readout"]]
                        if pred_path not in required: required.append(pred_path)
                        for i,(s,y) in enumerate(zip(scores,td["y_"+target])):
                            row={"collection":collection,"init":init,"target":target,"n":size,"family":td["family"][i],"case_id":td["case_id"][i],
                                 "objective":choice["objective"],"readout":choice["readout"],"rule":choice["rule"],"threshold":choice["threshold"],
                                 "epoch":choice["epoch"],"weight":choice["weight"],"device":str(device),
                                 **set_metrics(y,decide(s,choice["rule"],choice["threshold"]),s)}
                            rows.append(row)
                    for i,y in enumerate(td["y_"+target]):
                        for control,pred in [("all",np.ones(size,bool)),("none",np.zeros(size,bool))]:
                            rows.append({"collection":collection,"init":init,"target":target,"n":size,"family":td["family"][i],"case_id":td["case_id"][i],
                                         "objective":control,"readout":"native","rule":"native","threshold":0,"epoch":0,"weight":0,"device":"control",
                                         **set_metrics(y,pred)})
                del loaded
                if device.type=="cuda": torch.cuda.empty_cache()
        write_rows(cp/"per_case_metrics.csv",rows); required.append(cp/"per_case_metrics.csv")
        finish_unit(cp,required); emit(stage="learning_collection_complete",collection=collection,expected=lc["collections"])
    write_rows(out/"DATASET_HASHES.csv",[{"path":str(p.relative_to(out)),"sha256":sha(p)} for p in sorted(out.glob("collection_*/datasets/*.npz"))])
    finish_unit(out,[out/"DATASET_HASHES.csv",*[p/"COMPLETE.json" for p in sorted(out.glob("collection_*"))]])
