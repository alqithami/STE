from __future__ import annotations
import math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import atomic_json, atomic_npz, configure_torch, emit, seed_for, write_rows, finish_unit, unit_complete
from .data import latent_graph,sample_observations
from .operators import soft_core,hard_core
from .models import PairModel,pair_loss

def synchronize(device):
    if device.type=="cuda": torch.cuda.synchronize(device)

def measure_operator(P,y,target,temperature,K,checkpoint,cfg,device):
    cfgd=cfg["diagnostics"]; pt=torch.as_tensor(P,dtype=torch.float32,device=device)
    ii,jj=np.triu_indices(len(P),1)
    base=torch.logit(pt[ii,jj].clamp(1e-7,1-1e-7))
    labels=torch.as_tensor(y,dtype=torch.float32,device=device)
    fw=[]; bw=[]; gradients=None; output=None
    for rep in range(cfgd["warmup"]+cfgd["repetitions"]):
        z=base.detach().clone().requires_grad_(True)
        probs=torch.sigmoid(z); matrix=torch.full_like(pt,.5)
        matrix[ii,jj]=probs; matrix[jj,ii]=1-probs
        if device.type=="cuda" and rep==cfgd["warmup"]: torch.cuda.reset_peak_memory_stats(device)
        synchronize(device); start=time.perf_counter()
        score=soft_core(matrix,target,temperature,checkpoint=checkpoint,K=K)
        loss=F.binary_cross_entropy(score.clamp(1e-6,1-1e-6),labels)
        synchronize(device); middle=time.perf_counter(); loss.backward(); synchronize(device); end=time.perf_counter()
        if not torch.isfinite(score).all() or not torch.isfinite(z.grad).all(): raise RuntimeError("Nonfinite operator or gradient diagnostic")
        if rep>=cfgd["warmup"]: fw.append((middle-start)*1000); bw.append((end-middle)*1000)
        gradients=z.grad.detach().cpu().numpy(); output=score.detach().cpu().numpy()
        last_loss=float(loss.detach())
        del matrix,score,loss,z,probs
    peak=float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else None
    reserved=float(torch.cuda.max_memory_reserved(device)/2**20) if device.type=="cuda" else None
    return {"forward_median_ms":float(np.median(fw)),"backward_median_ms":float(np.median(bw)),
            "forward_p90_ms":float(np.quantile(fw,.9)),"backward_p90_ms":float(np.quantile(bw,.9)),
            "peak_allocated_mb":peak,"peak_reserved_mb":reserved,"gradient_l2":float(np.linalg.norm(gradients)),
            "gradient_max_abs":float(np.max(np.abs(gradients))),"gradient_near_zero_fraction":float(np.mean(np.abs(gradients)<cfgd["near_zero_gradient"])),
            "gradient_exact_zero_fraction":float(np.mean(gradients==0)),"output_saturation_fraction":float(np.mean((output<1e-6)|(output>1-1e-6))),
            "finite":True,"bce":last_loss},output,gradients,np.array(fw),np.array(bw)

def run_diagnostics(cfg,out,device):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rows=[]; required=[]
    for n in cfg["diagnostics"]["sizes"]:
        depths=sorted(set([1,int(math.ceil(math.log2(n))),int(math.ceil(n/2)),n-1]))
        for family in cfg["diagnostics"]["families"]:
            for replicate in range(cfg["diagnostics"]["independent_graphs"]):
                P=latent_graph(n,family,np.random.default_rng(seed_for(cfg["seed"],"diagnostic-input",n,family,replicate)))
                for temperature in cfg["diagnostics"]["temperatures"]:
                    for target in ("uc","tc"):
                        y=hard_core(P>.5,target)
                        with torch.no_grad(): full=soft_core(torch.as_tensor(P,dtype=torch.float32,device=device),target,temperature).cpu().numpy()
                        for K in ([None] if target=="uc" else depths):
                            for checkpoint in ([False] if target=="uc" else [False,True]):
                                key=f"n{n}_{family}_r{replicate}_{target}_t{temperature:g}_K{K}_cp{int(checkpoint)}"
                                p=out/"cases"/key; p.mkdir(parents=True,exist_ok=True)
                                if unit_complete(p): row=__import__('json').loads((p/"metrics.json").read_text())
                                else:
                                    configure_torch(seed_for(cfg["seed"],key),device)
                                    if device.type=="cuda": torch.cuda.empty_cache()
                                    metrics,score,grad,fw,bw=measure_operator(P,y,target,temperature,K,checkpoint,cfg,device)
                                    row={"n":n,"family":family,"replicate":replicate,"target":target,"temperature":temperature,
                                         "depth":0 if K is None else K,"checkpoint":checkpoint,"device":str(device),
                                         "linf_vs_full_depth":float(np.max(np.abs(score-full))),"mean_abs_vs_hard":float(np.mean(np.abs(score-y))),**metrics}
                                    atomic_json(p/"metrics.json",row)
                                    atomic_npz(p/"raw.npz",P=P,y=y,score=score,full_depth_score=full,upper_logit_gradient=grad,forward_ms=fw,backward_ms=bw)
                                    finish_unit(p,[p/"metrics.json",p/"raw.npz"])
                                rows.append(row); required.append(p/"COMPLETE.json")
                emit(stage="tc_diagnostics_graph_complete",n=n,family=family,replicate=replicate,device=str(device))
    write_rows(out/"OPERATOR_DIAGNOSTICS.csv",rows)
    finish_unit(out,[out/"OPERATOR_DIAGNOSTICS.csv",*required])

def run_benchmark(cfg,out,device):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); b=cfg["benchmark"]; rows=[]
    for n in b["sizes"]:
        W=[]; T=[]; labels={"uc":[],"tc":[]}
        for i in range(b["batch_size"]):
            rng=np.random.default_rng(seed_for(cfg["seed"],"benchmark",n,i))
            P=latent_graph(n,["ordered","planted","separated","rank_mixture"][i%4],rng)
            w,t=sample_observations(P,rng,10,.25,.2); W.append(w); T.append(t)
            for target in labels: labels[target].append(hard_core(P>.5,target))
        W=torch.as_tensor(np.array(W),dtype=torch.float32,device=device); T=torch.as_tensor(np.array(T),dtype=torch.float32,device=device)
        for target in ("uc","tc"):
            y=torch.as_tensor(np.array(labels[target]),dtype=torch.float32,device=device)
            for objective in ("pair","aux","ste"):
                configure_torch(seed_for(cfg["seed"],"benchmark-model",n),device)
                model=PairModel(cfg["learning"]["hidden"]).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                times=[]
                for step in range(b["warmup"]+b["repetitions"]):
                    if step==b["warmup"] and device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
                    synchronize(device); start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                    P,direct=model(W,T); loss=pair_loss(P,W)
                    if objective=="aux": loss+=F.binary_cross_entropy(direct.clamp(1e-6,1-1e-6),y)
                    elif objective=="ste": loss+=F.binary_cross_entropy(soft_core(P,target,cfg["learning"]["temperature"],checkpoint=cfg["learning"]["checkpoint_tc"]).clamp(1e-6,1-1e-6),y)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); synchronize(device)
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite benchmark loss")
                    if step>=b["warmup"]: times.append((time.perf_counter()-start)*1000)
                rows.append({"n":n,"target":target,"objective":objective,"batch_size":b["batch_size"],"device":str(device),
                             "checkpoint_tc":cfg["learning"]["checkpoint_tc"] if target=="tc" and objective=="ste" else False,
                             "median_step_ms":float(np.median(times)),"p90_step_ms":float(np.quantile(times,.9)),
                             "peak_allocated_mb":float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=="cuda" else None,
                             "parameters":sum(p.numel() for p in model.parameters())})
                del optimizer,model
                if device.type=="cuda": torch.cuda.empty_cache()
                emit(stage="training_cost_benchmark",n=n,target=target,objective=objective,median_ms=rows[-1]["median_step_ms"])
    write_rows(out/"TRAINING_COST.csv",rows); finish_unit(out,[out/"TRAINING_COST.csv"])
