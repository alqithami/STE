from pathlib import Path
import json
import numpy as np
import torch
from .common import atomic_json,atomic_npz,emit,seed_for,write_rows,finish_unit,unit_complete
from .baselines import count_methods
from .metrics import select_threshold,thresholds,decide,set_metrics
from .operators import soft_core

def case_results(data,i,cfg,device,path):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    if unit_complete(path):
        z=np.load(path/"scores.npz"); return {k:z[k] for k in z.files}
    score,fixed,joint,meta=count_methods(data["W"][i],data["T"][i],cfg,device,seed_for(cfg["seed"],"synthetic-posterior",str(data["case_id"][i])))
    with torch.no_grad():
        latent={t:soft_core(torch.as_tensor(data["P"][i],dtype=torch.float32,device=device),t,cfg["learning"]["temperature"]).cpu().numpy() for t in ("uc","tc")}
    arrays={}
    for target in ("uc","tc"):
        for method,s in score[target].items(): arrays[target+"__score__"+method]=s
        for method,s in fixed[target].items(): arrays[target+"__fixed__"+method]=s
        arrays[target+"__latent"]=latent[target]; arrays[target+"__y"]=data["y_"+target][i]
    atomic_npz(path/"scores.npz",**arrays)
    atomic_npz(path/"posterior_joint_samples.npz",**joint)
    atomic_json(path/"diagnostics.json",{"case_id":str(data["case_id"][i]),"device":str(device),**meta})
    finish_unit(path,[path/"scores.npz",path/"posterior_joint_samples.npz",path/"diagnostics.json"])
    return arrays

def run_count_study(cfg,learning,out,device):
    learning=Path(learning); out=Path(out); out.mkdir(parents=True,exist_ok=True); lc=cfg["learning"]
    for collection in range(lc["collections"]):
        cp=out/f"collection_{collection:02d}"; cp.mkdir(exist_ok=True)
        if unit_complete(cp): continue
        dp=learning/f"collection_{collection:02d}"/"datasets"
        dev=np.load(dp/"development.npz"); dev_results=[]; required=[]
        for i in range(len(dev["W"])):
            p=cp/"development"/f"case_{i:04d}"; dev_results.append(case_results(dev,i,cfg,device,p)); required.append(p/"COMPLETE.json")
        choices={}
        for target in ("uc","tc"):
            for method in ("ste_lse","post","copeland","smooth_copeland","winrate","btl","hodge","rank_centrality"):
                for rule in cfg["decisions"]["rules"]:
                    h,f=select_threshold([r[target+"__score__"+method] for r in dev_results],dev["y_"+target],thresholds(cfg),rule)
                    choices[target+"/"+method+"/"+rule]={"threshold":h,"development_f1":f}
        atomic_json(cp/"CHOICES_BEFORE_TEST.json",choices); rows=[]
        for n in lc["test_sizes"]:
            td=np.load(dp/f"test_n{n}.npz")
            for i in range(len(td["W"])):
                p=cp/f"test_n{n}"/f"case_{i:04d}"; result=case_results(td,i,cfg,device,p); required.append(p/"COMPLETE.json")
                common={"collection":collection,"case_id":str(td["case_id"][i]),"n":n,"family":str(td["family"][i]),"device":str(device)}
                for target in ("uc","tc"):
                    y=result[target+"__y"]
                    for method in ("ste_lse","post","copeland","smooth_copeland","winrate","btl","hodge","rank_centrality"):
                        s=result[target+"__score__"+method]
                        for rule in cfg["decisions"]["rules"]:
                            h=choices[target+"/"+method+"/"+rule]["threshold"]
                            rows.append({**common,"target":target,"method":method,"rule":rule,"threshold":h,"diagnostic":False,**set_metrics(y,decide(s,rule,h),s)})
                    for method in ("hard","post_half","post_gfm","all","none"):
                        rows.append({**common,"target":target,"method":method,"rule":"native","threshold":0,"diagnostic":False,**set_metrics(y,result[target+"__fixed__"+method])})
                    h=choices[target+"/ste_lse/absolute"]["threshold"]
                    latent=result[target+"__latent"]; estimated=result[target+"__score__ste_lse"]
                    rows.append({**common,"target":target,"method":"latent_P_ste","rule":"absolute","threshold":h,"diagnostic":True,**set_metrics(y,decide(latent,"absolute",h),latent)})
                    for method,s in [("oracle_k_ste",estimated),("oracle_k_latent_ste",latent)]:
                        pred=np.zeros(n,bool); pred[np.argsort(-s,kind="stable")[:int(y.sum())]]=True
                        rows.append({**common,"target":target,"method":method,"rule":"oracle_k","threshold":0,"diagnostic":True,**set_metrics(y,pred,s)})
                if (i+1)%64==0: emit(stage="count_study",collection=collection,n=n,complete=i+1,expected=len(td["W"]))
        write_rows(cp/"per_case_metrics.csv",rows)
        finish_unit(cp,[cp/"CHOICES_BEFORE_TEST.json",cp/"per_case_metrics.csv",*required])
        emit(stage="count_collection_complete",collection=collection,expected=lc["collections"])
    finish_unit(out,[p/"COMPLETE.json" for p in sorted(out.glob("collection_*"))])
