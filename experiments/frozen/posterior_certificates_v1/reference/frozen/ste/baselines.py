from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from .operators import hard_core,soft_core,sampled_strict_cores
from .metrics import gfm_from_samples

def normalize(s):
    s=np.asarray(s,float); span=s.max()-s.min()
    return (s-s.min())/span if span>1e-12 else np.full_like(s,.5)

def count_methods(W,T,cfg,device,seed):
    W=np.asarray(W,float); T=np.asarray(T,float); n=len(W); m=W+W.T
    P=(W+.5)/(m+1); np.fill_diagonal(P,.5); mask=~np.eye(n,dtype=bool); ii,jj=np.triu_indices(n,1)
    pt=torch.as_tensor(P,dtype=torch.float32,device=device)
    temp=cfg["learning"]["temperature"]
    with torch.no_grad():
        structural={target:soft_core(pt,target,temp).cpu().numpy() for target in ("uc","tc")}
    empirical=np.divide(W,m,out=np.full_like(W,.5),where=m>0)
    A=W>W.T; np.fill_diagonal(A,False)
    native={target:hard_core(A,target) for target in ("uc","tc")}
    copeland=(A.sum(1)-A.sum(0))/(2*(n-1))+.5
    soft_copeland=((expit((P-.5)/temp)*mask).sum(1)/(n-1))
    winrate=(P*mask).sum(1)/(n-1)
    weights=m[ii,jj]; wins=W[ii,jj]; l2=cfg["counts"]["btl_l2"]
    def objective(s):
        z=s[ii]-s[jj]; p=expit(z)
        loss=np.sum(weights*np.logaddexp(0,z)-wins*z)/max(weights.sum(),1)+l2*np.sum(s*s)/2
        residual=(weights*p-wins)/max(weights.sum(),1)
        g=np.bincount(ii,residual,minlength=n)-np.bincount(jj,residual,minlength=n)+l2*s
        return loss,g
    fit=minimize(objective,np.zeros(n),jac=True,method="L-BFGS-B",options={"maxiter":cfg["counts"]["btl_iterations"],"ftol":1e-12,"gtol":1e-8})
    if not np.isfinite(fit.x).all(): raise RuntimeError("Nonfinite BTL baseline")
    btl=normalize(fit.x)
    L=np.diag(m.sum(1))-m; logodds=np.log(P)-np.log1p(-P)
    hodge=normalize(np.linalg.solve(L+l2*np.eye(n),(m*logodds).sum(1)))
    degree=(m>0).sum(1); dmax=max(1,int(degree.max()))
    transition=np.where(m>0,empirical.T/dmax,0); np.fill_diagonal(transition,0)
    np.fill_diagonal(transition,1-transition.sum(1)); stationary=np.ones(n)/n
    converged=False
    for iterations in range(1000):
        updated=stationary@transition
        if np.max(np.abs(updated-stationary))<1e-12: converged=True; stationary=updated; break
        stationary=updated
    if not np.isfinite(stationary).all(): raise RuntimeError("Nonfinite Rank Centrality")
    rank_centrality=normalize(stationary)
    rng=np.random.default_rng(seed); draws=cfg["counts"]["posterior_draws"]
    samples=rng.beta(W[ii,jj]+.5,W[jj,ii]+.5,size=(draws,len(ii)))>.5
    a=np.zeros((draws,n,n),bool); a[:,ii,jj]=samples; a[:,jj,ii]=~samples
    with torch.no_grad(): sampled=sampled_strict_cores(torch.as_tensor(a,device=device))
    joint={k:v.cpu().numpy() for k,v in sampled.items()}
    common={"copeland":copeland,"smooth_copeland":soft_copeland,"winrate":winrate,"btl":btl,"hodge":hodge,"rank_centrality":rank_centrality}
    scores={target:{**common,"ste_lse":structural[target],"post":joint[target].mean(0)} for target in ("uc","tc")}
    fixed={}; gfm_values={}
    for target in ("uc","tc"):
        gfm,gfm_value=gfm_from_samples(joint[target])
        gfm_values[target]=gfm_value
        fixed[target]={"hard":native[target],"post_half":scores[target]["post"]>=.5,"post_gfm":gfm,
                       "all":np.ones(n,bool),"none":np.zeros(n,bool)}
    meta={"btl_converged":bool(fit.success),"btl_iterations":int(fit.nit),"btl_status":str(fit.message),
          "rank_centrality_converged":converged,"rank_centrality_iterations":iterations+1,
          "posterior_draws":draws,"observed_pairs":int(((m+T)>0)[ii,jj].sum()),
          "decisive_pairs":int((m>0)[ii,jj].sum()),"tie_only_pairs":int(((T>0)&(m==0))[ii,jj].sum()),
          "missing_pairs":int(((m+T)==0)[ii,jj].sum()),
          "gfm_expected_f1_uc":gfm_values["uc"],"gfm_expected_f1_tc":gfm_values["tc"]}
    return scores,fixed,joint,meta
