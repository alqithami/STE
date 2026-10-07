from __future__ import annotations
import itertools
import numpy as np
from scipy import stats

def decide(scores,rule,threshold=.5):
    s=np.asarray(scores,float)
    if rule=="absolute": return s>=threshold
    if rule=="relative":
        span=s.max()-s.min()
        z=(s-s.min())/span if span>1e-12 else np.full_like(s,.5)
        return z>=threshold
    if rule=="gap":
        order=np.argsort(-s,kind="stable"); gaps=s[order[:-1]]-s[order[1:]]
        if not len(gaps) or np.max(gaps)<=1e-12: return np.ones(len(s),bool)
        k=int(np.argmax(gaps))+1; y=np.zeros(len(s),bool); y[order[:k]]=True; return y
    raise ValueError(rule)

def set_metrics(y, pred, scores=None):
    y=np.asarray(y,bool); pred=np.asarray(pred,bool); tp=int((y&pred).sum())
    denom=int(y.sum()+pred.sum()); f1=2*tp/denom if denom else 1.
    result={"f1":float(f1),"exact":float(np.array_equal(y,pred)),
            "precision":float(tp/pred.sum()) if pred.sum() else float(not y.any()),
            "recall":float(tp/y.sum()) if y.sum() else float(not pred.any()),
            "target_size":int(y.sum()),"selected_size":int(pred.sum()),
            "core_class":"singleton" if y.sum()==1 else "full" if y.all() else "selective_non_singleton" if y.any() else "empty"}
    if scores is not None and y.any() and not y.all():
        s=np.asarray(scores,float); order=np.argsort(-s,kind="stable"); sy=y[order]
        group_ends=np.r_[np.flatnonzero(np.diff(s[order])!=0),len(s)-1]
        tp=np.cumsum(sy)[group_ends]; precision=tp/(group_ends+1); recall=tp/y.sum()
        ap=float(np.sum(precision*np.diff(np.r_[0,recall])))
        ranks=stats.rankdata(s,method="average"); pos=int(y.sum()); neg=len(y)-pos
        auc=float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))
        result.update(ap=ap,auc=auc)
    else: result.update(ap=np.nan,auc=np.nan)
    return result

def thresholds(cfg):
    step=cfg["decisions"]["threshold_step"]
    return [-1e-6,*np.arange(0,1+step/2,step).tolist(),1.000001]

def select_threshold(scores,ys,grid,rule="absolute",weights=None):
    if rule=="gap":
        values=[set_metrics(y,decide(s,rule))["f1"] for s,y in zip(scores,ys)]
        return .5,float(np.average(values,weights=weights))
    grid=np.asarray(grid,float); curves=[]
    for s,y in zip(scores,ys):
        s=np.asarray(s,float); y=np.asarray(y,bool)
        if rule=="relative":
            span=s.max()-s.min(); s=(s-s.min())/span if span>1e-12 else np.full_like(s,.5)
        pred=s[None,:]>=grid[:,None]; denom=pred.sum(1)+y.sum()
        curves.append(np.divide(2*(pred&y).sum(1),denom,out=np.ones(len(grid),float),where=denom>0))
    values=np.average(np.array(curves),axis=0,weights=weights)
    candidates=[(float(value),-abs(h-.5),-h,h) for h,value in zip(grid,values)]
    best=max(candidates); return float(best[3]),float(best[0])

def paired_inference(differences):
    d=np.asarray(differences,float); n=len(d); mean=float(d.mean())
    if n<2: return {"difference":mean,"ci_low":None,"ci_high":None,"p":None,"units":n}
    se=float(stats.sem(d)); radius=float(stats.t.ppf(.975,n-1)*se)
    if n<=16:
        signs=np.array(list(itertools.product([-1.,1.],repeat=n)))
        p=float(np.mean(np.abs(signs@d/n)>=abs(mean)-1e-12))
    else:
        rng=np.random.default_rng(13103); signs=rng.choice([-1.,1.],(100000,n)); null=np.abs(signs@d/n)
        p=float((1+np.sum(null>=abs(mean)-1e-12))/(len(null)+1))
    return {"difference":mean,"ci_low":mean-radius,"ci_high":mean+radius,"p":p,"units":n,"positive_units":int((d>0).sum())}

def holm(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); adjusted=np.empty(len(p)); last=0
    for rank,i in enumerate(order):
        last=max(last,(len(p)-rank)*p[i]); adjusted[i]=min(1.,last)
    return adjusted

def gfm_from_samples(samples):
    """Exact expected-F1 optimizer for the supplied finite joint membership distribution."""
    Y=np.asarray(samples,bool); n=Y.shape[1]; sizes=Y.sum(1)
    best_value=float(np.mean(sizes==0)); best=np.zeros(n,bool)
    for k in range(1,n+1):
        delta=(2*Y/(sizes[:,None]+k)).mean(0)
        order=np.argsort(-delta,kind="stable"); value=float(delta[order[:k]].sum())
        if value>best_value+1e-12:
            best_value=value; best=np.zeros(n,bool); best[order[:k]]=True
    return best,best_value
