from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from scipy.special import expit
from .common import seed_for, atomic_npz, sha
from .operators import hard_core, hard_core_independent

def latent_graph(n,family,rng):
    ii,jj=np.triu_indices(n,1)
    if family=="rank_mixture":
        utilities=rng.normal(0,1.8,(5,n)); q=rng.dirichlet(np.ones(5))
        P=sum(q[c]*expit(utilities[c,:,None]-utilities[c,None,:]) for c in range(5))
        # Exact ties have probability zero; retain a recorded tiny reciprocal jitter if one occurs.
        P=np.clip(P,.02,.98)
    elif family=="ordered":
        utilities=np.linspace(1.8,-1.8,n)+rng.normal(0,.05,n)
        P=expit(utilities[:,None]-utilities[None,:])
    else:
        A=np.zeros((n,n),bool)
        if family=="random": A[ii,jj]=rng.random(len(ii))>.5; A[jj,ii]=~A[ii,jj]
        elif family=="planted":
            possible=np.arange(3,max(4,n),2); s=int(rng.choice(possible[possible<=n]))
            for a in range(s):
                for b in range(s): A[a,b]=0<(b-a)%s<=s//2
            A[:s,s:]=True
            for a in range(s,n): A[a,a+1:]=True
        elif family=="separated":
            # The exact 4-vertex example in manuscript Fig. 2, dominating all outsiders.
            for a,b in [(0,1),(1,2),(2,0),(0,3),(1,3),(3,2)]: A[a,b]=True
            A[:4,4:]=True
            for a in range(4,n): A[a,a+1:]=True
        elif family=="long_path":
            # Strongly connected tournament: consecutive edges forward, all longer edges backward.
            # Vertex 0 reaches n-1 only along a path of length n-1.
            for a in range(n):
                for b in range(a+1,n): A[a,b]=(b==a+1); A[b,a]=(b!=a+1)
        else: raise ValueError(family)
        margins=rng.uniform(.04,.35,(n,n)); margins=np.triu(margins,1); margins+=margins.T
        P=.5+(2*A.astype(float)-1)*margins; np.fill_diagonal(P,.5)
    perm=rng.permutation(n); P=P[perm][:,perm]
    np.fill_diagonal(P,.5)
    if np.any(np.isclose(P[ii,jj],.5,atol=0,rtol=0)): raise ValueError("Exact latent majority tie")
    return P

def sample_observations(P,rng,budget,missing,tie_probability):
    n=len(P); W=np.zeros((n,n),np.int32); T=np.zeros_like(W); ii,jj=np.triu_indices(n,1)
    hetero=rng.lognormal(-.5*.7**2,.7,len(ii))
    totals=1+rng.poisson(budget*hetero)
    totals[rng.random(len(ii))<missing]=0
    ties=rng.binomial(totals,tie_probability); m=totals-ties
    wins=rng.binomial(m,P[ii,jj]); W[ii,jj]=wins; W[jj,ii]=m-wins
    T[ii,jj]=ties; T[jj,ii]=ties
    return W,T

def make_split(path,cfg,collection,phase,n,count):
    path=Path(path)
    if path.exists():
        z=np.load(path)
        return {k:z[k] for k in z.files}
    lc=cfg["learning"]; out={k:[] for k in ["P","W","T","y_uc","y_tc","family","case_id"]}
    for i in range(count):
        family=lc["families"][i%len(lc["families"])]
        case_id=f"c{collection:02d}/{phase}/n{n}/{i:05d}"
        rng=np.random.default_rng(seed_for(cfg["seed"],"fresh-data",case_id))
        P=latent_graph(n,family,rng); W,T=sample_observations(P,rng,lc["count_intensity"],lc["missing_probability"],lc["tie_probability"])
        A=P>.5; np.fill_diagonal(A,False)
        for target in ("uc","tc"):
            y=hard_core(A,target)
            if not np.array_equal(y,hard_core_independent(A,target)): raise RuntimeError("Truth-oracle disagreement")
            out["y_"+target].append(y)
        for k,v in (("P",P.astype(np.float64)),("W",W),("T",T),("family",family),("case_id",case_id)): out[k].append(v)
    out={k:np.array(v) for k,v in out.items()}; atomic_npz(path,**out)
    return out

def parse_profile(path):
    """PrefLib new ordinal format. Unranked alternatives remain unobserved."""
    text=Path(path).read_text(encoding="utf-8-sig"); meta={}; records=[]
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        if line.startswith("#"):
            if ":" in line: k,v=line[1:].split(":",1); meta[k.strip()]=v.strip()
            continue
        mult, order=line.split(":",1); groups=[]
        for token in re.findall(r"\{[^}]*\}|[+-]?\d+",order):
            groups.append([int(x) for x in re.findall(r"[+-]?\d+",token)])
        records.append((int(mult),groups))
    n=int(meta["NUMBER ALTERNATIVES"]); voters=int(meta["NUMBER VOTERS"])
    ids=sorted(int(k.split(" ")[-1]) for k in meta if k.startswith("ALTERNATIVE NAME "))
    if len(ids)!=n: raise ValueError("Missing alternative IDs")
    index={a:i for i,a in enumerate(ids)}; ranks=[]; multiplicity=[]
    for mult,groups in records:
        if mult<1: raise ValueError("Nonpositive ballot multiplicity")
        r=np.full(n,-1,np.int16); seen=set()
        for rank,group in enumerate(groups):
            for a in group:
                if a in seen or a not in index: raise ValueError("Invalid alternative in ballot")
                seen.add(a); r[index[a]]=rank
        ranks.append(r); multiplicity.append(mult)
    counts=np.asarray(multiplicity,np.int64); ranks=np.asarray(ranks)
    if counts.sum()!=voters: raise ValueError("Voter count differs from ballot multiplicities")
    if "NUMBER UNIQUE ORDERS" in meta and len(records)!=int(meta["NUMBER UNIQUE ORDERS"]): raise ValueError("Order count mismatch")
    if meta.get("DATA TYPE") not in ("soc","soi","toc","toi"): raise ValueError("Unsupported profile type")
    return {"meta":meta,"ranks":ranks,"multiplicity":counts,"n":n,"voters":voters,"alternative_ids":ids}

def ballot_counts(profile,multiplicity):
    ranks=profile["ranks"]; n=profile["n"]; mult=np.asarray(multiplicity,np.int64)
    W=np.zeros((n,n),np.int64); T=np.zeros_like(W)
    for a in range(n):
        for b in range(a+1,n):
            seen=(ranks[:,a]>=0)&(ranks[:,b]>=0)
            W[a,b]=mult[seen&(ranks[:,a]<ranks[:,b])].sum()
            W[b,a]=mult[seen&(ranks[:,b]<ranks[:,a])].sum()
            T[a,b]=T[b,a]=mult[seen&(ranks[:,a]==ranks[:,b])].sum()
    return W,T

def nested_subsamples(profile,fractions,rng):
    remaining=profile["multiplicity"].copy(); used=np.zeros_like(remaining); previous=0; out={}
    for fraction in sorted(fractions):
        take=max(1,int(np.floor(fraction*profile["voters"])))
        draw=rng.multivariate_hypergeometric(remaining,take-previous)
        used+=draw; remaining-=draw; previous=take
        W,T=ballot_counts(profile,used)
        out[fraction]=(W,T,used.copy())
    return out
