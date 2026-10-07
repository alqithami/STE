from __future__ import annotations
import math
import numpy as np
import torch

def adjacency(P):
    A=np.asarray(P)>.5
    np.fill_diagonal(A,False)
    return A

def hard_core(A, target):
    A=np.asarray(A,dtype=bool).copy(); np.fill_diagonal(A,False); n=len(A)
    if target=="tc":
        R=A.copy(); np.fill_diagonal(R,True)
        for k in range(n): R |= R[:,k,None] & R[k,None,:]
        return R.all(1)
    if target!="uc": raise ValueError(target)
    y=np.ones(n,dtype=bool)
    for a in range(n):
        for c in range(n):
            if c!=a and A[c,a] and not np.any(A[a] & ~A[c]):
                y[a]=False; break
    return y

def hard_core_independent(A, target):
    """Graph traversal TC and set-inclusion UC, independent of the matrix oracle."""
    n=len(A); wins=[{j for j in range(n) if A[i,j] and j!=i} for i in range(n)]
    out=[]
    for i in range(n):
        if target=="uc": out.append(not any(i in wins[j] and wins[i]<=wins[j] for j in range(n) if j!=i))
        else:
            visited={i}; stack=[i]
            while stack:
                for j in wins[stack.pop()] - visited: visited.add(j); stack.append(j)
            out.append(len(visited)==n)
    return np.array(out,dtype=bool)

def softmax_mean(x, gamma, dim):
    return gamma*(torch.logsumexp(x/gamma, dim=dim)-math.log(x.shape[dim]))

def softmin_mean(x, gamma, dim):
    return -softmax_mean(-x,gamma,dim)

def soft_edges(P, tau):
    mask=~torch.eye(P.shape[-1],dtype=torch.bool,device=P.device)
    return torch.sigmoid((P-.5)/tau)*mask

def soft_uc(P, tau=.035, gamma=.035):
    """Canonical manuscript Eqs. (2),(7)-(9), streamed over target a."""
    single=P.ndim==2
    if single: P=P[None]
    D=soft_edges(P,tau); B,n,_=D.shape; ids=torch.arange(n,device=P.device); out=[]
    for a in range(n):
        cs=ids[ids!=a]
        if n==2: witness=torch.zeros((B,1),dtype=P.dtype,device=P.device)
        else:
            products=D[:,a,:].unsqueeze(1)*(1-D[:,cs,:])
            valid=(ids[None,:]!=a)&(ids[None,:]!=cs[:,None])
            witness=softmax_mean(products[:,valid].reshape(B,n-1,n-2),gamma,-1)
        cover=D[:,cs,a]*(1-witness)
        out.append(1-softmax_mean(cover,gamma,-1))
    result=torch.stack(out,dim=-1)
    return result[0] if single else result

def _tc_step(Q,D,gamma):
    # Shape B,a,c,b; no stack of the two scalar min arguments is materialized.
    candidates=-gamma*(torch.logaddexp(-Q.unsqueeze(-1)/gamma,-D.unsqueeze(1)/gamma)-math.log(2))
    return softmax_mean(candidates,gamma,2)

def soft_tc(P,tau=.035,gamma=.035,K=None,checkpoint=False):
    """Canonical Eqs. (2)-(6), full K=n-1 unless explicitly bounded."""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    single=P.ndim==2
    if single: P=P[None]
    n=P.shape[-1]; K=n-1 if K is None else int(K)
    if K<1: raise ValueError("K must be positive")
    D=soft_edges(P,tau); Q=D; L=D/gamma
    for _ in range(1,K):
        if checkpoint and torch.is_grad_enabled() and D.requires_grad:
            Q=torch_checkpoint(_tc_step,Q,D,gamma,use_reentrant=False)
        else: Q=_tc_step(Q,D,gamma)
        L=torch.logaddexp(L,Q/gamma)
    R=gamma*(L-math.log(K))
    mask=~torch.eye(n,dtype=torch.bool,device=P.device)
    out=softmin_mean(R[:,mask].reshape(P.shape[0],n,n-1),gamma,-1)
    return out[0] if single else out

def soft_core(P,target,temperature=.035,checkpoint=False,K=None):
    if target=="uc": return soft_uc(P,temperature,temperature)
    if target=="tc": return soft_tc(P,temperature,temperature,K,checkpoint)
    raise ValueError(target)

def numpy_reference(P,target,tau=.035,gamma=.035,K=None):
    """Slow independent forward formula, for parity checks only."""
    from scipy.special import expit, logsumexp
    P=np.asarray(P); n=len(P); D=expit((P-.5)/tau); np.fill_diagonal(D,0)
    mx=lambda x: gamma*(logsumexp(np.asarray(x)/gamma)-np.log(len(x)))
    mn=lambda x: -mx(-np.asarray(x))
    if target=="uc":
        out=[]
        for a in range(n):
            cov=[]
            for c in range(n):
                if c==a: continue
                w=[D[a,b]*(1-D[c,b]) for b in range(n) if b not in (a,c)]
                cov.append(D[c,a]*(1-(mx(w) if w else 0)))
            out.append(1-mx(cov))
        return np.array(out)
    K=n-1 if K is None else K; Q=D.copy(); paths=[Q.copy()]
    for _ in range(1,K):
        Q=np.array([[mx([mn([Q[a,c],D[c,b]]) for c in range(n)]) for b in range(n)] for a in range(n)])
        paths.append(Q.copy())
    R=np.array([[mx([q[a,b] for q in paths]) for b in range(n)] for a in range(n)])
    return np.array([mn([R[a,b] for b in range(n) if b!=a]) for a in range(n)])

def sampled_strict_cores(A):
    """GPU Boolean closure for strict tournament samples; UC equals 2-kings."""
    n=A.shape[-1]; eye=torch.eye(n,dtype=torch.bool,device=A.device)[None]
    two=(A.float()@A.float())>0
    uc=(A|two|eye).all(-1)
    R=A|eye
    for _ in range(math.ceil(math.log2(max(1,n-1)))):
        R=R|((R.float()@R.float())>0)
    tc=R.all(-1)
    return {"uc":uc,"tc":tc}
