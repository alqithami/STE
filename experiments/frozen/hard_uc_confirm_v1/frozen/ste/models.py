import torch
from torch import nn

class PairModel(nn.Module):
    def __init__(self,hidden=32,relational=False):
        super().__init__(); self.relational=relational
        self.encoder=nn.Sequential(nn.Linear(16,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.pair=nn.Linear(hidden,1)
        self.head=nn.Sequential(nn.Linear(2*hidden+(4 if relational else 0),hidden),nn.Tanh(),nn.Linear(hidden,1))

    def forward(self,W,T):
        B,n,_=W.shape; mask=~torch.eye(n,dtype=torch.bool,device=W.device); m=W+W.transpose(1,2)
        p=(W+.5)/(m+1); e=(p-.5)*mask; counts=torch.log1p(m)/6
        tie_fraction=T/(m+T).clamp_min(1); observed=((m+T)>0).float()*mask
        nodes=torch.stack([e.sum(-1)/(n-1),e.abs().sum(-1)/(n-1),
                           (counts*mask).sum(-1)/(n-1),((m>0)*mask).float().sum(-1)/(n-1),
                           (tie_fraction*mask).sum(-1)/(n-1),observed.sum(-1)/(n-1)],-1)
        edge=torch.stack([e,counts*mask,tie_fraction*mask,observed],-1)
        feat=torch.cat([edge,nodes[:, :,None,:].expand(B,n,n,6),nodes[:,None,:,:].expand(B,n,n,6)],-1)
        h=self.encoder(feat); z=self.pair(h).squeeze(-1); P=torch.sigmoid(z-z.transpose(1,2))
        node=torch.cat([(h*mask[None,:,:,None]).sum(2)/(n-1),(h*mask[None,:,:,None]).sum(1)/(n-1)],-1)
        if self.relational:
            E=(P-.5)*mask
            extra=torch.stack([E.sum(-1)/(n-1),E.abs().sum(-1)/(n-1),
                               ((P*mask)@(P*mask)).sum(-1)/(n-1)**2,
                               (((P*mask)@(P*mask)).transpose(1,2)).sum(-1)/(n-1)**2],-1)
            node=torch.cat([node,extra],-1)
        return P,torch.sigmoid(self.head(node).squeeze(-1))

def pair_loss(P,W):
    n=P.shape[-1]; upper=torch.triu(torch.ones((n,n),device=P.device),diagonal=1)
    m=W+W.transpose(1,2); p=P.clamp(1e-6,1-1e-6)
    return (-(W*torch.log(p)+W.transpose(1,2)*torch.log1p(-p))*upper).sum()/(m*upper).sum().clamp_min(1)
