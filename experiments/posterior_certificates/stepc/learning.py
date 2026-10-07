"""Observed-count-only, permutation-equivariant predictive models.

The learned tournament laws are approximate predictive distributions. They are
neither calibrated probabilities nor exact Bayesian posteriors. ``q`` denotes
the probability of a majority *orientation*, not a matchup probability P.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FEATURE_VERSION = "observed-count-equivariant-v2-normalized-upper-edge-law"
MODEL_KINDS = ("ordinary", "relational", "learned_edge", "posterior_mixture")


def mixture_orientation_probabilities_torch(components, weights):
    """Differentiable float64 marginal of the native normalized sampling law.

    Return a reciprocal q with an exact .5 diagonal. Keep components/weights
    unchanged: the existing sampler's upper-edge probabilities and normalized
    component draws define the law. V2 changes future fitting numerics and must
    never be substituted for replay of historical v1 checkpoints.
    """
    if (components.ndim != 3 or components.shape[-1] != components.shape[-2]
            or components.shape[-1] < 1 or weights.shape != (len(components),)):
        raise ValueError("Invalid mixture shapes")
    if (not bool(torch.isfinite(components).all())
            or bool(((components < 0) | (components > 1)).any())):
        raise ValueError("Invalid component probabilities")
    total = weights.to(torch.float64).sum()
    if (not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())
            or not bool(torch.isfinite(total)) or not bool(total > 0)):
        raise ValueError("Invalid mixture weights")
    if not bool(torch.allclose(components + components.transpose(-1, -2),
                              torch.ones_like(components), atol=1e-6, rtol=0)):
        raise ValueError("Nonreciprocal component probabilities")
    normalized = weights.to(torch.float64)
    normalized = normalized / total
    n = components.shape[-1]
    i, j = torch.triu_indices(n, n, 1, device=components.device)
    upper = (normalized[:, None] * components[:, i, j].to(torch.float64)).sum(0).clamp(0., 1.)
    result = torch.full((n, n), .5, dtype=torch.float64, device=components.device)
    result[i, j], result[j, i] = upper, 1. - upper
    return result


def observed_tensors(view, device="cpu"):
    """Deliberately accepts only observed fields; never accesses reference truth."""
    return (torch.as_tensor(np.asarray(view["W"]), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(view["T"]), dtype=torch.float32, device=device))


def count_features(W, T):
    """Shared node and oriented-edge features, with no candidate identifiers."""
    n = W.shape[-1]
    if W.ndim != 2 or W.shape != T.shape or W.shape[0] != n:
        raise ValueError("A view must contain square W and T of the same shape")
    if not bool(torch.isfinite(W).all() and torch.isfinite(T).all()):
        raise ValueError("Nonfinite observed counts")
    decisive = W + W.T
    p = (W + .5) / (decisive + 1.)
    margin = (W - W.T) / (decisive + 1.)
    eye = torch.eye(n, dtype=torch.bool, device=W.device)
    valid = (~eye).float()
    den = max(n - 1, 1)
    count = torch.log1p(decisive)
    tie = T / (decisive + T + 1.)
    coverage = decisive / (decisive + 1.)
    pair = torch.stack((p-.5, margin, count, tie, coverage), -1) * valid[..., None]
    def mean(v):
        return (v * valid).sum(-1) / den
    nodes = torch.stack((mean(p), mean(margin), mean(margin.square()),
                         mean(count), mean(tie), mean(coverage),
                         mean(p.square()), mean((p > .5).float())), -1)
    return nodes, pair


class PredictiveModel(nn.Module):
    """Direct heads or a reciprocal edge law with a whole-tournament mixture."""
    def __init__(self, kind, hidden=32):
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(kind)
        self.kind, self.hidden = kind, int(hidden)
        self.encoder = nn.Sequential(nn.Linear(8, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        if kind == "relational":
            self.message = nn.Sequential(nn.Linear(2*hidden+5, hidden), nn.Tanh())
            self.direct = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        elif kind == "ordinary":
            self.direct = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        else:
            self.components = 2 if kind == "posterior_mixture" else 1
            self.pair = nn.Sequential(nn.Linear(2*hidden+5, hidden), nn.Tanh(),
                                      nn.Linear(hidden, self.components))
            if self.components == 2:
                self.mixture = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, W, T):
        x, e = count_features(W, T)
        h = self.encoder(x)
        n = W.shape[0]
        hi = h[:, None, :].expand(n, n, -1)
        hj = h[None, :, :].expand(n, n, -1)
        pair_input = torch.cat((hi, hj, e), -1)
        if self.kind == "ordinary":
            return {"scores": torch.sigmoid(self.direct(h).squeeze(-1))}
        if self.kind == "relational":
            valid = (~torch.eye(n, dtype=torch.bool, device=W.device)).float()
            messages = self.message(pair_input) * valid[..., None]
            context = messages.sum(1) / max(n-1, 1)
            return {"scores": torch.sigmoid(self.direct(torch.cat((h, context), -1)).squeeze(-1))}
        raw = self.pair(pair_input)
        logits = (raw - raw.transpose(0, 1)).permute(2, 0, 1)
        qs = torch.sigmoid(logits)
        if self.components == 2:
            pooled = torch.cat((h.mean(0), h.square().mean(0)))
            mixture_logits = self.mixture(pooled)
            logweights = torch.log_softmax(mixture_logits, -1)
            weights = logweights.exp()
        else:
            weights = torch.ones(1, device=W.device)
            logweights = torch.zeros(1, device=W.device)
        return {"q": mixture_orientation_probabilities_torch(qs, weights),
                "components": qs, "weights": weights, "logits": logits,
                "logweights": logweights}


def hard_uc_torch(A):
    """Exact hard UC for a batch of strict tournaments, with no relaxation."""
    from .posterior import exact_uc
    return exact_uc(A, validate=True)


def f1_reward(predicted, truth):
    truth = truth.bool()
    intersection = (predicted & truth).sum(-1).float()
    denominator = predicted.sum(-1).float() + truth.sum(-1).float()
    return torch.where(denominator > 0, 2*intersection / denominator.clamp_min(1),
                       torch.ones_like(denominator))


def score_function_uc_loss(output, truth, draws=64, generator=None):
    """Unbiased likelihood-ratio gradient of negative expected hard-UC F1.

    One global component is sampled per WHOLE tournament. The leave-one-out
    reward baseline is independent of the scored draw; the ordinary sample mean
    baseline would bias this gradient without a correction. Rewards and this
    baseline are frozen; log probabilities retain their gradients.
    """
    if draws < 2:
        raise ValueError("At least two draws are needed for the leave-one-out baseline")
    q, weights = output["components"], output["weights"]
    n = q.shape[-1]
    ij = torch.triu_indices(n, n, 1, device=q.device)
    component = torch.multinomial(weights.detach(), draws, replacement=True, generator=generator)
    probabilities = q[component][:, ij[0], ij[1]]
    edges = torch.rand(probabilities.shape, device=q.device, generator=generator) < probabilities.detach()
    A = torch.zeros((draws, n, n), dtype=torch.bool, device=q.device)
    A[:, ij[0], ij[1]] = edges
    A[:, ij[1], ij[0]] = ~edges
    rewards = f1_reward(hard_uc_torch(A), truth).detach()
    baseline = (rewards.sum() - rewards) / (draws - 1)
    selected_logits = output["logits"][component][:, ij[0], ij[1]]
    edge_logprob = -F.binary_cross_entropy_with_logits(selected_logits, edges.float(), reduction="none").sum(-1)
    logprob = edge_logprob + output["logweights"][component]
    return -((rewards - baseline).detach() * logprob).mean(), rewards.mean().item()


def supervised_loss(model, case, device, weight=0., draws=64, generator=None):
    W, T = observed_tensors(case, device)
    output = model(W, T)
    y = torch.as_tensor(np.asarray(case["y"]), dtype=torch.float32, device=device)
    if model.kind in ("ordinary", "relational"):
        loss = F.binary_cross_entropy(output["scores"].clamp(1e-7, 1-1e-7), y)
        return loss, {"supervised_uc_bce": float(loss.detach()), "sampled_f1": None}
    n = W.shape[0]
    ij = torch.triu_indices(n, n, 1, device=W.device)
    labels = torch.as_tensor(np.asarray(case["fullA"]), dtype=torch.float32, device=device)
    bce = F.binary_cross_entropy(output["q"][ij[0], ij[1]].clamp(1e-7, 1-1e-7),
                                 labels[ij[0], ij[1]].to(output["q"].dtype))
    reward = None
    loss = bce
    if model.kind == "posterior_mixture":
        reinforce, reward = score_function_uc_loss(output, y, draws, generator)
        loss = bce + weight * reinforce
    return loss, {"orientation_bce": float(bce.detach()), "sampled_f1": reward}


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(arr.shape).encode()); digest.update(arr.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def predict_observed(model, view, device="cpu"):
    model.eval()
    output = model(*observed_tensors(view, device))
    result = {k: v.detach().cpu().numpy() for k, v in output.items()}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise RuntimeError("Nonfinite predictive output")
    return result
