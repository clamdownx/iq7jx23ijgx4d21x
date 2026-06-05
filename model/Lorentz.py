import torch
from utils.compute import minkowski_pairwise_inner, project_lorentz, minkowski_rowwise_inner

EPS = 1e-8

class Lorentz:
    def __init__(self, k=1.0, learnable=False):
        self.k = k
        self.learnable = learnable

    def cinner(self, x, y):
        return minkowski_pairwise_inner(x, y)

    def inner(self, x, y, keepdim=False):
        out = minkowski_rowwise_inner(x, y)
        if not keepdim:
            out = out.squeeze(-1)
        return out

    def to_poincare(self, x):
        dn = x.size(-1) - 1
        return x.narrow(-1, 1, dn) / (x.narrow(-1, 0, 1) + 1.0)

    def from_poincare(self, x, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=-1, keepdim=True)
        return torch.cat((1 + x_norm_square, 2 * x), dim=-1) / (1.0 - x_norm_square + eps)

    def Frechet_mean(self, embeddings, weights=None, keepdim=False):
        if weights is None:
            w = torch.ones(embeddings.size(0), device=embeddings.device, dtype=embeddings.dtype)
            w = w / w.sum()
        else:
            w = weights / weights.sum().clamp_min(EPS)
        while w.dim() < embeddings.dim():
            w = w.unsqueeze(-1)
        z = torch.sum(embeddings * w, dim=0, keepdim=keepdim)
        return project_lorentz(z)


manifold = Lorentz()