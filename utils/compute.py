import torch

EPS = 1e-8



def minkowski_rowwise_inner(x, y):
    return -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def minkowski_pairwise_inner(x, y):
    x_time, x_space = x[:, :1], x[:, 1:]
    y_time, y_space = y[:, :1], y[:, 1:]
    return -x_time @ y_time.T + x_space @ y_space.T


def project_lorentz(x):
    space = x[..., 1:]
    time = torch.sqrt(torch.clamp(1.0 + torch.sum(space * space, dim=-1, keepdim=True), min=1.0 + EPS))
    return torch.cat([time, space], dim=-1)


def euclid_to_lorentz(x):
    zero = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
    return project_lorentz(torch.cat([zero, x], dim=-1))


def lorentz_dist(x, y):
    ip = -minkowski_rowwise_inner(x, y)
    ip = torch.clamp(ip, min=1.0 + 1e-5)
    return torch.acosh(ip).squeeze(-1)


def poincare_to_tangent(p):
    norm = torch.norm(p, dim=-1, keepdim=True).clamp_min(EPS)
    norm = torch.clamp(norm, max=1.0 - 1e-5)
    scale = 2.0 * torch.atanh(norm) / norm
    return scale * p


def tangent_to_poincare(v):
    norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.tanh(norm / 2.0) * v / norm


def poincare_dist_batch(u, v, eps=1e-5):
    sqnorm_u = torch.sum(u * u, dim=-1, keepdim=True)
    sqnorm_v = torch.sum(v * v, dim=-1, keepdim=True).t()
    sqdist = torch.cdist(u, v, p=2).pow(2)
    denom = torch.clamp(1 - sqnorm_u, min=eps) * torch.clamp(1 - sqnorm_v, min=eps)
    dist = torch.acosh(torch.clamp(1 + 2 * sqdist / denom, min=1 + eps))
    return dist