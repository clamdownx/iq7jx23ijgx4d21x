import math
import warnings

import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic import manifold
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_scatter import scatter_sum

from model.Lorentz import manifold
from utils.compute import lorentz_dist, poincare_dist_batch

EPS = 1e-8

warnings.filterwarnings("ignore")




def compute_align_loss(Z_v_list, missing_mask=None, min_overlap=2):
    V = len(Z_v_list)
    device_ = Z_v_list[0].device
    loss = torch.tensor(0.0, device=device_)
    pair_cnt = 0

    if missing_mask is None:
        for v in range(V):
            for w in range(v + 1, V):
                dist = lorentz_dist(Z_v_list[v], Z_v_list[w])
                loss = loss + (dist / (1.0 + dist)).pow(2).mean()
                pair_cnt += 1
        return loss / max(pair_cnt, 1)

    mask_t = missing_mask.to(device_)
    for v in range(V):
        for w in range(v + 1, V):
            valid = (mask_t[:, v] > 0) & (mask_t[:, w] > 0)
            if valid.sum().item() < min_overlap:
                continue
            dist = lorentz_dist(Z_v_list[v][valid], Z_v_list[w][valid])
            loss = loss + (dist / (1.0 + dist)).pow(2).mean()
            pair_cnt += 1

    if pair_cnt == 0:
        return torch.tensor(0.0, device=device_)
    return loss / pair_cnt


def compute_self_rec_loss(Z_v_list, X_targets, missing_mask, decoders):
    loss = torch.tensor(0.0, device=Z_v_list[0].device)
    total = 0
    for v in range(len(Z_v_list)):
        exist_mask = missing_mask[:, v] == 1
        n_exist = int(exist_mask.sum().item())
        if n_exist == 0:
            continue
        z_space = manifold.to_poincare(Z_v_list[v])
        x_rec = decoders[v](z_space)
        loss = loss + F.mse_loss(x_rec[exist_mask], X_targets[v][exist_mask])
        total += 1
    return loss / max(total, 1)


def compute_completion_loss(H_fused, X_targets, missing_mask, decoders, observed_weight=0.2):
    loss = torch.tensor(0.0, device=H_fused.device)
    total = 0

    for v in range(len(X_targets)):
        x_rec = decoders[v](H_fused)
        target = X_targets[v]
        mask_v = missing_mask[:, v].to(H_fused.device)

        observed = (mask_v == 1).float().unsqueeze(1)

        if observed.sum() == 0:
            continue

        weights = observed * observed_weight

        mse = ((x_rec - target) ** 2) * weights
        loss = loss + mse.sum() / weights.sum().clamp_min(EPS) / target.shape[1]
        total += 1

    return loss / max(total, 1)


def compute_cse_loss(Z_fused, adj_list, missing_mask=None, t2=1.0, r2=2.0):
    p = manifold.to_poincare(Z_fused)
    V = len(adj_list)
    total_loss = torch.tensor(0.0, device=Z_fused.device)
    N = p.size(0)
    max_entropy = math.log2(max(N, 2))

    for v in range(V):
        adj = adj_list[v].coalesce()
        edge_index = adj.indices()
        edge_weight = adj.values()

        if edge_index.size(1) == 0:
            continue

        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        node_degrees = scatter_sum(edge_weight, edge_index[0], dim_size=N).clamp_min(EPS)
        vol_G = node_degrees.sum().clamp_min(EPS)

        n_edges = edge_index.size(1)
        sample_budget = min(n_edges, max(2048, int(2 * N)))
        if n_edges > sample_budget:
            sample_idx = torch.randperm(n_edges, device=Z_fused.device)[:sample_budget]
            edge_index = edge_index[:, sample_idx]
            edge_weight = edge_weight[sample_idx]

        i_idx, j_idx = edge_index[0], edge_index[1]
        pi, pj = p[i_idx], p[j_idx]

        dist_ik = poincare_dist_batch(pi, p)
        dist_jk = poincare_dist_batch(pj, p)

        logits = -(dist_ik + dist_jk - r2) / max(t2, EPS)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(logits.size(0), device=Z_fused.device), i_idx] = False
        mask[torch.arange(logits.size(0), device=Z_fused.device), j_idx] = False
        logits = logits.masked_fill(~mask, -1e9)

        weights = torch.softmax(logits, dim=-1)

        community_volumes = torch.sum(weights * node_degrees.unsqueeze(0), dim=-1)
        community_volumes = community_volumes + node_degrees[i_idx] + node_degrees[j_idx]
        community_volumes = community_volumes.clamp_min(1.0)

        log_vol_ratio = torch.log2(community_volumes / vol_G)

        if missing_mask is not None:
            mm = missing_mask[:, v].to(Z_fused.device).float()
            edge_conf = 0.25 + 0.75 * 0.5 * (mm[i_idx] + mm[j_idx])
        else:
            edge_conf = torch.ones_like(edge_weight)

        weighted_edge = edge_weight * edge_conf
        se_loss = -torch.sum(weighted_edge * log_vol_ratio) / weighted_edge.sum().clamp_min(EPS)

        total_loss = total_loss + se_loss / max_entropy

    return total_loss / V


def multi_scale_cse(Z_fused, adj_list, missing_mask=None, r2_list=(1.2, 2.0, 2.8), t2=1.0):
    loss = Z_fused.new_tensor(0.0)
    for r2 in r2_list:
        loss = loss + compute_cse_loss(Z_fused, adj_list, missing_mask=missing_mask, t2=t2, r2=r2)
    return loss / len(r2_list)
