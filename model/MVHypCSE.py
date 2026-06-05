import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from model.Lorentz import manifold
from utils.compute import euclid_to_lorentz, project_lorentz, \
    tangent_to_poincare, poincare_to_tangent


class AdaptiveSparseGraph(nn.Module):
    def __init__(self, adj_scipy, device):
        super().__init__()
        adj_coo = adj_scipy.tocoo()
        indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col))).to(device)
        values = torch.FloatTensor(adj_coo.data).to(device)

        self.register_buffer('base_indices', indices)
        self.edge_weight = nn.Parameter(values.clone())
        self.N = adj_scipy.shape[0]

    def get_adj(self):
        eye_idx = torch.arange(self.N, device=self.base_indices.device).unsqueeze(0).repeat(2, 1)
        idx = torch.cat([self.base_indices, eye_idx], dim=1)
        w_base = torch.clamp(self.edge_weight, min=1e-6)
        w = torch.cat([w_base, torch.ones(self.N, device=self.edge_weight.device)])

        deg = scatter_sum(w, idx[0], dim_size=self.N).clamp_min(1e-8)
        d_inv_sqrt = torch.pow(deg, -0.5)
        row, col = idx[0], idx[1]
        norm_w = w * d_inv_sqrt[row] * d_inv_sqrt[col]
        return torch.sparse_coo_tensor(idx, norm_w, (self.N, self.N)).coalesce()



class LorentzLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear(self.dropout(x))
        return project_lorentz(h)


class LorentzAgg(nn.Module):
    def __init__(self, in_features, dropout, use_att):
        super().__init__()
        self.use_att = use_att
        self.dropout = nn.Dropout(dropout)
        if use_att:
            self.key_linear = LorentzLinear(in_features, in_features)
            self.query_linear = LorentzLinear(in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(1))
            self.scale = nn.Parameter(torch.ones(1) * math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att = 2 + 2 * manifold.cinner(query, key)
            att = att / self.scale.clamp_min(1e-6) + self.bias
            att = torch.sigmoid(att)
            adj_dense = adj.to_dense()
            support = torch.matmul(adj_dense * att, x)
        else:
            support = torch.sparse.mm(adj, x)
        return project_lorentz(support)


class LorentzGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, use_att=False, use_res=False):
        super().__init__()
        self.linear = LorentzLinear(in_features, out_features, dropout=dropout)
        self.agg = LorentzAgg(out_features, dropout, use_att)
        self.use_res = use_res
        self.res_proj = None
        if use_res and in_features != out_features:
            self.res_proj = LorentzLinear(in_features, out_features, dropout=0.0)

    def forward(self, x, adj):
        h = self.linear(x)
        h = self.agg(h, adj)

        if not self.use_res:
            return h

        x_proj = self.res_proj(x) if self.res_proj is not None else x

        h_p = manifold.to_poincare(h)
        x_p = manifold.to_poincare(x_proj)

        h_tan = poincare_to_tangent(h_p)
        x_tan = poincare_to_tangent(x_p)

        res_tan = h_tan + x_tan

        res_p = tangent_to_poincare(res_tan)
        res = manifold.from_poincare(res_p)
        return project_lorentz(res)


class LorentzGraphEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, n_layers=2, dropout=0.5, use_att=False, use_res=False):
        super().__init__()
        dims = [in_features] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            LorentzGraphConvolution(
                dims[i], dims[i + 1], dropout, use_att,
                use_res=(use_res and i > 0)
            ) for i in range(n_layers)
        ])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x



class MVHypCSE(nn.Module):
    def __init__(
            self,
            feat_dims,
            hidden_dim,
            embed_dim,
            num_views,
            n_classes,
            n_layers=2,
            dropout=0.5,
            use_att=False,
            use_res=False,
    ):
        super().__init__()
        self.V = num_views
        self.C = n_classes
        self.embed_dim = embed_dim

        self.encoders = nn.ModuleList(
            [LorentzGraphEncoder(feat_dims[v] + 1, hidden_dim, embed_dim + 1, n_layers, dropout, use_att, use_res)
             for v in range(num_views)]
        )

        self.decoders = nn.ModuleList(
            [nn.Linear(embed_dim, feat_dims[v]) for v in range(num_views)]
        )

        self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)

        #  Mask Token
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(feat_dims[v]) * 0.01) for v in range(num_views)
        ])

    def forward(self, X_views, adj_list, missing_mask):
        Z_v_list = []
        for v in range(self.V):
            x = X_views[v]

            if missing_mask is not None:
                m = missing_mask[:, v].unsqueeze(1)
                x = torch.where(m == 1, x, self.mask_tokens[v])

            x_in = euclid_to_lorentz(x)
            z = self.encoders[v](x_in, adj_list[v])
            z = project_lorentz(z)

            p = manifold.to_poincare(z)
            p = F.normalize(p, p=2, dim=-1) * 0.999
            z = manifold.from_poincare(p)
            z = project_lorentz(z)

            Z_v_list.append(z)

        weights = F.softmax(self.view_weights, dim=0)
        Z_fused = manifold.Frechet_mean(torch.stack(Z_v_list), weights=weights)
        Z_fused = project_lorentz(Z_fused)

        H_fused = poincare_to_tangent(manifold.to_poincare(Z_fused))

        return Z_fused, Z_v_list, H_fused, weights
