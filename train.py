import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import fcluster, linkage
from torch.optim import Adam

from model.Lorentz import manifold
from model.MVHypCSE import AdaptiveSparseGraph, MVHypCSE
from model.losses import compute_align_loss, compute_completion_loss, compute_self_rec_loss, \
    multi_scale_cse
from utils.compute import poincare_to_tangent
from utils.evaluate import evaluate_clustering

warnings.filterwarnings("ignore")

EPS = 1e-8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainOutputs:
    p_emb: np.ndarray
    v_emb: np.ndarray
    tree_linkage: np.ndarray
    epoch: int
    acc: float
    results: dict
    model_state: dict
    adj_states: list



def train_mv_hypcse(
        adj_list,
        X_input_views,
        missing_mask,
        n_classes,
        labels,
        hidden_dim=64,
        embed_dim=32,
        n_layers=2,
        epochs=500,
        lr=1e-3,
        dropout=0.5,
        use_att=False,
        use_res=False,
        lambda_align=1.0,
        lambda_self_rec=0.5,
        lambda_comp=1.0,
        lambda_cse=1.0,
        lambda_pseudo=0.2,
        cse_start_epoch=50,
        comp_observed_weight=0.2,
):
    N = adj_list[0].shape[0]
    V = len(adj_list)
    missing_mask_t = torch.from_numpy(missing_mask).to(device)

    adj_graphs = nn.ModuleList([
        AdaptiveSparseGraph(adj_list[v], device) for v in range(V)
    ])

    feat_dims = [X.shape[1] for X in X_input_views]
    model = MVHypCSE(
        feat_dims, hidden_dim, embed_dim, V, n_classes,
        n_layers=n_layers, dropout=dropout, use_att=use_att, use_res=use_res,
    ).to(device)

    optimizer = Adam(list(model.parameters()) + list(adj_graphs.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("begin train...")
    print(
        f" loss weight : align={lambda_align}, self_rec={lambda_self_rec}, comp={lambda_comp}, cse={lambda_cse}, pseudo={lambda_pseudo}")
    print(f"  GCN: {n_layers}, Dropout: {dropout}, Attention: {use_att}, Residual: {use_res}")

    X_input_tensors = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_input_views]

    acc = -1.0
    epoch = -1
    v_emb = None
    p_emb = None
    Z_link = None
    results = None
    model_state = None
    adj_states = None

    for epoch in range(epochs):
        model.train()
        adj_graphs.train()
        optimizer.zero_grad()

        adj_t_list = [g.get_adj() for g in adj_graphs]
        Z_fused, Z_v_list, H_fused, view_weights = model(X_input_tensors, adj_t_list, missing_mask_t)

        loss_align = compute_align_loss(Z_v_list, missing_mask_t)
        loss_self_rec = compute_self_rec_loss(Z_v_list, X_input_tensors, missing_mask_t, model.decoders)
        loss_comp = compute_completion_loss(
            H_fused, X_input_tensors, missing_mask_t, model.decoders, observed_weight=comp_observed_weight
        )

        warmup_start = cse_start_epoch
        warmup_end = cse_start_epoch + 100
        if epoch >= warmup_start:
            loss_cse = multi_scale_cse(Z_fused, adj_t_list, missing_mask=missing_mask_t, r2_list=(1.2, 2.0, 2.8),
                                       t2=1.0)
            cse_alpha = lambda_cse * min(1.0, (epoch - warmup_start) / max(1, warmup_end - warmup_start))
        else:
            loss_cse = Z_fused.new_tensor(0.0)
            cse_alpha = 0.0

        total_loss = (
                lambda_align * loss_align +
                lambda_self_rec * loss_self_rec +
                lambda_comp * loss_comp +
                cse_alpha * loss_cse
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Epoch {epoch + 1}: loss error")
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(adj_graphs.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        adj_graphs.eval()
        with torch.no_grad():
            adj_t_list_final = [g.get_adj() for g in adj_graphs]
            Z_fused_eval, _, _, _ = model(X_input_tensors, adj_t_list_final, missing_mask_t)
            p_emb_eval = manifold.to_poincare(Z_fused_eval).cpu().numpy()
            v_emb_eval = poincare_to_tangent(manifold.to_poincare(Z_fused_eval)).cpu().numpy()

        try:
            Z_link_eval = linkage(v_emb_eval, method="ward")
            hier_labels_eval = fcluster(Z_link_eval, t=n_classes, criterion="maxclust")
            flat_res_eval = evaluate_clustering(labels, hier_labels_eval)
            acc_eval = flat_res_eval['ACC']
        except Exception as e:
            print(f"Epoch {epoch + 1}: fail ({e})")
            acc_eval = -1.0
            flat_res_eval = {'ACC': -1.0, 'NMI': -1.0, 'ARI': -1.0, 'F-score': -1.0}
            Z_link_eval = None

        print(f"Epoch {epoch + 1}: loss={total_loss:.4f}")

        if acc_eval > acc:
            acc = acc_eval
            epoch = epoch
            p_emb = p_emb_eval.copy()
            v_emb = v_emb_eval.copy()
            if Z_link_eval is not None:
                Z_link = Z_link_eval.copy()
            results = flat_res_eval
            model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adj_states = [{k: v.detach().cpu().clone() for k, v in g.state_dict().items()} for g in adj_graphs]

        model.train()
        adj_graphs.train()


    if model_state is not None:
        model.load_state_dict(model_state)
        for g, s in zip(adj_graphs, adj_states):
            g.load_state_dict(s)

    print(f"\ntrain end: Epoch {epoch + 1}, ACC={acc:.4f}")

    return TrainOutputs(
        p_emb=p_emb if p_emb is not None else p_emb_eval,
        v_emb=v_emb if v_emb is not None else v_emb_eval,
        tree_linkage=Z_link if Z_link is not None else Z_link_eval if 'Z_link_eval' in dir() else None,
        epoch=epoch,
        acc=acc,
        results=results if results is not None else {},
        model_state=model_state if model_state is not None else {},
        adj_states=adj_states if adj_states is not None else [],
    )