import numpy as np

import scipy.sparse as sp

from sklearn.metrics.pairwise import pairwise_distances


def build_knn_graph(features, k=15):
    dist = pairwise_distances(features, metric='cosine')
    n = dist.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(dist, np.inf)
    for i in range(n):
        kk = min(k, n - 1)
        if kk <= 0:
            continue
        idx = np.argpartition(dist[i], kk)[:kk]
        weights = 1.0 - dist[i, idx]
        weights = np.clip(weights, 0, 1)
        adj[i, idx] = weights
    adj = (adj + adj.T) / 2
    adj_sp = sp.csr_matrix(adj)
    adj_sp.setdiag(0)
    adj_sp.eliminate_zeros()
    return adj_sp


def calculate_structural_entropy(adj):
    if sp.issparse(adj):
        degrees = adj.sum(axis=1).A1
    else:
        degrees = adj.sum(axis=1)
    total = degrees.sum()
    if total == 0:
        return 0.0
    prob = degrees[degrees > 0] / total
    return -np.sum(prob * np.log2(prob))


def select_k_by_entropy(features, k_candidates=None):
    n = features.shape[0]
    if n <= 2:
        return 1, 0.0
    if k_candidates is None:
        max_k = min(30, n - 1)
        k_candidates = list(range(5, max_k + 1)) if max_k >= 5 else list(range(1, max_k + 1))

    if len(k_candidates) == 0:
        return 1, 0.0

    entropies = []
    for k in k_candidates:
        adj = build_knn_graph(features, k=k)
        entropies.append(calculate_structural_entropy(adj))

    best_k = k_candidates[-1]
    best_entropy = entropies[-1]
    for i in range(1, len(k_candidates) - 1):
        if entropies[i] < entropies[i - 1] and entropies[i] < entropies[i + 1]:
            best_k = k_candidates[i]
            best_entropy = entropies[i]
            break
    return best_k, best_entropy


def consensus_guided_completion(adj_list, missing_mask, k_list, confidence_threshold=0.15):
    N = adj_list[0].shape[0]
    V = len(adj_list)

    adj_out = []
    for adj in adj_list:
        if sp.issparse(adj):
            adj_out.append(adj.tolil())
        else:
            adj_out.append(sp.lil_matrix(adj))

    for v in range(V):
        missing_nodes = np.where(missing_mask[:, v] == 0)[0]
        if len(missing_nodes) == 0:
            continue

        for i in missing_nodes:
            candidate_scores = {}

            for u in range(V):
                if u == v:
                    continue
                if missing_mask[i, u] == 0:
                    continue

                adj_u = adj_list[u]
                if sp.issparse(adj_u):
                    _, cols = adj_u[i].nonzero()
                    cols = cols.tolist()
                else:
                    cols = np.where(adj_u[i] > 0)[0].tolist()

                for j in cols:
                    if missing_mask[j, u] == 0 or missing_mask[j, v] == 0:
                        continue

                    w = float(adj_u[i, j])
                    if w <= 0:
                        continue

                    if j not in candidate_scores:
                        candidate_scores[j] = [0, 0.0]
                    candidate_scores[j][0] += 1
                    candidate_scores[j][1] += w

            if len(candidate_scores) == 0:
                continue

            candidates = []
            scores = []
            for j, (cnt, wsum) in candidate_scores.items():
                avg_w = wsum / cnt
                support_ratio = cnt / max(V - 1, 1)
                conf = avg_w * (1.0 + support_ratio)
                candidates.append(j)
                scores.append(conf)

            scores = np.array(scores)
            valid_mask = scores >= confidence_threshold
            if valid_mask.sum() == 0:
                continue

            valid_candidates = [candidates[idx] for idx in np.where(valid_mask)[0]]
            valid_scores = scores[valid_mask]

            k = min(k_list[v], len(valid_candidates))
            if len(valid_candidates) > k:
                topk_idx = np.argpartition(valid_scores, -k)[-k:]
            else:
                topk_idx = np.arange(len(valid_candidates))

            max_s = valid_scores.max()
            if max_s > 0:
                for idx in topk_idx:
                    j = valid_candidates[idx]
                    w = float(valid_scores[idx] / max_s)
                    adj_out[v][i, j] = w
                    adj_out[v][j, i] = w

    for v in range(V):
        adj_out[v] = adj_out[v].tocsr()
        adj_out[v] = (adj_out[v] + adj_out[v].T) / 2.0
        adj_out[v].setdiag(0)
        adj_out[v].eliminate_zeros()

    return adj_out
