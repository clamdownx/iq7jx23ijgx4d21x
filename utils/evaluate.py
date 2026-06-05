import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import spectral_clustering
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, accuracy_score

warnings.filterwarnings("ignore")



def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla1 * ind_cla2)
    m = linear_sum_assignment(-G)
    m = np.array(m).T
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        for j in m:
            if j[1] == i:
                newL2[L2 == Label2[i]] = Label1[j[0]]
    return newL2


def evaluate_clustering(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    label_map = {pred: true for pred, true in zip(ind[0], ind[1])}
    y_pred_aligned = np.array([label_map[l] for l in y_pred])
    acc = sum(w[i, j] for i, j in zip(*ind)) * 1.0 / y_pred.size
    nmi = normalized_mutual_info_score(y_true, y_pred_aligned)
    ari = adjusted_rand_score(y_true, y_pred_aligned)
    f1 = f1_score(y_true, y_pred_aligned, average="macro")
    return {"ACC": acc, "NMI": nmi, "ARI": ari, "F-score": f1}


def cluster_and_evaluate(adj, true_labels, n_clusters, n_init=10):
    if sp.issparse(adj):
        adj = adj.toarray()
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) / 2

    row_sum = adj.sum(axis=1)
    if np.any(row_sum == 0):
        isolated = np.where(row_sum == 0)[0]
        for i in isolated:
            adj[i, i] = 1.0

    try:
        sc = spectral_clustering(n_clusters=n_clusters, affinity='precomputed',
                                 random_state=0, n_init=n_init, assign_labels='discretize')
        pred = sc.fit_predict(adj)
    except Exception as e:
        print(f"  failed: {e}")
        return 0.0, 0.0

    nmi = normalized_mutual_info_score(true_labels, pred)
    pred_mapped = best_map(true_labels, pred)
    acc = accuracy_score(true_labels, pred_mapped)
    return nmi, acc
