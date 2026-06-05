import json


def linkage_to_tree(linkage_mat, n_samples):
    nodes = {i: {"id": int(i), "leaf": True} for i in range(n_samples)}
    for row_idx, row in enumerate(linkage_mat):
        left = int(row[0])
        right = int(row[1])
        dist = float(row[2])
        count = int(row[3])
        node_id = n_samples + row_idx
        nodes[node_id] = {
            "id": int(node_id),
            "leaf": False,
            "distance": dist,
            "count": count,
            "left": nodes[left],
            "right": nodes[right],
        }
    return nodes[n_samples + linkage_mat.shape[0] - 1]


def save_tree_json(tree, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)