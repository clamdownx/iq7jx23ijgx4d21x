import argparse
import json
import csv
import os
import traceback
import warnings
from datetime import datetime, timedelta
import time
import numpy as np
import scipy.sparse as sp
import torch
from scipy.cluster.hierarchy import fcluster

from train import train_mv_hypcse
from utils.coding_tree import linkage_to_tree, save_tree_json
from utils.config import get_config
from utils.dataload import get_multiview_data, missing
from utils.evaluate import evaluate_clustering
from utils.graph_construct import select_k_by_entropy, build_knn_graph, consensus_guided_completion
import random

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

EPS = 1e-8


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()


def run_single(args, incomplete_views, missing_mask, labels):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    V = len(incomplete_views)
    N = incomplete_views[0].shape[0]
    C = len(np.unique(labels))

    print(f"\n{'=' * 60}")
    print(
        f"dataset: {args.dataset}, samples: {N}, views: {V}, class: {C}, missing_rate: {args.missing_rate}")
    print(f"{'=' * 60}")

    fully_missing_after = np.sum(~missing_mask.any(axis=1))

    print("\n=== incomplete graph construct ===")
    adj_incomplete_list = []
    k_opt_list = []

    for v in range(V):
        valid_idx = np.where(missing_mask[:, v] == 1)[0]
        valid_features = incomplete_views[v][valid_idx]

        if len(valid_idx) < 10:
            k_opt = min(15, max(1, len(valid_idx) - 1))
        else:
            k_opt, _ = select_k_by_entropy(valid_features)

        k_opt_list.append(k_opt)

        sub_adj = build_knn_graph(valid_features, k=max(k_opt, 1))
        full_adj = sp.lil_matrix((N, N), dtype=np.float32)
        sub_adj_coo = sub_adj.tocoo()
        row_global = valid_idx[sub_adj_coo.row]
        col_global = valid_idx[sub_adj_coo.col]
        full_adj[row_global, col_global] = sub_adj_coo.data
        full_adj = full_adj.tocsr()
        full_adj = (full_adj + full_adj.T) / 2
        full_adj.setdiag(0)
        full_adj.eliminate_zeros()
        adj_incomplete_list.append(full_adj)

    if args.missing_rate > 0:
        print("\n=== complete graph construct ===")
        adj_completed_list = consensus_guided_completion(
            adj_incomplete_list, missing_mask, k_opt_list
        )
    else:
        adj_completed_list = adj_incomplete_list

    print("\n=== training ===")
    input_feats = [incomplete_views[v].astype(np.float32) for v in range(V)]

    outputs = train_mv_hypcse(
        adj_completed_list,
        input_feats,
        missing_mask,
        C,
        labels,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        use_att=args.use_att,
        use_res=args.use_res,
        lambda_align=args.lambda_align,
        lambda_self_rec=args.lambda_self_rec,
        lambda_comp=args.lambda_comp,
        lambda_cse=args.lambda_cse,
        cse_start_epoch=args.cse_start_epoch,
        comp_observed_weight=args.comp_observed_weight,
    )

    print(f"\n===  Epoch {outputs.epoch + 1}, ACC={outputs.acc:.4f} ===")
    print(" Ward metric：")
    for k, v in outputs.results.items():
        print(f"  {k}: {v:.4f}")

    hier_labels = fcluster(outputs.tree_linkage, t=C, criterion="maxclust")
    flat_res = evaluate_clustering(labels, hier_labels)

    print("\n tree cluster：")
    for k, v in flat_res.items():
        print(f"  {k}: {v:.4f}")

    return {
        "dataset": args.dataset,
        "hierarchical": flat_res,
        "epoch": outputs.epoch,
        "acc": outputs.acc,
        "v_emb": outputs.v_emb,
        "p_emb": outputs.p_emb,
        "tree_linkage": outputs.tree_linkage,
        "model_state": outputs.model_state,
        "adj_states": outputs.adj_states,
        "results": outputs.results,
    }


def batch_train(dataset_configs, base_args):
    total_start_time = time.time()
    base_out_dir = base_args.out_dir
    os.makedirs(base_out_dir, exist_ok=True)
    n_repeats = base_args.n_repeats

    for dataset_name, custom_cfg in dataset_configs.items():
        base_args_dataset = argparse.Namespace(**vars(base_args))
        base_args_dataset.dataset = dataset_name
        base_args_dataset.out_dir = os.path.join(base_out_dir, dataset_name)
        for param_name, param_value in custom_cfg.items():
            setattr(base_args_dataset, param_name, param_value)

        set_seed(base_args.seed)
        views, labels = get_multiview_data(dataset_name)
        incomplete_views, missing_mask = missing(views, missing_rate=base_args.missing_rate)

        dataset_dir = os.path.join(base_out_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        result_file = os.path.join(dataset_dir, "results.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"=== Results for {dataset_name} ===\n")
            param_summary = {k: v for k, v in vars(base_args_dataset).items() if k != "out_dir"}
            f.write(f"Params: {json.dumps(param_summary, ensure_ascii=False)}\n")
            f.write(f"Repeats: {n_repeats}\n")
            f.write("-" * 60 + "\n")

        all_results = []

        for rep in range(n_repeats):
            current_seed = base_args.seed + rep
            args_rep = argparse.Namespace(**vars(base_args_dataset))
            args_rep.seed = current_seed
            args_rep.out_dir = dataset_dir
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                start_time = time.time()
                print(
                    f"\n>>> [{timestamp}] begin train: {dataset_name}, repeat {rep + 1}/{n_repeats}")
                results = run_single(args_rep, incomplete_views, missing_mask, labels)
                train_duration = time.time() - start_time
                results['train_duration'] = str(timedelta(seconds=int(train_duration)))
                results['repeat'] = rep + 1
                all_results.append(results)
                with open(result_file, "a", encoding="utf-8") as f:
                    f.write(f"Repeat {rep + 1}/{n_repeats}\n")
                    f.write(f"   Epoch: {results['epoch'] + 1}\n")
                    f.write(f"   ACC: {results['acc']:.4f}\n")
                    f.write(f"  Hierarchical: " + ", ".join(
                        f"{k}={v:.4f}" for k, v in results['hierarchical'].items()) + "\n")
                    f.write(f"  Train time: {results['train_duration']}\n")
                    f.write("-" * 40 + "\n")
                print(f"   result appended to {result_file}")

            except Exception as e:
                error_msg = f"train {dataset_name} repeat {rep + 1} failed: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                fail_file = os.path.join(dataset_dir, f"fail_rep{rep + 1}.txt")
                with open(fail_file, "w") as f:
                    f.write(f"Error: {error_msg}\n{traceback.format_exc()}")
                with open(result_file, "a", encoding="utf-8") as f:
                    f.write(f"Repeat {rep + 1}/{n_repeats}  ** FAILED **\n")
                    f.write(f"  Error: {error_msg}\n")
                    f.write("-" * 40 + "\n")
                continue

        if len(all_results) > 0:
            avg_acc = np.mean([r['acc'] for r in all_results])
            std_acc = np.std([r['acc'] for r in all_results])
            avg_epoch = np.mean([r['epoch'] + 1 for r in all_results])

            hier_metrics = set()
            for r in all_results:
                hier_metrics.update(r['hierarchical'].keys())

            avg_hier = {m: np.mean([r['hierarchical'].get(m, np.nan) for r in all_results]) for m in hier_metrics}
            std_hier = {m: np.std([r['hierarchical'].get(m, np.nan) for r in all_results]) for m in hier_metrics}

            result = max(all_results, key=lambda x: x['acc'])

            print(f"\n=== Result: ACC={result['acc']:.4f} ===")
            print(f"\n=== Average Results over {len(all_results)} successful runs ===")
            print(f"  Avg Epoch: {avg_epoch:.2f}")
            print(f"  Avg ACC: {avg_acc:.4f} +/- {std_acc:.4f}")
            print("  Hierarchical:")
            for m in sorted(hier_metrics):
                print(f"    {m}: {avg_hier[m]:.4f} +/- {std_hier[m]:.4f}")

            tree = linkage_to_tree(result['tree_linkage'], len(labels))
            tree_path = os.path.join(dataset_dir, f"{dataset_name}_coding_tree.json")
            linkage_path = os.path.join(dataset_dir, f"{dataset_name}_linkage.npy")
            np.save(linkage_path, result['tree_linkage'])
            save_tree_json(tree, tree_path)
            print(f"tree saved: {tree_path}")

            np.save(os.path.join(dataset_dir, f"{dataset_name}_v_emb.npy"), result['v_emb'])
            np.save(os.path.join(dataset_dir, f"{dataset_name}_p_emb.npy"), result['p_emb'])
            print(f"embeddings saved")

            model_path = os.path.join(dataset_dir, f"{dataset_name}_model.pt")
            torch.save({
                'model_state_dict': result['model_state'],
                'adj_graphs_states': result['adj_states'],
                'epoch': result['epoch'],
                'acc': result['acc'],
                'results': result['results'],
            }, model_path)
            print(f"model saved: {model_path}")

            raw_results_clean = []
            for r in all_results:
                raw_results_clean.append({
                    "repeat": r["repeat"],
                    "epoch": r["epoch"],
                    "acc": r["acc"],
                    "hierarchical": r["hierarchical"],
                    "train_duration": r["train_duration"]
                })
            raw_results_path = os.path.join(dataset_dir, "raw_results.json")
            with open(raw_results_path, "w", encoding="utf-8") as f:
                json.dump(raw_results_clean, f, ensure_ascii=False, indent=2, default=str)
            print(f"raw results saved: {raw_results_path}")

            summary = {
                "dataset": dataset_name,
                "missing_rate": base_args.missing_rate,
                "n_repeats": n_repeats,
                "n_success": len(all_results),
                "acc": float(result['acc']),
                "epoch": int(result['epoch']),
                "avg_epoch": float(avg_epoch),
                "avg_acc": float(avg_acc),
                "std_acc": float(std_acc),
                "hierarchical": {
                    "avg": {k: float(v) for k, v in avg_hier.items()},
                    "std": {k: float(v) for k, v in std_hier.items()}
                },
                "all_runs": raw_results_clean
            }
            summary_path = os.path.join(dataset_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"summary saved: {summary_path}")

            csv_path = os.path.join(dataset_dir, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = ["repeat", "epoch", "acc", "train_duration"]
                for m in sorted(hier_metrics):
                    headers.append(f"hier_{m}")
                writer.writerow(headers)
                for r in all_results:
                    row = [r["repeat"], r["epoch"] + 1, f"{r['acc']:.4f}", r["train_duration"]]
                    for m in sorted(hier_metrics):
                        row.append(f"{r['hierarchical'].get(m, float('nan')):.4f}")
                    writer.writerow(row)
            print(f"csv saved: {csv_path}")

            with open(result_file, "a", encoding="utf-8") as f:
                f.write(f"\n=== Average over {len(all_results)} successful runs ===\n")
                f.write(f"Epoch: {result['epoch'] + 1}, ACC: {result['acc']:.4f}\n")
                f.write(f"Avg   Epoch: {avg_epoch:.2f}\n")
                f.write(f"Avg   ACC: {avg_acc:.4f} +/- {std_acc:.4f}\n")
                f.write("Hierarchical:\n")
                for m in sorted(hier_metrics):
                    f.write(f"  {m}: {avg_hier[m]:.4f} +/- {std_hier[m]:.4f}\n")
        else:
            with open(result_file, "a", encoding="utf-8") as f:
                f.write("\nAll repeats failed.\n")

        print(f">>> {dataset_name} finished, {result_file}")

    total_duration = time.time() - total_start_time
    total_duration_str = str(timedelta(seconds=int(total_duration)))
    print(f"\n{'=' * 60}")
    print(f" finished！time：{total_duration_str}")
    print(f"{'=' * 60}")
    total_time_file = os.path.join(base_out_dir, "total_training_time.txt")
    with open(total_time_file, "w", encoding="utf-8") as f:
        f.write(f"batch total time：{total_duration_str}\n")
        f.write(f"finish time：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-view Clustering')
    parser.add_argument('--dataset', type=str, default='MSRCV1')
    parser.add_argument('--out_dir', type=str, default='./result/outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_att', action='store_true', default=False)
    parser.add_argument('--use_res', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_self_rec', type=float, default=1.0)
    parser.add_argument('--lambda_comp', type=float, default=1.0)
    parser.add_argument('--lambda_cse', type=float, default=1.0)
    parser.add_argument('--cse_start_epoch', type=int, default=50)
    parser.add_argument('--comp_observed_weight', type=float, default=0.2)
    parser.add_argument('--n_repeats', type=int, default=5, help='Number of repeats per dataset')

    args = parser.parse_args()
    # data_name = ['MSRC-v1', '100leaves', 'Handwritten', 'LandUse-21', 'Scene-15']
    data_name = ['MSRC-v1']
    for name in data_name:
        for i in [0.0, 0.1, 0.3, 0.5]:
            args.missing_rate = i
            args.dataset = name
            args.out_dir = f'./result/outputs{args.missing_rate}'
            DATASET_CONFIGS = {
                args.dataset: get_config(args.dataset, args.missing_rate)
            }
            batch_train(DATASET_CONFIGS, args)


if __name__ == "__main__":
    main()
