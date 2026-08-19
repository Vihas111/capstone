"""
Task-supervised GNN embeddings for the pair-level link classifier -- the
lever findings.md flagged as the actual remaining gap after the spectral-
embedding result (combined HGB ROC-AUC 0.784): that embedding is frozen and
unsupervised (SVD of Hetionet alone, never sees a single DDI label), while
SumGNN/medicX both train their representations END-TO-END against the DDI
task itself. This trains a small 2-layer GCN (torch_geometric) over the
whole Hetionet graph (47,031 nodes, all 24 edge types) with a symmetric
pairwise decoder, optimized directly against sampled DrugBank interaction
pairs -- architecturally the same idea as SumGNN, far simpler (no
relation-aware message passing, no subgraph reasoning, no external KG
embedding pretraining).

No leakage risk from the graph itself: Hetionet has no drug-drug
interaction edge type at all, so the SAME fixed graph structure is reused
across all folds -- only the link-prediction head + node embeddings are
retrained fresh per fold (standard k-fold practice), on that fold's
training pairs only.

Same population/sampling protocol as the other pair-classifier scripts
(1,460-drug Hetionet-covered population, size-matched negatives) for direct
comparability.

Usage:
    python scripts/run_pair_link_classifier_gnn_kfold.py \
        --n-samples 50000 --k 5 --epochs 150 --out checkpoints/pair_link_classifier_gnn_kfold_results.json
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GCNConv

from scripts.evaluate_mechanistic_overlap_link_prediction import build_profile_by_drug
from scripts.hetionet_features import build_pyg_graph, load_hetionet_compounds

DATA_DIR = Path("data/processed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--out", default="checkpoints/pair_link_classifier_gnn_kfold_results.json")
    return p.parse_args()


def sample_positive_pairs(population, n_samples, rng):
    positive_pairs = set()
    with open(DATA_DIR / "drugbank_interactions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            a, b = row["drug1"], row["drug2"]
            if a in population and b in population and a != b:
                positive_pairs.add(frozenset((a, b)))

    n = min(n_samples, len(positive_pairs))
    sampled = rng.sample(sorted(positive_pairs, key=lambda s: sorted(s)), n)
    return sampled, positive_pairs


def sample_size_matched_negatives(sampled_positives, all_positive_pairs, n_needed, rng):
    degree = Counter()
    for pair in sampled_positives:
        for d in pair:
            degree[d] += 1

    weighted_pool = []
    for drug, count in degree.items():
        weighted_pool.extend([drug] * count)

    negatives = set()
    attempts = 0
    max_attempts = n_needed * 200
    while len(negatives) < n_needed and attempts < max_attempts:
        a, b = rng.sample(weighted_pool, 2)
        if a == b:
            attempts += 1
            continue
        pair = frozenset((a, b))
        if pair not in all_positive_pairs and pair not in negatives:
            negatives.add(pair)
        attempts += 1

    return list(negatives), attempts


class HetionetGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, out_dim=128, dropout=0.2):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.normal_(self.node_emb.weight, std=0.1)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, edge_index):
        x = self.node_emb.weight
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class PairDecoder(nn.Module):
    """Symmetric under (a, b) -> (b, a) swap -- drug pairs are unordered,
    so raw concat(z_a, z_b) would wrongly make the score order-dependent."""

    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, za, zb):
        feat = torch.cat([za * zb, torch.abs(za - zb), za + zb], dim=-1)
        return self.mlp(feat).squeeze(-1)


def binary_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def run_fold(edge_index, node_index, all_pairs, y, train_idx, test_idx, args, device):
    rng = random.Random(args.seed)

    # Carve a small internal validation split from train, for early stopping.
    train_idx = list(train_idx)
    rng.shuffle(train_idx)
    n_val = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[:n_val]
    fit_idx = train_idx[n_val:]

    def to_node_pairs(idx_list):
        a_idx, b_idx, labels = [], [], []
        for i in idx_list:
            a, b = tuple(all_pairs[i])
            a_idx.append(node_index[f"Compound::{a}"])
            b_idx.append(node_index[f"Compound::{b}"])
            labels.append(y[i])
        return (torch.tensor(a_idx, device=device), torch.tensor(b_idx, device=device),
                torch.tensor(labels, dtype=torch.float32, device=device))

    fit_a, fit_b, fit_y = to_node_pairs(fit_idx)
    val_a, val_b, val_y = to_node_pairs(val_idx)
    test_a, test_b, test_y = to_node_pairs(test_idx)

    num_nodes = len(node_index)
    encoder = HetionetGCN(num_nodes, args.hidden_dim, args.out_dim).to(device)
    decoder = PairDecoder(args.out_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        z = encoder(edge_index)
        logits = decoder(z[fit_a], z[fit_b])
        loss = F.binary_cross_entropy_with_logits(logits, fit_y)
        loss.backward()
        optimizer.step()

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z = encoder(edge_index)
            val_logits = decoder(z[val_a], z[val_b])
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(val_y.cpu().numpy(), val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = ({k: v.clone() for k, v in encoder.state_dict().items()},
                          {k: v.clone() for k, v in decoder.state_dict().items()})
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                break

    encoder.load_state_dict(best_state[0])
    decoder.load_state_dict(best_state[1])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(edge_index)
        test_logits = decoder(z[test_a], z[test_b])
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    y_test = test_y.cpu().numpy()
    preds = (test_probs >= 0.5).astype(int)
    m = binary_metrics(y_test, preds)
    m["roc_auc"] = roc_auc_score(y_test, test_probs)
    m["pr_auc"] = average_precision_score(y_test, test_probs)
    m["best_epoch_val_auc"] = float(best_val_auc)
    m["stopped_epoch"] = epoch + 1
    return m


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DrugBank profiles, fingerprints, and Hetionet graph (PyG format)...")
    profile_by_drug = build_profile_by_drug()
    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_ids = set(fp_data["drug_ids"])
    hetio_compounds = load_hetionet_compounds()
    node_index, edge_index = build_pyg_graph()
    edge_index = edge_index.to(device)

    population = hetio_compounds & set(profile_by_drug.keys()) & fp_ids
    print(f"  Restricted population: {len(population)} drugs. Graph: {len(node_index)} nodes, "
          f"{edge_index.shape[1]} directed edges.")

    print("Sampling positive pairs...")
    sampled_positives, all_positive_pairs = sample_positive_pairs(population, args.n_samples, rng)
    n = len(sampled_positives)
    print(f"  {n} positive pairs sampled.")

    print("Sampling size-matched (degree-preserving) negative pairs...")
    sampled_negatives, attempts = sample_size_matched_negatives(sampled_positives, all_positive_pairs, n, rng)
    print(f"  {len(sampled_negatives)} negative pairs sampled ({attempts} attempts).")

    all_pairs = sampled_positives + sampled_negatives
    y = np.array([1] * len(sampled_positives) + [0] * len(sampled_negatives))

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    dummy_x = np.zeros((len(all_pairs), 1))

    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(skf.split(dummy_x, y)):
        print(f"\n===== Fold {i + 1}/{args.k} =====")
        m = run_fold(edge_index, node_index, all_pairs, y, train_idx, test_idx, args, device)
        fold_metrics.append(m)
        print(f"  ROC-AUC={m['roc_auc']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  (stopped epoch {m['stopped_epoch']}, best val AUC {m['best_epoch_val_auc']:.4f})")

    def mean_std(key):
        vals = [m[key] for m in fold_metrics]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n===== Summary (k={args.k}-fold, size-matched negatives, task-supervised GCN) =====")
    entry = {}
    for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
        m, s = mean_std(key)
        entry[key] = {"mean": m, "std": s}
    print(f"  ROC-AUC {entry['roc_auc']['mean']:.4f}+-{entry['roc_auc']['std']:.4f}  "
          f"precision {entry['precision']['mean']:.4f}+-{entry['precision']['std']:.4f}  "
          f"recall {entry['recall']['mean']:.4f}+-{entry['recall']['std']:.4f}  "
          f"F1 {entry['f1']['mean']:.4f}+-{entry['f1']['std']:.4f}")

    summary = {
        "k": args.k,
        "n_positive": n,
        "n_negative": len(sampled_negatives),
        "population_size": len(population),
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": str(device),
        "metrics": entry,
        "fold_details": fold_metrics,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
