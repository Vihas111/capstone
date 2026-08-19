"""
Tests whether Hetionet knowledge-graph features close the gap between this
project's pair-level link classifier (honest, size-matched ROC-AUC 0.650 --
scripts/run_pair_link_classifier_sizematched_kfold.py) and published
KG-based DDI models (SumGNN/medicX, ~0.87-0.95). This is the one lever
findings.md identified as architecturally correct for this task (known
drugs, undocumented combinations) but not yet tried.

IMPORTANT population caveat (checked before writing this script): Hetionet
only covers 1,552 compounds, a much smaller population than the 10,192-drug
DrugBank profile population the old 0.650 baseline was measured on. Directly
comparing "0.650 (old, big population)" vs. "X (new, small population)"
would confound "did the KG help" with "is the restricted population
easier/harder." This script re-baselines EVERYTHING -- old 8 DrugBank-only
features, Hetionet-only features, and the combined 16-feature set -- on the
IDENTICAL restricted population and IDENTICAL sampled pairs, same
size-matched (degree-preserving) negative sampling protocol as before, so
the three numbers are directly comparable to each other. The old 0.650 is
kept only as a loose sanity-check reference, not the real comparison point.

Usage:
    python scripts/run_pair_link_classifier_hetionet_kfold.py \
        --n-samples 50000 --k 5 --out checkpoints/pair_link_classifier_hetionet_kfold_results.json
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.evaluate_mechanistic_overlap_link_prediction import build_profile_by_drug, overlap_score
from scripts.hetionet_features import (
    HETIO_2ND_ORDER_FEATURE_NAMES,
    HETIO_ADAMIC_ADAR_FEATURE_NAMES,
    HETIO_FEATURE_NAMES,
    adamic_adar_pair_features,
    build_full_graph_embeddings,
    build_node_degree,
    build_touched_genes_and_pathways,
    embedding_pair_features,
    hetionet_2nd_order_pair_features,
    hetionet_pair_features,
    load_gene_graph,
    load_hetionet_compounds,
    load_hetionet_graph,
)
from scripts.mechanism_knn_transfer import tanimoto_similarity
from scripts.run_pair_link_classifier_richfeatures_kfold import FEATURE_NAMES as OLD_FEATURE_NAMES
from scripts.run_pair_link_classifier_richfeatures_kfold import binary_metrics, profile_size

DATA_DIR = Path("data/processed")
EMB_K = 96
HETIO_EMBEDDING_FEATURE_NAMES = (["spectral_emb_cosine_sim", "spectral_emb_dot_product"]
                                  + [f"spectral_emb_prod_{i}" for i in range(EMB_K)])
ALL_FEATURE_NAMES = (OLD_FEATURE_NAMES + HETIO_FEATURE_NAMES + HETIO_2ND_ORDER_FEATURE_NAMES
                      + HETIO_ADAMIC_ADAR_FEATURE_NAMES + HETIO_EMBEDDING_FEATURE_NAMES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--out", default="checkpoints/pair_link_classifier_hetionet_kfold_results.json")
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


def build_all_features(profile_by_drug, fp_by_drug, neighbor_sets, pharm_class_by_drug, resembles_pairs,
                        touched_genes, pathways_by_drug, gene_gene_adj, node_degree, embeddings, pairs):
    x = np.zeros((len(pairs), len(ALL_FEATURE_NAMES)), dtype=np.float64)
    n_hetio1 = len(HETIO_FEATURE_NAMES)
    n_2nd = len(HETIO_2ND_ORDER_FEATURE_NAMES)
    n_aa = len(HETIO_ADAMIC_ADAR_FEATURE_NAMES)
    off1 = 8
    off2 = off1 + n_hetio1
    off3 = off2 + n_2nd
    off4 = off3 + n_aa

    for i, pair in enumerate(pairs):
        a, b = tuple(pair)

        _, per_kind = overlap_score(profile_by_drug, a, b)
        for j, kind in enumerate(("enzyme", "target", "transporter", "carrier")):
            x[i, j] = per_kind[kind]

        sim = tanimoto_similarity(fp_by_drug[a].unsqueeze(0), fp_by_drug[b].unsqueeze(0)).item()
        x[i, 4] = sim

        size_a = profile_size(profile_by_drug, a)
        size_b = profile_size(profile_by_drug, b)
        x[i, 5] = size_a
        x[i, 6] = size_b
        x[i, 7] = min(size_a, size_b)

        hetio_feats = hetionet_pair_features(neighbor_sets, pharm_class_by_drug, resembles_pairs, a, b)
        x[i, off1:off2] = hetio_feats

        second_order = hetionet_2nd_order_pair_features(touched_genes, pathways_by_drug, gene_gene_adj, a, b)
        x[i, off2:off3] = second_order

        aa_feats = adamic_adar_pair_features(neighbor_sets, node_degree, a, b)
        x[i, off3:off4] = aa_feats

        emb_feats = embedding_pair_features(embeddings, a, b, k=EMB_K, with_elementwise=True)
        x[i, off4:] = emb_feats

    return x


def run_variant(name, x, y, skf, model="logreg"):
    fold_metrics = []
    for train_idx, test_idx in skf.split(x, y):
        x_train, x_test = x[train_idx].copy(), x[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        if model == "logreg":
            mean = x_train.mean(axis=0, keepdims=True)
            std = x_train.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0
            x_train_norm = (x_train - mean) / std
            x_test_norm = (x_test - mean) / std
            clf = LogisticRegression(max_iter=1000)
            clf.fit(x_train_norm, y_train)
            probs = clf.predict_proba(x_test_norm)[:, 1]
        elif model == "hgb":
            clf = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.05,
                                                  l2_regularization=1.0, random_state=42, early_stopping=True,
                                                  validation_fraction=0.1, n_iter_no_change=20)
            clf.fit(x_train, y_train)
            probs = clf.predict_proba(x_test)[:, 1]
        else:
            raise ValueError(model)

        preds = (probs >= 0.5).astype(int)
        m = binary_metrics(y_test, preds)
        m["roc_auc"] = roc_auc_score(y_test, probs)
        m["pr_auc"] = average_precision_score(y_test, probs)
        fold_metrics.append(m)

    def mean_std(key):
        vals = [m[key] for m in fold_metrics]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    entry = {key: dict(zip(("mean", "std"), mean_std(key))) for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1")}
    print(f"  [{name:12s}] ROC-AUC {entry['roc_auc']['mean']:.4f}+-{entry['roc_auc']['std']:.4f}  "
          f"precision {entry['precision']['mean']:.4f}+-{entry['precision']['std']:.4f}  "
          f"recall {entry['recall']['mean']:.4f}+-{entry['recall']['std']:.4f}  "
          f"F1 {entry['f1']['mean']:.4f}+-{entry['f1']['std']:.4f}")
    return entry


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("Loading DrugBank profiles, fingerprints, and Hetionet graph...")
    profile_by_drug = build_profile_by_drug()
    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))
    hetio_compounds = load_hetionet_compounds()
    neighbor_sets, pharm_class_by_drug, resembles_pairs = load_hetionet_graph()
    gene_gene_adj, gene_pathway_adj = load_gene_graph()
    touched_genes, pathways_by_drug = build_touched_genes_and_pathways(neighbor_sets, gene_pathway_adj)
    node_degree = build_node_degree(neighbor_sets)
    print(f"  Computing full-graph spectral embeddings (k={EMB_K}, all 47k nodes, all edge types)...")
    embeddings = build_full_graph_embeddings(k=EMB_K, seed=args.seed)

    population = hetio_compounds & set(profile_by_drug.keys()) & set(fp_by_drug.keys())
    print(f"  Restricted population (DrugBank profile + fingerprint + Hetionet Compound node): {len(population)} drugs")

    print("Sampling positive pairs...")
    sampled_positives, all_positive_pairs = sample_positive_pairs(population, args.n_samples, rng)
    n = len(sampled_positives)
    print(f"  {n} positive pairs sampled (of {len(all_positive_pairs)} restricted-population documented pairs).")

    print("Sampling size-matched (degree-preserving) negative pairs...")
    sampled_negatives, attempts = sample_size_matched_negatives(sampled_positives, all_positive_pairs, n, rng)
    print(f"  {len(sampled_negatives)} negative pairs sampled ({attempts} attempts).")

    all_pairs = sampled_positives + sampled_negatives
    y = np.array([1] * len(sampled_positives) + [0] * len(sampled_negatives))

    print("Building features (8 DrugBank/fingerprint + 8 Hetionet direct + 2 gene/pathway-mediated "
          "+ 6 Adamic-Adar + 2 spectral-embedding)...")
    x = build_all_features(profile_by_drug, fp_by_drug, neighbor_sets, pharm_class_by_drug, resembles_pairs,
                            touched_genes, pathways_by_drug, gene_gene_adj, node_degree, embeddings, all_pairs)

    min_size_idx = OLD_FEATURE_NAMES.index("min_profile_size")
    pos_mean_size = x[y == 1, min_size_idx].mean()
    neg_mean_size = x[y == 0, min_size_idx].mean()
    print(f"  Sanity check -- mean min_profile_size: positives={pos_mean_size:.2f}  negatives={neg_mean_size:.2f}")

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    print(f"\n===== Results (k={args.k}-fold, size-matched negatives, restricted to {len(population)}-drug Hetionet-covered population) =====")
    print("  Loose reference (old baseline, FULL 10,192-drug population, NOT the same population): logreg-8feat ROC-AUC 0.650+-0.003")

    n_hetio_total = len(ALL_FEATURE_NAMES) - 8
    variants = {}
    variants["old8_rebaselined_logreg"] = run_variant("old8/logreg", x[:, :8], y, skf, model="logreg")
    variants[f"hetionet_only_logreg"] = run_variant(f"hetio{n_hetio_total}/logreg", x[:, 8:], y, skf, model="logreg")
    variants["combined_logreg"] = run_variant("comb/logreg", x, y, skf, model="logreg")
    variants["old8_hgb"] = run_variant("old8/hgb", x[:, :8], y, skf, model="hgb")
    variants[f"hetionet_only_hgb"] = run_variant(f"hetio{n_hetio_total}/hgb", x[:, 8:], y, skf, model="hgb")
    variants["combined_hgb"] = run_variant("comb/hgb", x, y, skf, model="hgb")

    # Fit once on all data for interpretable mean coefficients on the combined model.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_norm = (x - mean) / std
    clf_full = LogisticRegression(max_iter=1000)
    clf_full.fit(x_norm, y)
    coefs = {name: float(c) for name, c in zip(ALL_FEATURE_NAMES, clf_full.coef_[0])}
    print(f"\n  Combined-model logreg coefficients (standardized features, fit on full sample): {coefs}")

    hgb_full = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.05,
                                               l2_regularization=1.0, random_state=42, early_stopping=True,
                                               validation_fraction=0.1, n_iter_no_change=20)
    hgb_full.fit(x, y)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(hgb_full, x, y, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1)
    importances = {name: float(v) for name, v in zip(ALL_FEATURE_NAMES, perm.importances_mean)}
    print(f"\n  Combined-model HGB permutation importances (ROC-AUC drop, fit on full sample): {importances}")

    summary = {
        "k": args.k,
        "n_positive": n,
        "n_negative": len(sampled_negatives),
        "population_size": len(population),
        "feature_names": ALL_FEATURE_NAMES,
        "old_feature_names": OLD_FEATURE_NAMES,
        "hetio_feature_names": HETIO_FEATURE_NAMES,
        "hetio_2nd_order_feature_names": HETIO_2ND_ORDER_FEATURE_NAMES,
        "hetio_adamic_adar_feature_names": HETIO_ADAMIC_ADAR_FEATURE_NAMES,
        "hetio_embedding_feature_names": HETIO_EMBEDDING_FEATURE_NAMES,
        "sanity_check_mean_min_profile_size": {"positives": float(pos_mean_size), "negatives": float(neg_mean_size)},
        "variants": variants,
        "combined_model_logreg_coefficients": coefs,
        "combined_model_hgb_permutation_importances": importances,
        "loose_reference_old_baseline_full_population": {"roc_auc_mean": 0.650, "roc_auc_std": 0.003, "note": "measured on the full 10192-drug population, NOT the same restricted population -- not a fair direct comparison, see old8_rebaselined_logreg for that"},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
