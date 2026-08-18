"""
Escalation step after scripts/run_pair_link_classifier_kfold.py's cheap
reweighting attempt turned out to be a wash (ROC-AUC 0.7046 vs. 0.7042 --
learned weights barely moved the metrics, because with all-positive counts
and all-positive learned weights, the "any overlap" decision boundary is
essentially invariant to the exact weighting -- reweighting only reorders
pairs THAT ALREADY have nonzero overlap, a small fraction of the total
sample, so its effect on aggregate ROC-AUC is naturally small).

This escalates to features that discriminate even among the ~zero-overlap
majority (where the 4-count feature set has nothing to say at all):
  - tanimoto_sim   : Morgan/ECFP fingerprint similarity between the pair --
                     continuous, non-zero for almost every pair, captures
                     "these two drugs are chemically similar" independent
                     of whether their DOCUMENTED profiles happen to overlap.
  - profile_size_a/b : total documented profile size (enzyme+target+
                     transporter+carrier count) per drug -- controls for
                     "well-studied/promiscuous" drugs naturally having more
                     overlap chances, and may carry signal on its own
                     (heavily-annotated drugs are often heavily-interacting
                     drugs in DrugBank's own curation practice).
  - min_profile_size : min(size_a, size_b) -- a sparse-profile drug caps
                     how much OVERLAP is even possible regardless of true
                     biology; worth having as an explicit feature rather
                     than a confound the other features silently absorb.

Same held-out evaluation protocol (5-fold, random pair split, same sampled
population) as the two prior link-prediction scripts, for direct
comparison against both: naive_sum (0.7042 ROC-AUC) and logreg-on-4-counts
(0.7046 ROC-AUC).

Usage:
    python scripts/run_pair_link_classifier_richfeatures_kfold.py \
        --n-samples 50000 --k 5 --out checkpoints/pair_link_classifier_richfeatures_kfold_results.json
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.evaluate_mechanistic_overlap_link_prediction import build_profile_by_drug, overlap_score
from scripts.mechanism_knn_transfer import tanimoto_similarity

DATA_DIR = Path("data/processed")
KIND_ORDER = ("enzyme", "target", "transporter", "carrier")
FEATURE_NAMES = list(KIND_ORDER) + ["tanimoto_sim", "profile_size_a", "profile_size_b", "min_profile_size"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--out", default="checkpoints/pair_link_classifier_richfeatures_kfold_results.json")
    return p.parse_args()


def sample_pairs(profile_by_drug, n_samples, rng):

    any_profile = set(profile_by_drug.keys())

    positive_pairs = set()
    with open(DATA_DIR / "drugbank_interactions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            a, b = row["drug1"], row["drug2"]
            if a in any_profile and b in any_profile and a != b:
                positive_pairs.add(frozenset((a, b)))

    n = min(n_samples, len(positive_pairs))
    sampled_positives = rng.sample(sorted(positive_pairs, key=lambda s: sorted(s)), n)

    profile_list = sorted(any_profile)
    negatives = set()
    attempts = 0
    while len(negatives) < n and attempts < n * 50:
        a, b = rng.sample(profile_list, 2)
        pair = frozenset((a, b))
        if pair not in positive_pairs and pair not in negatives:
            negatives.add(pair)
        attempts += 1

    return sampled_positives, list(negatives), len(positive_pairs)


def profile_size(profile_by_drug, drug):
    return sum(len(profile_by_drug[drug][kind]) for kind in KIND_ORDER)


def build_features(profile_by_drug, fp_by_drug, pairs):

    x = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float64)

    fp_missing = 0
    for i, pair in enumerate(pairs):
        a, b = tuple(pair)
        _, per_kind = overlap_score(profile_by_drug, a, b)
        for j, kind in enumerate(KIND_ORDER):
            x[i, j] = per_kind[kind]

        if a in fp_by_drug and b in fp_by_drug:
            sim = tanimoto_similarity(fp_by_drug[a].unsqueeze(0), fp_by_drug[b].unsqueeze(0)).item()
        else:
            sim = 0.0
            fp_missing += 1
        x[i, 4] = sim

        size_a = profile_size(profile_by_drug, a)
        size_b = profile_size(profile_by_drug, b)
        x[i, 5] = size_a
        x[i, 6] = size_b
        x[i, 7] = min(size_a, size_b)

    if fp_missing:
        print(f"  ({fp_missing}/{len(pairs)} pairs had a drug with no fingerprint -- tanimoto_sim set to 0 for those)")

    return x


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


def main():

    args = parse_args()
    rng = random.Random(args.seed)

    print("Loading per-drug profiles and fingerprints...")
    profile_by_drug = build_profile_by_drug()
    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    print("Sampling positive/negative pairs...")
    sampled_positives, sampled_negatives, total_positive_pairs = sample_pairs(
        profile_by_drug, args.n_samples, rng
    )
    n = len(sampled_positives)
    print(f"  {n} positive, {len(sampled_negatives)} negative "
          f"(of {total_positive_pairs} total restricted-population positive pairs)")

    all_pairs = sampled_positives + sampled_negatives
    y = np.array([1] * len(sampled_positives) + [0] * len(sampled_negatives))

    print("Building rich features (4 overlap counts + tanimoto_sim + profile sizes)...")
    x = build_features(profile_by_drug, fp_by_drug, all_pairs)

    # Standardize continuous features (tanimoto_sim, profile sizes) for a
    # well-conditioned logistic regression -- fit on TRAIN only, per fold.
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    fold_metrics = {"logreg_4feat": [], "logreg_8feat": []}
    fold_coefs = []

    for i, (train_idx, test_idx) in enumerate(skf.split(x, y)):

        print(f"\n===== Fold {i + 1}/{args.k} =====")

        x_train, x_test = x[train_idx].copy(), x[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        x_train_norm = (x_train - mean) / std
        x_test_norm = (x_test - mean) / std

        # Reference: logreg on just the original 4 overlap counts (this
        # fold's own split, for a clean apples-to-apples comparison).
        clf4 = LogisticRegression(max_iter=1000)
        clf4.fit(x_train_norm[:, :4], y_train)
        probs4 = clf4.predict_proba(x_test_norm[:, :4])[:, 1]
        preds4 = (probs4 >= 0.5).astype(int)
        m4 = binary_metrics(y_test, preds4)
        m4["roc_auc"] = roc_auc_score(y_test, probs4)
        m4["pr_auc"] = average_precision_score(y_test, probs4)
        fold_metrics["logreg_4feat"].append(m4)

        # All 8 features.
        clf8 = LogisticRegression(max_iter=1000)
        clf8.fit(x_train_norm, y_train)
        probs8 = clf8.predict_proba(x_test_norm)[:, 1]
        preds8 = (probs8 >= 0.5).astype(int)
        m8 = binary_metrics(y_test, preds8)
        m8["roc_auc"] = roc_auc_score(y_test, probs8)
        m8["pr_auc"] = average_precision_score(y_test, probs8)
        fold_metrics["logreg_8feat"].append(m8)
        fold_coefs.append({name: float(c) for name, c in zip(FEATURE_NAMES, clf8.coef_[0])})

        print(f"  logreg_4feat: ROC-AUC={m4['roc_auc']:.4f}  precision={m4['precision']:.4f}  recall={m4['recall']:.4f}")
        print(f"  logreg_8feat: ROC-AUC={m8['roc_auc']:.4f}  precision={m8['precision']:.4f}  recall={m8['recall']:.4f}")

    def mean_std(key, variant):
        vals = [m[key] for m in fold_metrics[variant]]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {"k": args.k, "n_positive": n, "n_negative": len(sampled_negatives), "feature_names": FEATURE_NAMES, "variants": {}}

    print(f"\n===== Summary (k={args.k}-fold) =====")
    print("  Reference: naive_sum ROC-AUC 0.7042+-0.0018, logreg-on-4-counts 0.7046+-0.0018 (prior results)")
    for variant in ("logreg_4feat", "logreg_8feat"):
        entry = {}
        for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            m, s = mean_std(key, variant)
            entry[key] = {"mean": m, "std": s}
        summary["variants"][variant] = entry
        print(f"  [{variant:14s}] ROC-AUC {entry['roc_auc']['mean']:.4f}+-{entry['roc_auc']['std']:.4f}  "
              f"precision {entry['precision']['mean']:.4f}+-{entry['precision']['std']:.4f}  "
              f"recall {entry['recall']['mean']:.4f}+-{entry['recall']['std']:.4f}  "
              f"F1 {entry['f1']['mean']:.4f}+-{entry['f1']['std']:.4f}")

    mean_coefs = {name: statistics.mean(c[name] for c in fold_coefs) for name in FEATURE_NAMES}
    summary["mean_learned_coefficients_8feat"] = mean_coefs
    print(f"\n  Mean learned coefficients (8-feature model, standardized features): {mean_coefs}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
