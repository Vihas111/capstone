"""
Size-matched negative sampling -- the standard fix for the "degree bias"
confound scripts/run_pair_link_classifier_richfeatures_kfold.py's ablation
found: profile-size features drove nearly all of that script's ROC-AUC
jump (0.7046 -> 0.8713 from sizes alone, vs. 0.7046 -> 0.7064 from
fingerprint similarity alone), which could reflect genuine biology (a
mechanistically "busier" drug really is more likely to interact) or
curation bias (well-studied drugs get more documented interactions
regardless of true mechanism) -- the plain random-negative-sampling setup
can't distinguish these.

Fix: instead of drawing negative pairs uniformly at random from the drug
population, draw each negative endpoint with probability proportional to
how often that drug appears among the SAMPLED POSITIVE pairs (i.e.
degree-preserving negative sampling -- standard practice in link-
prediction literature specifically for this confound). This makes
negatives' profile-size distribution match positives' BY CONSTRUCTION, so
whatever residual ROC-AUC survives can no longer be explained by "big
drugs get sampled as positives more often" -- it has to come from actual
per-pair structure (protein overlap, fingerprint similarity).

Same 4-feature vs. 8-feature comparison, same k-fold protocol, as the
prior two link-classifier scripts, for direct comparison:
  - uniform negatives:      naive_sum 0.7042, logreg-4feat 0.7046, logreg-8feat 0.8735
  - size-matched negatives: (this script)

Usage:
    python scripts/run_pair_link_classifier_sizematched_kfold.py \
        --n-samples 50000 --k 5 --out checkpoints/pair_link_classifier_sizematched_kfold_results.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.evaluate_mechanistic_overlap_link_prediction import build_profile_by_drug, overlap_score
from scripts.mechanism_knn_transfer import tanimoto_similarity
from scripts.run_pair_link_classifier_richfeatures_kfold import FEATURE_NAMES, build_features, binary_metrics

DATA_DIR = Path("data/processed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--out", default="checkpoints/pair_link_classifier_sizematched_kfold_results.json")
    return p.parse_args()


def sample_positive_pairs(profile_by_drug, n_samples, rng):

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
    return sampled_positives, positive_pairs


def sample_size_matched_negatives(sampled_positives, all_positive_pairs, n_needed, rng):
    """Draws negative endpoints with probability proportional to each
    drug's degree within sampled_positives -- makes the negative set's
    profile-size distribution match the positive set's by construction."""

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


def main():

    args = parse_args()
    rng = random.Random(args.seed)

    print("Loading per-drug profiles and fingerprints...")
    profile_by_drug = build_profile_by_drug()
    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    print("Sampling positive pairs...")
    sampled_positives, all_positive_pairs = sample_positive_pairs(profile_by_drug, args.n_samples, rng)
    n = len(sampled_positives)
    print(f"  {n} positive pairs sampled.")

    print("Sampling SIZE-MATCHED (degree-preserving) negative pairs...")
    sampled_negatives, attempts = sample_size_matched_negatives(sampled_positives, all_positive_pairs, n, rng)
    print(f"  {len(sampled_negatives)} negative pairs sampled ({attempts} attempts).")

    all_pairs = sampled_positives + sampled_negatives
    y = np.array([1] * len(sampled_positives) + [0] * len(sampled_negatives))

    print("Building rich features (4 overlap counts + tanimoto_sim + profile sizes)...")
    x = build_features(profile_by_drug, fp_by_drug, all_pairs)

    # Sanity check: confirm size-matching actually worked -- compare mean
    # min_profile_size between positives and negatives (should now be close).
    min_size_idx = FEATURE_NAMES.index("min_profile_size")
    pos_mean_size = x[y == 1, min_size_idx].mean()
    neg_mean_size = x[y == 0, min_size_idx].mean()
    print(f"  Sanity check -- mean min_profile_size: positives={pos_mean_size:.2f}  negatives={neg_mean_size:.2f}")

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

        clf4 = LogisticRegression(max_iter=1000)
        clf4.fit(x_train_norm[:, :4], y_train)
        probs4 = clf4.predict_proba(x_test_norm[:, :4])[:, 1]
        preds4 = (probs4 >= 0.5).astype(int)
        m4 = binary_metrics(y_test, preds4)
        m4["roc_auc"] = roc_auc_score(y_test, probs4)
        m4["pr_auc"] = average_precision_score(y_test, probs4)
        fold_metrics["logreg_4feat"].append(m4)

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

    summary = {
        "k": args.k, "n_positive": n, "n_negative": len(sampled_negatives),
        "feature_names": FEATURE_NAMES,
        "sanity_check_mean_min_profile_size": {"positives": float(pos_mean_size), "negatives": float(neg_mean_size)},
        "variants": {},
    }

    print(f"\n===== Summary (k={args.k}-fold, SIZE-MATCHED negatives) =====")
    print("  Reference (uniform-random negatives): naive_sum 0.7042, logreg-4feat 0.7046, logreg-8feat 0.8735")
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
