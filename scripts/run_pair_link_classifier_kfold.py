"""
k-fold evaluation of a LEARNED classifier on top of the same per-kind
shared-protein overlap counts scripts/evaluate_mechanistic_overlap_link_prediction.py
already computes -- the cheapest possible next step after that benchmark
found ROC-AUC 0.704 / 96.6% precision / 42.1% recall using a naive
equal-weighted sum of the 4 kinds. That sum treats enzyme/target/
transporter/carrier overlap as equally predictive, but the same benchmark's
own per-kind breakdown showed they aren't (enzyme ROC-AUC 0.664 vs.
target/carrier ~0.52) -- a logistic regression on the SAME 4 features,
letting it learn the right weights instead of assuming equal ones, is
the obvious cheap next step before trying anything with new features.

IMPORTANT (see findings/findings.md's write-up of the design decision this
implements): this classifier is deliberately scoped as an ADDITIVE third
signal for scripts/mechanism_lookup.py, only ever surfaced for pairs with
NO documented DrugBank interaction text -- never a replacement for the
deterministic lookup, which stays ~100% reliable for documented pairs and
is never touched by this. Same "predicted_X sits alongside documented X,
never replaces it" pattern this project already uses for the mechanism
gap-filler's --predict flag.

Uses a plain random k-fold split over sampled PAIRS (not drug-disjoint,
unlike the mechanism gap-filler's cold-start splits) -- this task is
explicitly about known drugs / new COMBINATIONS (the axis DeepDDI/SumGNN's
transductive split/medicX benchmark), not about generalizing to unseen
drugs, so a random pair split is the methodologically appropriate
evaluation here, matching how the literature being compared against
evaluates too.

Usage:
    python scripts/run_pair_link_classifier_kfold.py \
        --n-samples 50000 --k 5 --out checkpoints/pair_link_classifier_kfold_results.json
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.evaluate_mechanistic_overlap_link_prediction import build_profile_by_drug, overlap_score

DATA_DIR = Path("data/processed")
KIND_ORDER = ("enzyme", "target", "transporter", "carrier")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/pair_link_classifier_kfold_results.json")
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


def build_features(profile_by_drug, pairs):
    """[N, 4] per-kind shared-protein counts, in KIND_ORDER."""
    x = np.zeros((len(pairs), len(KIND_ORDER)), dtype=np.float64)
    for i, pair in enumerate(pairs):
        a, b = tuple(pair)
        _, per_kind = overlap_score(profile_by_drug, a, b)
        for j, kind in enumerate(KIND_ORDER):
            x[i, j] = per_kind[kind]
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

    print("Loading per-drug profiles...")
    profile_by_drug = build_profile_by_drug()

    print("Sampling positive/negative pairs...")
    sampled_positives, sampled_negatives, total_positive_pairs = sample_pairs(
        profile_by_drug, args.n_samples, rng
    )
    n = len(sampled_positives)
    print(f"  {n} positive, {len(sampled_negatives)} negative "
          f"(of {total_positive_pairs} total restricted-population positive pairs)")

    all_pairs = sampled_positives + sampled_negatives
    y = np.array([1] * len(sampled_positives) + [0] * len(sampled_negatives))

    print("Building per-kind overlap-count features...")
    x = build_features(profile_by_drug, all_pairs)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    fold_metrics = {"naive_sum": [], "logreg": []}
    fold_coefs = []

    for i, (train_idx, test_idx) in enumerate(skf.split(x, y)):

        print(f"\n===== Fold {i + 1}/{args.k} =====")

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Naive equal-weighted sum (the existing benchmark's scoring rule),
        # re-evaluated on this exact test fold for a fair comparison.
        naive_scores = x_test.sum(axis=1)
        naive_preds = (naive_scores > 0).astype(int)
        naive_m = binary_metrics(y_test, naive_preds)
        naive_m["roc_auc"] = roc_auc_score(y_test, naive_scores)
        naive_m["pr_auc"] = average_precision_score(y_test, naive_scores)
        fold_metrics["naive_sum"].append(naive_m)

        # Logistic regression on the SAME 4 features, fit on train only.
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)
        logreg_probs = clf.predict_proba(x_test)[:, 1]
        logreg_preds = (logreg_probs >= 0.5).astype(int)
        logreg_m = binary_metrics(y_test, logreg_preds)
        logreg_m["roc_auc"] = roc_auc_score(y_test, logreg_probs)
        logreg_m["pr_auc"] = average_precision_score(y_test, logreg_probs)
        fold_metrics["logreg"].append(logreg_m)
        fold_coefs.append({kind: float(c) for kind, c in zip(KIND_ORDER, clf.coef_[0])})

        print(f"  naive_sum: ROC-AUC={naive_m['roc_auc']:.4f}  precision={naive_m['precision']:.4f}  recall={naive_m['recall']:.4f}")
        print(f"  logreg:    ROC-AUC={logreg_m['roc_auc']:.4f}  precision={logreg_m['precision']:.4f}  recall={logreg_m['recall']:.4f}")
        print(f"  learned coefficients: {fold_coefs[-1]}")

    def mean_std(key, variant):
        vals = [m[key] for m in fold_metrics[variant]]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {"k": args.k, "n_positive": n, "n_negative": len(sampled_negatives), "variants": {}}

    print(f"\n===== Summary (k={args.k}-fold) =====")
    for variant in ("naive_sum", "logreg"):
        entry = {}
        for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"):
            m, s = mean_std(key, variant)
            entry[key] = {"mean": m, "std": s}
        summary["variants"][variant] = entry
        print(f"  [{variant:10s}] ROC-AUC {entry['roc_auc']['mean']:.4f}+-{entry['roc_auc']['std']:.4f}  "
              f"precision {entry['precision']['mean']:.4f}+-{entry['precision']['std']:.4f}  "
              f"recall {entry['recall']['mean']:.4f}+-{entry['recall']['std']:.4f}  "
              f"F1 {entry['f1']['mean']:.4f}+-{entry['f1']['std']:.4f}")

    mean_coefs = {kind: statistics.mean(c[kind] for c in fold_coefs) for kind in KIND_ORDER}
    summary["mean_learned_coefficients"] = mean_coefs
    print(f"\n  Mean learned coefficients (higher = more predictive weight): {mean_coefs}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
