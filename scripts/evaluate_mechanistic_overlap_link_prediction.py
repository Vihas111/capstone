"""
Evaluates scripts/mechanism_lookup.py's deterministic mechanistic-overlap
signal as a link predictor for the "known drugs, undocumented combination"
task -- the exact axis DeepDDI (Ryu et al. 2018, PNAS), SumGNN (Yu et al.
2021, Bioinformatics)'s transductive split, and the medicX KG-embedding
approach (arXiv:2308.04172) all benchmark, and the one requested after
comparing this project against that literature (see
findings/findings.md's "Comparison to published DDI models" section).

This was never measured before: mechanism_lookup.py's shared()-based
overlap check is used in production, but its actual predictive value as a
link predictor -- "does shared-protein overlap between two drugs'
DOCUMENTED profiles correlate with whether DrugBank documents them as
interacting" -- had never been benchmarked the way the literature does.

Method: sample DOCUMENTED interaction pairs (positives) and random
non-documented pairs (negatives), both restricted to drugs that have at
least one DrugBank enzyme/target/transporter/carrier profile entry (the
"known drugs" population -- this deliberately excludes the cold-start/
unseen-drug case, which is scripts/run_mechanism_kfold.py's job, not
this one). Score each pair by counting shared proteins across all 4 kinds,
computed ONLY from each drug's own profile -- blind to whether DrugBank's
interaction text exists for that specific pair. Report ROC-AUC/PR-AUC
(the same metrics DeepDDI/SumGNN/medicX report) plus a binary "any overlap"
operating point for an interpretable comparison.

Usage:
    python scripts/evaluate_mechanistic_overlap_link_prediction.py \
        --n-samples 50000 --out checkpoints/mechanistic_overlap_link_prediction_results.json
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path("data/processed")


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50000,
                    help="Number of positive pairs to sample (negatives sampled 1:1). "
                    "Capped at the actual number of restricted-population positive pairs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/mechanistic_overlap_link_prediction_results.json")
    return p.parse_args()


def build_profile_by_drug():
    """{drug_id: {"enzyme": set(protein_names), "target": ..., "transporter": ..., "carrier": ...}}"""

    tables = {
        "enzyme": ("drug_enzymes.jsonl", "enzyme"),
        "target": ("drug_targets.jsonl", "target"),
        "transporter": ("drug_transporters.jsonl", "transporter"),
        "carrier": ("drug_carriers.jsonl", "carrier"),
    }

    profile_by_drug = {}

    for kind, (filename, key_field) in tables.items():
        for row in load_jsonl(DATA_DIR / filename):
            drug = row["drug"]
            profile_by_drug.setdefault(drug, {"enzyme": set(), "target": set(), "transporter": set(), "carrier": set()})
            profile_by_drug[drug][kind].add(row[key_field])

    return profile_by_drug


def overlap_score(profile_by_drug, a, b):
    """Total shared-protein count across all 4 kinds -- the graded score.
    Also returns per-kind counts for the breakdown."""

    pa, pb = profile_by_drug[a], profile_by_drug[b]
    per_kind = {kind: len(pa[kind] & pb[kind]) for kind in ("enzyme", "target", "transporter", "carrier")}
    return sum(per_kind.values()), per_kind


def main():

    args = parse_args()
    rng = random.Random(args.seed)

    print("Loading per-drug profiles...")
    profile_by_drug = build_profile_by_drug()
    any_profile = set(profile_by_drug.keys())
    print(f"  {len(any_profile)} drugs have at least one profile entry.")

    print("Scanning documented interactions (restricted to the profile-bearing population)...")
    positive_pairs = set()
    with open(DATA_DIR / "drugbank_interactions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            a, b = row["drug1"], row["drug2"]
            if a in any_profile and b in any_profile and a != b:
                positive_pairs.add(frozenset((a, b)))
    print(f"  {len(positive_pairs)} restricted-population documented pairs found.")

    n = min(args.n_samples, len(positive_pairs))
    sampled_positives = rng.sample(sorted(positive_pairs, key=lambda s: sorted(s)), n)
    print(f"Sampled {n} positive pairs.")

    print(f"Sampling {n} negative pairs (random non-documented pairs, same population)...")
    profile_list = sorted(any_profile)
    negatives = set()
    attempts = 0
    while len(negatives) < n and attempts < n * 50:
        a, b = rng.sample(profile_list, 2)
        pair = frozenset((a, b))
        if pair not in positive_pairs and pair not in negatives:
            negatives.add(pair)
        attempts += 1
    sampled_negatives = list(negatives)
    print(f"  Sampled {len(sampled_negatives)} negative pairs ({attempts} attempts).")

    print("Scoring all pairs by shared-protein overlap (blind to interaction text)...")
    scores, labels, per_kind_scores = [], [], {"enzyme": [], "target": [], "transporter": [], "carrier": []}

    for pair in sampled_positives:
        a, b = tuple(pair)
        total, per_kind = overlap_score(profile_by_drug, a, b)
        scores.append(total)
        labels.append(1)
        for kind, count in per_kind.items():
            per_kind_scores[kind].append(count)

    for pair in sampled_negatives:
        a, b = tuple(pair)
        total, per_kind = overlap_score(profile_by_drug, a, b)
        scores.append(total)
        labels.append(0)
        for kind, count in per_kind.items():
            per_kind_scores[kind].append(count)

    from sklearn.metrics import average_precision_score, roc_auc_score

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    # Binary "any overlap" operating point, for an interpretable number
    # comparable to DeepDDI/medicX's reported accuracy/F1 (a single
    # threshold), not just a probabilistic ranking metric.
    preds_any_overlap = [1 if s > 0 else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds_any_overlap, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds_any_overlap, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds_any_overlap, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds_any_overlap, labels) if p == 0 and l == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels)

    print(f"\n===== Results (n_positive={n}, n_negative={len(sampled_negatives)}) =====")
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(f"'Any overlap' operating point: accuracy={accuracy:.4f}  precision={precision:.4f}  "
          f"recall={recall:.4f}  F1={f1:.4f}")

    per_kind_auc = {}
    for kind, kind_scores in per_kind_scores.items():
        try:
            k_roc = roc_auc_score(labels, kind_scores)
            k_pr = average_precision_score(labels, kind_scores)
        except ValueError:
            k_roc, k_pr = float("nan"), float("nan")
        per_kind_auc[kind] = {"roc_auc": k_roc, "pr_auc": k_pr}
        print(f"  [{kind:12s}] ROC-AUC={k_roc:.4f}  PR-AUC={k_pr:.4f}")

    results = {
        "n_positive": n,
        "n_negative": len(sampled_negatives),
        "population_size": len(any_profile),
        "total_restricted_positive_pairs": len(positive_pairs),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "any_overlap_operating_point": {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        },
        "per_kind": per_kind_auc,
        "reference_literature": {
            "DeepDDI_PNAS2018": {"accuracy": 0.924, "note": "86-class DDI type, 192284 DDIs, structure-only"},
            "SumGNN_transductive_DrugBank": {"f1": 0.8685, "accuracy": 0.9266},
            "SumGNN_transductive_TWOSIDES": {"roc_auc": 0.9486, "pr_auc": 0.9335},
            "medicX_KG_embedding": {"f1": 0.9519, "note": "ComplEx+LSTM, DrugBank link prediction"},
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
