"""
Computes pos_weight for the mechanism gap-filler, from an EXPLICIT train-id
list only -- fixes a leakage bug in scripts/build_mechanism_labels.py, which
computed pos_weight over the full 3,880-drug population (train+val+test
combined) before scripts/build_mechanism_split.py even existed in the
pipeline. Inconsistent with this project's own stated discipline
(train_mechanism_predictor.py already computes prior_logits train-only for
exactly this reason -- "avoid leaking val/test statistics into model init").

Also supports --per-category: separately-clamped pos_weight for
enzyme/transporter labels vs target labels (split by the label vocab's
"<kind>:" prefix), for the Phase 2 ablation in findings/findings.md's
follow-up work. NOTE: measured directly against this dataset, target labels
already average a HIGHER pos_weight than enzyme/transporter under the
uniform formula (88.6% of target labels are already clamped at the ceiling)
-- transporter is the empirically weak kind, not target. --per-category
lets you raise transporter's ceiling specifically; it is NOT a "PD needs
more weight" knob by default.

Usage:
    python scripts/compute_mechanism_pos_weight.py \
        --dataset-path data/processed/mechanism_dataset.pt \
        --split-path data/processed/mechanism_drug_split.json --split train \
        --out data/processed/mechanism_pos_weight.pt

    python scripts/compute_mechanism_pos_weight.py --per-category \
        --transporter-clamp-max 200 --out /tmp/pcw_test.pt
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--split-path", default="data/processed/mechanism_drug_split.json")
    p.add_argument("--split", default="train", help="Which key of --split-path to use as the train population.")
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--clamp-min", type=float, default=1.0)
    p.add_argument("--clamp-max", type=float, default=100.0)
    p.add_argument("--per-category", action="store_true",
                    help="Clamp enzyme/transporter vs target labels separately.")
    p.add_argument("--enzyme-clamp-max", type=float, default=None, help="Overrides --clamp-max for enzyme labels.")
    p.add_argument("--transporter-clamp-max", type=float, default=None, help="Overrides --clamp-max for transporter labels.")
    p.add_argument("--target-clamp-max", type=float, default=None, help="Overrides --clamp-max for target labels.")
    p.add_argument("--out", default="data/processed/mechanism_pos_weight.pt")
    p.add_argument("--stats-out", default=None, help="Optional path to also save summary stats JSON.")
    return p.parse_args()


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids, label_matrix = dataset["drug_ids"], dataset["labels"]
    label_by_drug = dict(zip(drug_ids, label_matrix))

    with open(args.split_path) as f:
        split = json.load(f)
    train_ids = [d for d in split[args.split] if d in label_by_drug]

    y_train = torch.stack([label_by_drug[d] for d in train_ids])
    n = y_train.shape[0]

    positives = y_train.sum(dim=0)
    negatives = n - positives
    raw = negatives / (positives + 1.0)

    if not args.per_category:
        pos_weight = torch.clamp(raw, min=args.clamp_min, max=args.clamp_max)
    else:
        with open(args.vocab_path) as f:
            vocab = json.load(f)["labels"]

        clamp_max_by_kind = {
            "enzyme": args.enzyme_clamp_max or args.clamp_max,
            "transporter": args.transporter_clamp_max or args.clamp_max,
            "target": args.target_clamp_max or args.clamp_max,
        }

        pos_weight = torch.empty_like(raw)
        for i, label in enumerate(vocab):
            kind = label.split(":", 1)[0]
            pos_weight[i] = torch.clamp(
                raw[i], min=args.clamp_min, max=clamp_max_by_kind[kind]
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pos_weight, args.out)

    stats = {
        "n_train": n,
        "min_weight": float(pos_weight.min()),
        "max_weight": float(pos_weight.max()),
        "mean_weight": float(pos_weight.mean()),
        "per_category": args.per_category,
    }

    print(f"Computed pos_weight from {n} train drugs ({args.split_path}[{args.split!r}]).")
    print(stats)
    print(f"Saved: {args.out}")

    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
