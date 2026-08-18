"""
Generalizes scripts/build_mechanism_split.py's "group by identical
fingerprint, then shuffle and partition" logic to k folds, for an honest
mean+-std macro/micro-AP instead of one point estimate from a single
388-drug test split (small enough to be noisy on its own -- see
findings/findings.md).

Same grouping rationale as build_mechanism_split.py: a handful of DrugBank
entries share the exact same structure (different salt/formulation entries
for the same molecule); grouping keeps each such cluster entirely within
one fold so near-duplicate structures can't leak across train/val/test.

Usage:
    python scripts/build_mechanism_kfold_splits.py --k 5 --seed 42 \
        --out data/processed/mechanism_kfold_splits.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed/mechanism_kfold_splits.json")
    return p.parse_args()


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids = dataset["drug_ids"]

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    groups = defaultdict(list)
    for drug in drug_ids:
        key = tuple(fp_by_drug[drug].tolist())
        groups[key].append(drug)

    group_list = list(groups.values())
    print(f"{len(drug_ids)} drugs -> {len(group_list)} distinct-fingerprint groups.")

    rng = random.Random(args.seed)
    rng.shuffle(group_list)

    k = args.k
    chunks = [[] for _ in range(k)]
    for i, group in enumerate(group_list):
        chunks[i % k].extend(group)

    folds = []
    for i in range(k):
        test_ids = chunks[i]
        val_ids = chunks[(i + 1) % k]
        train_ids = [d for j, c in enumerate(chunks) if j not in (i, (i + 1) % k) for d in c]
        folds.append({"train": train_ids, "val": val_ids, "test": test_ids})
        print(f"  fold {i}: train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")

    out = {"k": k, "seed": args.seed, "folds": folds}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
