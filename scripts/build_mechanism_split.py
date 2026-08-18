"""
Cold-start-by-drug split for the mechanism gap-filler dataset (per-drug
multi-label classification -- simpler than scripts/build_drug_split.py's
pair-level cold start, since there's no "pair" concept here at all, just
individual drugs).

Splits are grouped by IDENTICAL fingerprint bit-vector before shuffling, not
by individual drug: a handful of DrugBank entries share the exact same
structure (e.g. different salt/formulation entries for the same molecule),
and letting near-duplicate structures land on both sides of train/test would
let the model "cheat" by memorizing a structure it technically saw in a
different guise during training.

Usage:
    python scripts/build_mechanism_split.py \
        --dataset-path data/processed/mechanism_dataset.pt \
        --fingerprints-path data/processed/drug_fingerprints.pt \
        --out data/processed/mechanism_drug_split.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# remaining ~0.10 -> test


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--out", default="data/processed/mechanism_drug_split.json")
    return p.parse_args()


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids = dataset["drug_ids"]

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    # Group drugs by identical fingerprint bit-vector.
    groups = defaultdict(list)
    for drug in drug_ids:
        key = tuple(fp_by_drug[drug].tolist())
        groups[key].append(drug)

    group_list = list(groups.values())
    n_dup_groups = sum(1 for g in group_list if len(g) > 1)
    n_dup_drugs = sum(len(g) for g in group_list if len(g) > 1)
    print(f"{len(drug_ids)} drugs -> {len(group_list)} distinct-fingerprint groups "
          f"({n_dup_groups} groups covering {n_dup_drugs} drugs share a fingerprint "
          f"with at least one other drug).")

    rng = random.Random(SEED)
    rng.shuffle(group_list)

    n = len(group_list)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_groups = group_list[:train_end]
    val_groups = group_list[train_end:val_end]
    test_groups = group_list[val_end:]

    split = {
        "train": [d for g in train_groups for d in g],
        "val": [d for g in val_groups for d in g],
        "test": [d for g in test_groups for d in g],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    print("\nMechanism drug split (cold-start, grouped by identical structure):")
    for name, ids in split.items():
        print(f"  {name:5s} {len(ids)} drugs")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
