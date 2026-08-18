"""
Self-training step 2: merges data/processed/mechanism_pseudo_labels.pt into
an AUGMENTED training set. Pseudo-labeled drugs are added ONLY to train --
val/test stay real-labels-only and untouched, so before/after comparison on
the original test split remains trustworthy (an augmented "test" set would
include unverifiable pseudo-labels).

scripts/train_mechanism_predictor.py needs NO changes to consume this --
augmentation happens via data-file composition, read through its existing
--dataset-path/--split-path flags.

Usage:
    python scripts/merge_mechanism_pseudo_labels.py \
        --dataset-out data/processed/mechanism_dataset_augmented.pt \
        --split-out data/processed/mechanism_drug_split_augmented.json
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--split-path", default="data/processed/mechanism_drug_split.json")
    p.add_argument("--pseudo-labels-path", default="data/processed/mechanism_pseudo_labels.pt")
    p.add_argument("--dataset-out", default="data/processed/mechanism_dataset_augmented.pt")
    p.add_argument("--split-out", default="data/processed/mechanism_drug_split_augmented.json")
    return p.parse_args()


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    pseudo = torch.load(args.pseudo_labels_path, weights_only=False)

    with open(args.split_path) as f:
        split = json.load(f)

    overlap = set(dataset["drug_ids"]) & set(pseudo["drug_ids"])
    if overlap:
        raise SystemExit(f"{len(overlap)} pseudo-labeled drugs overlap with the real "
                          f"dataset -- this should be impossible (pseudo-label pool is "
                          f"defined as excluded from mechanism_dataset.pt). Aborting.")

    augmented_drug_ids = dataset["drug_ids"] + pseudo["drug_ids"]
    augmented_labels = torch.cat([dataset["labels"], pseudo["labels"]], dim=0)

    augmented_split = {
        "train": split["train"] + pseudo["drug_ids"],
        "val": split["val"],
        "test": split["test"],
    }

    Path(args.dataset_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"drug_ids": augmented_drug_ids, "labels": augmented_labels}, args.dataset_out)
    with open(args.split_out, "w") as f:
        json.dump(augmented_split, f, indent=2)

    print(f"Augmented dataset: {len(augmented_drug_ids)} drugs "
          f"({len(dataset['drug_ids'])} real + {len(pseudo['drug_ids'])} pseudo-labeled).")
    print(f"Augmented train split: {len(augmented_split['train'])} drugs "
          f"({len(split['train'])} real + {len(pseudo['drug_ids'])} pseudo-labeled). "
          f"val ({len(split['val'])}) and test ({len(split['test'])}) unchanged, real-labels-only.")
    print(f"Saved: {args.dataset_out}")
    print(f"Saved: {args.split_out}")


if __name__ == "__main__":
    main()
