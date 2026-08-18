"""
create_splits.py splits at the SHARD level with plain random.shuffle -- it
says nothing about which drugs land where, so the same drug can (and almost
certainly does) appear in both train and val/test shards. That inflates
validation metrics: the model can do well on a pair partly by having already
seen one of the two drugs during training, which is the specific failure
mode the memory note "cold-start evaluation splits ... important for
realistic model evaluation" is about.

This script instead splits at the DRUG level: every individual drug is
assigned to exactly one of train/val/test. A pair (graph) is only usable in a
split if BOTH its drugs belong to that split's drug set; pairs that straddle
two drug-splits are dropped entirely rather than leaked into either side.
This is strict cold-start (holds out entire drugs) and will discard some
pairs -- that's the intended, honest tradeoff, not a bug.

Usage:
    python scripts/build_drug_split.py --shard-dir data/tensorized_v2 \
        --out data/processed/drug_split.json

Output format:
    {"train": [id, id, ...], "val": [...], "test": [...]}

data_utils/shard_dataset_v2.py's ShardDatasetV2 consumes this directly via
its `drug_split` / `split_name` arguments.
"""

import argparse
import json
import random
from pathlib import Path

import torch


SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# remaining ~0.10 -> test


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="data/tensorized_v2")
    parser.add_argument(
        "--out", default="data/processed/drug_split.json"
    )
    args = parser.parse_args()

    shard_files = sorted(Path(args.shard_dir).glob("shard_*.pt"))

    if not shard_files:
        raise SystemExit(
            f"No shards found under {args.shard_dir} -- check --shard-dir"
        )

    print(f"Scanning {len(shard_files)} shards for the full drug vocabulary...")

    all_drugs = set()

    for shard_path in shard_files:

        graphs = torch.load(shard_path, weights_only=False)

        for graph in graphs:
            x = graph["drug"].x
            all_drugs.update(int(i) for i in x.squeeze(-1).tolist())

    all_drugs = sorted(all_drugs)
    print(f"Found {len(all_drugs)} distinct drugs across all shards.")

    rng = random.Random(SEED)
    rng.shuffle(all_drugs)

    n = len(all_drugs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    split = {
        "train": all_drugs[:train_end],
        "val": all_drugs[train_end:val_end],
        "test": all_drugs[val_end:],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    print("\nDrug split (cold-start, held out at the DRUG level):")
    for name, drugs in split.items():
        print(f"  {name:5s} {len(drugs)} drugs")
    print(
        "\nNote: pairs where the two drugs fall in different splits are "
        "dropped by ShardDatasetV2 at load time, not here -- this file only "
        "records which drugs belong to which split."
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
