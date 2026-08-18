"""
label_freq scan confirmed: 32.3% of the ~4486 ADE classes have <5 positive
examples across the ENTIRE 172,754-graph dataset, 43.1% have <10, median is
16. Macro-AP averaged over the full label space is dominated by classes that
are close to unlearnable from data volume alone -- this is almost certainly
why macro-AP went flat once the model stopped overfitting (loss and micro-AP
both improving normally, macro-AP frozen).

This mirrors your own project's precedent: build_mechanism_labels.py already
uses --min-freq 10 on the mechanism-prediction track for the same reason.
This script builds the equivalent for the ADE-GNN track: a boolean mask of
which classes have >= --min-freq positive examples DATASET-WIDE, so
train_v2.py can report/checkpoint against a denser, more honest subset while
still training the full 4486-way head (no change to the model's output
space or downstream JSON contract -- this only affects which classes count
toward the macro-AP used for model selection).

Usage:
    python scripts/build_label_mask.py --min-freq 10 \
        --shard-dir data/tensorized_v2 --out data/processed/label_mask.json
"""

import argparse
import json
from pathlib import Path

import torch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="data/tensorized_v2")
    parser.add_argument("--min-freq", type=int, default=10)
    parser.add_argument("--out", default="data/processed/label_mask.json")
    args = parser.parse_args()

    shard_files = sorted(Path(args.shard_dir).glob("shard_*.pt"))
    if not shard_files:
        raise SystemExit(f"No shards found under {args.shard_dir}")

    print(f"Scanning {len(shard_files)} shards for label frequency...")

    positives = None
    total = 0

    for shard_path in shard_files:
        graphs = torch.load(shard_path, weights_only=False)
        for g in graphs:
            y = g.y.squeeze(0)
            if positives is None:
                positives = torch.zeros_like(y)
            positives += y
            total += 1

    num_classes = positives.numel()
    mask = (positives >= args.min_freq).tolist()
    n_kept = sum(mask)

    with open(args.out, "w") as f:
        json.dump(
            {
                "min_freq": args.min_freq,
                "num_classes": num_classes,
                "num_kept": n_kept,
                "mask": mask,
            },
            f,
        )

    print(
        f"\nKept {n_kept}/{num_classes} classes "
        f"({100 * n_kept / num_classes:.1f}%) with >= {args.min_freq} "
        f"positive examples across {total} graphs."
    )
    print(f"Saved: {args.out}")
    print(
        "\ntrain_v2.py's --label-mask flag will use this to restrict "
        "macro-AP to this denser subset. The model's output head is "
        "unaffected -- it still predicts all classes."
    )


if __name__ == "__main__":
    main()
