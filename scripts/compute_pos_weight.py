import json
from pathlib import Path

import torch


NUM_CLASSES = 4486


def main():

    positives = torch.zeros(NUM_CLASSES)
    total_graphs = 0

    shards = sorted(
        Path("data/tensorized_v2").glob("shard_*.pt")
    )

    for shard in shards:

        print(f"Processing {shard.name}")

        graphs = torch.load(
            shard,
            weights_only=False,
        )

        for g in graphs:

            positives += g.y.squeeze(0)
            total_graphs += 1

    negatives = total_graphs - positives

    pos_weight = negatives / (positives + 1.0)

    # Prevent ultra-rare classes from exploding the loss
    pos_weight = torch.clamp(
        pos_weight,
        min=1.0,
        max=100.0,
    )

    torch.save(
        pos_weight,
        "data/processed/pos_weight.pt",
    )

    stats = {
        "num_classes": NUM_CLASSES,
        "num_graphs": total_graphs,
        "min_weight": float(pos_weight.min()),
        "max_weight": float(pos_weight.max()),
        "mean_weight": float(pos_weight.mean()),
    }

    with open(
        "data/processed/pos_weight_stats.json",
        "w",
    ) as f:
        json.dump(stats, f, indent=2)

    print("\nSaved:")
    print("  data/processed/pos_weight.pt")
    print("  data/processed/pos_weight_stats.json")
    print(stats)


if __name__ == "__main__":
    main()