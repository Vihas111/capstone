import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter

import torch

from data_utils.shard_dataset import ShardDataset


def analyze_split(name, split_file, ade_names):

    dataset = ShardDataset(split_file)

    num_graphs = 0
    total_positives = 0
    counts = Counter()

    for graph in dataset:

        num_graphs += 1

        labels = torch.where(graph.y.flatten() > 0)[0].tolist()

        total_positives += len(labels)

        counts.update(labels)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Graphs: {num_graphs:,}")
    print(f"Positive labels: {total_positives:,}")
    print(f"Average labels/graph: {total_positives / num_graphs:.2f}")

    return counts


def main():

    with open("data/tensorized/ade_vocab.json") as f:
        ade_vocab = json.load(f)

    idx_to_ade = {
        v: k
        for k, v in ade_vocab.items()
    }

    print(f"Total ADE classes: {len(idx_to_ade):,}")

    train_counts = analyze_split(
        "TRAIN",
        "data/splits/train.txt",
        idx_to_ade,
    )

    val_counts = analyze_split(
        "VALIDATION",
        "data/splits/val.txt",
        idx_to_ade,
    )

    test_counts = analyze_split(
        "TEST",
        "data/splits/test.txt",
        idx_to_ade,
    )

    total_counts = train_counts + val_counts + test_counts

    print("\nTop 20 most frequent ADEs")
    print("=" * 60)

    for idx, count in total_counts.most_common(20):

        print(
            f"{idx_to_ade[idx]:50s} {count:8d}"
        )

    print("\nBottom 20 least frequent ADEs")
    print("=" * 60)

    least_common = sorted(
        total_counts.items(),
        key=lambda x: x[1]
    )[:20]

    for idx, count in least_common:

        print(
            f"{idx_to_ade[idx]:50s} {count:8d}"
        )


if __name__ == "__main__":
    main()