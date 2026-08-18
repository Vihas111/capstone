"""
Computes the ACTUAL max node index per node type across every tensorized
shard, so models/rgcn_v2.py can size its embedding tables correctly instead
of the baseline's hardcoded `nn.Embedding(5000, hidden_dim)` for everything.

This is intentionally derived from the data on disk rather than trusted to
match `len(drugs.jsonl)` from memory/CLAUDE.md, in case the id scheme used
when building the graphs (build_hetero_graph.py / generate_pair_graphs.py)
isn't a dense 0..N-1 range for every node type.

Usage:
    python scripts/build_vocab_sizes.py --shard-dir data/tensorized_v2 \
        --out data/processed/vocab_sizes.json
"""

import argparse
import json
from pathlib import Path

import torch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="data/tensorized_v2")
    parser.add_argument(
        "--out", default="data/processed/vocab_sizes.json"
    )
    args = parser.parse_args()

    shard_files = sorted(Path(args.shard_dir).glob("shard_*.pt"))

    if not shard_files:
        raise SystemExit(
            f"No shards found under {args.shard_dir} -- check --shard-dir"
        )

    print(f"Scanning {len(shard_files)} shards...")

    max_idx = {}

    for i, shard_path in enumerate(shard_files):

        graphs = torch.load(shard_path, weights_only=False)

        for graph in graphs:
            for node_type in graph.node_types:

                x = graph[node_type].x
                if x is None or x.numel() == 0:
                    continue

                local_max = int(x.squeeze(-1).max())

                if node_type not in max_idx or local_max > max_idx[node_type]:
                    max_idx[node_type] = local_max

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(shard_files)} shards scanned")

    # vocab size = max index + 1 (indices are 0-based)
    vocab_sizes = {k: v + 1 for k, v in max_idx.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(vocab_sizes, f, indent=2)

    print("\nVocab sizes:")
    for k, v in sorted(vocab_sizes.items()):
        print(f"  {k:15s} {v}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
