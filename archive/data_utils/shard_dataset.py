from pathlib import Path

import torch
from torch.utils.data import IterableDataset


def add_reverse_edges(data):

    edge_types = list(data.edge_types)

    for src, rel, dst in edge_types:

        edge_index = data[(src, rel, dst)].edge_index

        rev_rel = f"rev_{rel}"

        rev_edge_index = edge_index.flip(0)

        data[(dst, rev_rel, src)].edge_index = rev_edge_index

    return data


class ShardDataset(IterableDataset):

    def __init__(self, split_file):

        with open(split_file) as f:
            self.shards = [
                Path(line.strip())
                for line in f
                if line.strip()
            ]

    def __iter__(self):

        for shard_path in self.shards:

            graphs = torch.load(
                shard_path,
                weights_only=False
            )

            for graph in graphs:

                yield add_reverse_edges(graph)