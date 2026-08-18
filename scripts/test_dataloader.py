import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch_geometric.loader import DataLoader
from data_utils.shard_dataset import ShardDataset


dataset = ShardDataset(
    "data/splits/train.txt"
)

loader = DataLoader(
    dataset,
    batch_size=4,
)

batch = next(iter(loader))

print(batch)
print()
print("Node types:")
print(batch.node_types)
print()
print("Edge types:")
print(batch.edge_types)
