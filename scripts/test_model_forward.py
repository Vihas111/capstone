import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from torch_geometric.loader import DataLoader

from data_utils.shard_dataset import ShardDataset
from models.rgcn_baseline import RGCNBaseline


dataset = ShardDataset(
    "data/splits/train.txt"
)

loader = DataLoader(
    dataset,
    batch_size=4,
)

batch = next(iter(loader))

model = RGCNBaseline(
    batch.metadata(),
)

out = model(batch)

print("Output shape:")
print(out.shape)
