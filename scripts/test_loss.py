import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from data_utils.shard_dataset import ShardDataset
from models.rgcn_baseline import RGCNBaseline


NUM_ADE_CLASSES = 4486
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    dataset = ShardDataset("data/splits/train.txt")

    loader = DataLoader(
        dataset,
        batch_size=4,
    )

    batch = next(iter(loader))
    batch = batch.to(DEVICE)

    model = RGCNBaseline(
        batch.metadata(),
        out_dim=NUM_ADE_CLASSES,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    logits = model(batch)

    print("Logits shape :", logits.shape)
    print("Targets shape:", batch.y.shape)

    loss = criterion(
        logits,
        batch.y
    )

    print("Loss:", loss.item())

    loss.backward()

    print("Backward pass successful!")


if __name__ == "__main__":
    main()