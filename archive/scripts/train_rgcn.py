import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from data_utils.shard_dataset import ShardDataset
from models.rgcn_baseline import RGCNBaseline


NUM_ADE_CLASSES = 4486
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_memory_gb():
    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.memory_allocated() / 1024**3


def main():

    dataset = ShardDataset("data/splits/train.txt")

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Use GLOBAL metadata, never first_batch.metadata()
    # ---------------------------------------------------------
    metadata = torch.load(
        "data/processed/global_metadata.pt",
        weights_only=False,
    )

    model = RGCNBaseline(
        metadata,
        hidden_dim=128,
        out_dim=NUM_ADE_CLASSES,
    ).to(DEVICE)
    # ---------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.BCEWithLogitsLoss()

    model.train()

    running_loss = 0.0

    for step, batch in enumerate(loader):

        # Smoke test: only first 1000 graphs
        if step >= 125:
            break

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch)

        loss = criterion(
            logits,
            batch.y,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if step % 10 == 0:

            print(
                f"Step {step:03d} | "
                f"Loss {loss.item():.4f} | "
                f"GPU {gpu_memory_gb():.2f} GB"
            )

    print("\nFinished smoke test.")
    print(
        f"Average loss: "
        f"{running_loss / (step + 1):.4f}"
    )

    Path("checkpoints").mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        "checkpoints/rgcn_smoke_test.pt",
    )

    print(
        "Saved: checkpoints/rgcn_smoke_test.pt"
    )


if __name__ == "__main__":
    main()