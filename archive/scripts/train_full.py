import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from data_utils.shard_dataset import ShardDataset
from models.rgcn_baseline import RGCNBaseline


NUM_ADE_CLASSES = 4486
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
NUM_EPOCHS = 30
PATIENCE = 7
LR = 1e-3
HIDDEN_DIM = 128


def gpu_memory_gb():

    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.memory_allocated() / 1024**3


def evaluate(model, loader, criterion):

    model.eval()

    losses = []
    all_targets = []
    all_probs = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(DEVICE)

            logits = model(batch)

            loss = criterion(
                logits,
                batch.y,
            )

            losses.append(loss.item())

            probs = torch.sigmoid(logits)

            all_targets.append(
                batch.y.cpu()
            )

            all_probs.append(
                probs.cpu()
            )

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_probs).numpy()

    pr_auc = average_precision_score(
        y_true,
        y_pred,
        average="micro",
    )

    return {
        "loss": sum(losses) / len(losses),
        "pr_auc": pr_auc,
    }


def main():

    print("Loading datasets...")

    train_dataset = ShardDataset(
        "data/splits/train.txt"
    )

    val_dataset = ShardDataset(
        "data/splits/val.txt"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )

    print("Loading global metadata...")

    metadata = torch.load(
        "data/processed/global_metadata.pt",
        weights_only=False,
    )

    model = RGCNBaseline(
        metadata,
        hidden_dim=HIDDEN_DIM,
        out_dim=NUM_ADE_CLASSES,
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.BCEWithLogitsLoss()

    Path("checkpoints").mkdir(
        exist_ok=True
    )

    best_pr_auc = 0.0
    patience_counter = 0

    history = []

    for epoch in range(NUM_EPOCHS):

        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

        model.train()

        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(train_loader):

            batch = batch.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(batch)

            loss = criterion(
                logits,
                batch.y,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if step % 200 == 0:

                print(
                    f"Step {step:05d} | "
                    f"Loss {loss.item():.4f} | "
                    f"GPU {gpu_memory_gb():.2f} GB"
                )

        train_loss = running_loss / steps

        print("\nRunning validation...")

        metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        val_loss = metrics["loss"]
        val_pr_auc = metrics["pr_auc"]

        print(
            f"\nEpoch {epoch + 1} summary:"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            f"Val PR-AUC: {val_pr_auc:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
            }
        )

        with open(
            "checkpoints/training_history.json",
            "w",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

        if val_pr_auc > best_pr_auc:

            best_pr_auc = val_pr_auc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "checkpoints/best_rgcn.pt",
            )

            print(
                "Saved new best checkpoint."
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    print("\nTraining finished.")
    print(
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )


if __name__ == "__main__":
    main()