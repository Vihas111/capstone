import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader

from data_utils.shard_dataset import ShardDataset
from models.rgcn_baseline import RGCNBaseline


NUM_ADE_CLASSES = 4486
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    dataset = ShardDataset("data/splits/val.txt")

    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # USE GLOBAL METADATA (same as training)
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

    model.load_state_dict(
        torch.load(
            "checkpoints/rgcn_smoke_test.pt",
            map_location=DEVICE,
            weights_only=False,
        )
    )

    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_probs = []
    losses = []

    with torch.no_grad():

        for step, batch in enumerate(loader):

            batch = batch.to(DEVICE)

            logits = model(batch)

            loss = criterion(logits, batch.y)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)

            all_targets.append(batch.y.cpu())
            all_probs.append(probs.cpu())

            if step % 20 == 0:
                print(f"Processed batch {step}")

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_probs).numpy()

    torch.save(
    {
        "preds": y_pred,
        "targets": y_true,
    },
        "checkpoints/val_predictions.pt",
    )

    print(
        "\nSaved validation predictions."
    )
    print("\nValidation loss:")
    print(sum(losses) / len(losses))

    print("\nComputing metrics...")

    micro_roc = roc_auc_score(
        y_true,
        y_pred,
        average="micro",
    )

    micro_pr = average_precision_score(
        y_true,
        y_pred,
        average="micro",
    )

    print(f"Micro ROC-AUC: {micro_roc:.4f}")
    print(f"Micro PR-AUC : {micro_pr:.4f}")


if __name__ == "__main__":
    main()