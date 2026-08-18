import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from torch_geometric.loader import DataLoader

from data_utils.shard_dataset import ShardDataset
from models.rgcn_baseline import RGCNBaseline


NUM_ADE_CLASSES = 4486
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def precision_at_k(y_true, y_pred, k=10):

    total = 0.0

    for i in range(len(y_true)):

        top_k = y_pred[i].argsort()[-k:]

        hits = y_true[i][top_k].sum()

        total += hits / k

    return total / len(y_true)


def recall_at_k(y_true, y_pred, k=10):

    total = 0.0

    for i in range(len(y_true)):

        positives = y_true[i].sum()

        if positives == 0:
            continue

        top_k = y_pred[i].argsort()[-k:]

        hits = y_true[i][top_k].sum()

        total += hits / positives

    return total / len(y_true)


def hits_at_k(y_true, y_pred, k=10):

    total = 0

    for i in range(len(y_true)):

        top_k = y_pred[i].argsort()[-k:]

        hits = y_true[i][top_k].sum()

        if hits > 0:
            total += 1

    return total / len(y_true)


def main():

    dataset = ShardDataset(
        "data/splits/test.txt"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
    )

    metadata = torch.load(
        "data/processed/global_metadata.pt",
        weights_only=False,
    )

    model = RGCNBaseline(
        metadata,
        hidden_dim=128,
        out_dim=NUM_ADE_CLASSES,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "checkpoints/best_rgcn.pt",
            map_location=DEVICE,
            weights_only=True,
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

            if step % 20 == 0:
                print(f"Processed batch {step}")

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_probs).numpy()

    print("\nTEST LOSS")
    print("=" * 60)
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

    p10 = precision_at_k(
        y_true,
        y_pred,
        k=10,
    )

    r10 = recall_at_k(
        y_true,
        y_pred,
        k=10,
    )

    h10 = hits_at_k(
        y_true,
        y_pred,
        k=10,
    )

    print(f"\nMicro ROC-AUC : {micro_roc:.4f}")
    print(f"Micro PR-AUC  : {micro_pr:.4f}")

    print("\nRanking metrics")
    print("=" * 60)
    print(f"Precision@10 : {p10:.4f}")
    print(f"Recall@10    : {r10:.4f}")
    print(f"Hits@10      : {h10:.4f}")


if __name__ == "__main__":
    main()