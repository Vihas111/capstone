import torch
import numpy as np


def main():

    data = torch.load(
        "checkpoints/val_predictions.pt",
        weights_only=False,
    )

    y_true = data["targets"]
    y_pred = data["preds"]

    topk = 10

    hits = []

    for i in range(len(y_true)):

        true_set = set(
            np.where(y_true[i] > 0)[0]
        )

        pred_set = set(
            np.argsort(y_pred[i])[-topk:]
        )

        overlap = len(
            true_set & pred_set
        )

        hits.append(overlap)

    hits = np.array(hits)

    print("\nPrediction quality")
    print("=" * 60)

    print(
        "Mean hits@10:",
        hits.mean()
    )

    print(
        "Max hits@10:",
        hits.max()
    )

    print(
        "Min hits@10:",
        hits.min()
    )

    print(
        "Graphs with ≥5 hits:",
        (hits >= 5).sum()
    )

    print(
        "Graphs with 0 hits:",
        (hits == 0).sum()
    )


if __name__ == "__main__":
    main()