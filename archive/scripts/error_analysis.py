import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score


NUM_CLASSES = 4486


def main():

    with open("data/tensorized/ade_vocab.json") as f:
        vocab = json.load(f)

    reverse_vocab = {
        v: k
        for k, v in vocab.items()
    }

    history = []

    preds = []
    targets = []

    print("Loading validation predictions...")

    data = torch.load(
        "checkpoints/val_predictions.pt",
        weights_only=False,
    )

    preds = data["preds"]
    targets = data["targets"]

    print("Computing per-class AP...")

    aps = []

    for i in range(NUM_CLASSES):

        y_true = targets[:, i]
        y_pred = preds[:, i]

        if y_true.sum() == 0:
            continue

        try:

            ap = average_precision_score(
                y_true,
                y_pred,
            )

            aps.append(
                (
                    ap,
                    i,
                    int(y_true.sum()),
                )
            )

        except:
            pass

    aps.sort(reverse=True)

    print("\nTOP 20 BEST ADEs")
    print("=" * 80)

    for ap, idx, count in aps[:20]:

        print(
            f"{ap:.4f}   "
            f"{count:6d}   "
            f"{reverse_vocab[idx]}"
        )

    print("\nBOTTOM 20 ADEs")
    print("=" * 80)

    for ap, idx, count in aps[-20:]:

        print(
            f"{ap:.4f}   "
            f"{count:6d}   "
            f"{reverse_vocab[idx]}"
        )


if __name__ == "__main__":
    main()