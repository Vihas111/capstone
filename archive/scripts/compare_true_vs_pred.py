import json
import torch


def main():

    with open("data/tensorized/ade_vocab.json") as f:
        vocab = json.load(f)

    reverse_vocab = {
        v: k
        for k, v in vocab.items()
    }

    data = torch.load(
        "checkpoints/val_predictions.pt",
        weights_only=False,
    )

    y_true = data["targets"]
    y_pred = data["preds"]

    idx = 0

    true_labels = torch.where(
        torch.tensor(y_true[idx]) > 0
    )[0]

    top10 = torch.topk(
        torch.tensor(y_pred[idx]),
        k=10,
    )

    print("\nTRUE LABELS")
    print("=" * 60)

    for x in true_labels:

        print(reverse_vocab[x.item()])

    print("\nTOP10 PREDICTIONS")
    print("=" * 60)

    for score, x in zip(
        top10.values,
        top10.indices,
    ):

        print(
            f"{score.item():.4f} "
            f"{reverse_vocab[x.item()]}"
        )


if __name__ == "__main__":
    main()