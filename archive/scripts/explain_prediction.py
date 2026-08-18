import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch_geometric.loader import DataLoader

from models.rgcn_baseline import RGCNBaseline


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_ade_names():

    with open("data/tensorized/ade_vocab.json") as f:
        vocab = json.load(f)

    return {v: k for k, v in vocab.items()}


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python scripts/explain_prediction.py "
            "<shard_id> <graph_id>"
        )
        return

    shard_id = int(sys.argv[1])
    graph_id = int(sys.argv[2])

    graphs = torch.load(
        f"data/tensorized/shard_{shard_id:03d}.pt",
        weights_only=False,
    )

    graph = graphs[graph_id]

    loader = DataLoader(
        [graph],
        batch_size=1,
    )

    batch = next(iter(loader))

    metadata = torch.load(
        "data/processed/global_metadata.pt",
        weights_only=False,
    )

    model = RGCNBaseline(
        metadata,
        hidden_dim=128,
        out_dim=4486,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "checkpoints/best_rgcn.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    batch = batch.to(DEVICE)

    with torch.no_grad():

        probs = torch.sigmoid(
            model(batch)
        )[0]

    ade_names = load_ade_names()

    true_labels = set(
        torch.where(batch.y[0] > 0)[0].cpu().tolist()
    )

    top20 = torch.topk(
        probs,
        k=20,
    )

    predicted = top20.indices.cpu().tolist()

    correct = [
        x for x in predicted
        if x in true_labels
    ]

    missed = sorted(
        true_labels - set(predicted)
    )

    print("\nTRUE ADE COUNT:", len(true_labels))
    print("TOP20 HITS:", len(correct))

    print("\nCORRECT PREDICTIONS")
    print("=" * 80)

    for idx in correct:

        print(ade_names[idx])

    print("\nMISSED ADEs")
    print("=" * 80)

    for idx in missed:

        print(ade_names[idx])

    print("\nTOP20 PREDICTIONS")
    print("=" * 80)

    for score, idx in zip(
        top20.values.cpu(),
        top20.indices.cpu(),
    ):

        mark = "✓" if idx.item() in true_labels else " "

        print(
            f"{mark} "
            f"{score.item():.4f} "
            f"{ade_names[idx.item()]}"
        )


if __name__ == "__main__":
    main()