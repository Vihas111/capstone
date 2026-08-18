import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.rgcn_baseline import RGCNBaseline
from torch_geometric.loader import DataLoader


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_ade_names():

    with open("data/tensorized/ade_vocab.json") as f:
        vocab = json.load(f)

    return {v: k for k, v in vocab.items()}


def main():

    shard_id = int(sys.argv[1])
    graph_id = int(sys.argv[2])

    print(f"Loading shard {shard_id:03d}...")

    graphs = torch.load(
        f"data/tensorized/shard_{shard_id:03d}.pt",
        weights_only=False,
    )

    graph = graphs[graph_id]

    loader = DataLoader(
        [graph],
        batch_size=1,
    )

    graph = next(iter(loader))

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

    graph = graph.to(DEVICE)

    with torch.no_grad():

        probs = torch.sigmoid(
            model(graph)
        )[0]

    ade_names = load_ade_names()

    print("\nTRUE ADEs")
    print("=" * 80)

    true_idx = torch.where(
        graph.y[0] > 0
    )[0]

    for idx in true_idx:

        print(
            ade_names[idx.item()]
        )

    print("\nTOP 20 PREDICTIONS")
    print("=" * 80)

    topk = torch.topk(
        probs,
        k=20,
    )

    for score, idx in zip(
        topk.values.cpu(),
        topk.indices.cpu(),
    ):

        print(
            f"{score.item():.4f}  "
            f"{ade_names[idx.item()]}"
        )


if __name__ == "__main__":
    main()