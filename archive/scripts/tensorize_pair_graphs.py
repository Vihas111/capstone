import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_progress(path, graph_idx, shard_idx, written):
    progress = {
        "last_processed_index": graph_idx,
        "last_shard": shard_idx,
        "graphs_written": written,
    }

    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def load_progress(path):
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def build_heterodata(graph, ade_vocab):

    data = HeteroData()

    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------
    nodes_by_type = {}

    for node in graph["nodes"]:
        nodes_by_type.setdefault(
            node["type"], []
        ).append(node["id"])

    node_maps = {}

    for node_type, node_ids in nodes_by_type.items():

        node_maps[node_type] = {
            node_id: idx
            for idx, node_id in enumerate(node_ids)
        }

        # Placeholder node features
        data[node_type].x = torch.arange(
            len(node_ids),
            dtype=torch.long,
        ).view(-1, 1)

    # --------------------------------------------------
    # Edges
    # --------------------------------------------------
    edge_store = {}

    for edge in graph["edges"]:

        src = edge["source"]
        dst = edge["target"]
        rel = edge["relation"]

        src_type = None
        dst_type = None

        for node_type, mapping in node_maps.items():

            if src in mapping:
                src_type = node_type

            if dst in mapping:
                dst_type = node_type

        if src_type is None or dst_type is None:
            continue

        key = (
            src_type,
            rel,
            dst_type,
        )

        edge_store.setdefault(
            key, []
        ).append(
            [
                node_maps[src_type][src],
                node_maps[dst_type][dst],
            ]
        )

    for key, edges in edge_store.items():

        edge_index = torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()

        data[key].edge_index = edge_index

    # --------------------------------------------------
    # Multi-hot ADE labels
    # --------------------------------------------------
    y = torch.zeros(
        len(ade_vocab),
        dtype=torch.float32,
    )

    for x in graph.get("ade_evidence", []):

        ade = x["ade"]

        if ade in ade_vocab:
            y[ade_vocab[ade]] = 1.0

    data.y = y.unsqueeze(0)

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    data.drug1 = graph["drug1"]
    data.drug2 = graph["drug2"]

    return data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="data/pair_graphs",
    )

    parser.add_argument(
        "--output-dir",
        default="data/tensorized",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--min-nodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-edges",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading vocabularies...")

    relation_vocab = load_json(
        "data/processed/relation_vocab.json"
    )

    node_type_vocab = load_json(
        "data/processed/node_type_vocab.json"
    )

    ade_vocab = load_json(
        output_dir / "ade_vocab.json"
    )

    print(f"Relations: {len(relation_vocab)}")
    print(f"Node types: {len(node_type_vocab)}")
    print(f"ADEs: {len(ade_vocab)}")

    graph_files = sorted(
        input_dir.glob("*.json")
    )

    print(f"Found {len(graph_files):,} graphs")

    progress_file = output_dir / "progress.json"

    start_idx = 0
    shard_idx = 0
    total_written = 0

    if args.resume:

        progress = load_progress(
            progress_file
        )

        if progress:

            start_idx = (
                progress["last_processed_index"] + 1
            )

            shard_idx = (
                progress["last_shard"] + 1
            )

            total_written = (
                progress["graphs_written"]
            )

            print(
                f"Resuming from graph "
                f"{start_idx:,}"
            )

    skipped = 0
    shard = []

    iterator = tqdm(
        enumerate(
            graph_files[start_idx:],
            start=start_idx,
        ),
        total=len(graph_files) - start_idx,
    )

    for graph_idx, graph_path in iterator:

        try:

            graph = load_json(
                graph_path
            )

            num_nodes = len(
                graph["nodes"]
            )

            num_edges = len(
                graph["edges"]
            )

            if num_nodes < args.min_nodes:
                skipped += 1
                continue

            if num_edges < args.min_edges:
                skipped += 1
                continue

            data = build_heterodata(
                graph,
                ade_vocab,
            )

            shard.append(data)

            if len(shard) >= args.shard_size:

                out_file = (
                    output_dir
                    / f"shard_{shard_idx:03d}.pt"
                )

                torch.save(
                    shard,
                    out_file,
                )

                total_written += len(shard)

                print(
                    f"\nSaved {out_file.name} "
                    f"({len(shard)} graphs)"
                )

                save_progress(
                    progress_file,
                    graph_idx,
                    shard_idx,
                    total_written,
                )

                shard_idx += 1
                shard = []

        except Exception as e:

            print(
                f"\nERROR: {graph_path.name}"
            )
            print(e)

            skipped += 1

    if shard:

        out_file = (
            output_dir
            / f"shard_{shard_idx:03d}.pt"
        )

        torch.save(
            shard,
            out_file,
        )

        total_written += len(shard)

        save_progress(
            progress_file,
            len(graph_files) - 1,
            shard_idx,
            total_written,
        )

        print(
            f"\nSaved final shard "
            f"{out_file.name}"
        )

    print("\nDone.")
    print(f"Written: {total_written:,}")
    print(f"Skipped: {skipped:,}")


if __name__ == "__main__":
    main()