"""
v2: fixes two real bugs found running the v1 version of this script against
actual project data:

1. Casing/whitespace mismatch: build_hetero_graph.py stores target/enzyme/
   transporter node ids as `row["target"].lower().strip()` (e.g.
   "tumor necrosis factor"), but generate_pair_graphs.py (which builds
   data/pair_graphs/*.json) uses the SAME source file's value RAW, with
   original casing (e.g. "Tumor necrosis factor"). Same underlying data,
   inconsistent normalization between the two scripts. v1 of this script
   built a vocab straight from nodes.jsonl's (already-lowercased) ids, so
   any lookup using the pair-graph's raw casing failed. Fixed by normalizing
   every non-drug node id the same way (`.strip().lower()`) both when
   building the vocab and when tensorize_pair_graphs_v2.py looks ids up.

2. Missing node types: build_hetero_graph.py never loads carrier/pathway/
   category nodes at all (it's scoped to the ADE-GNN track), so nodes.jsonl
   has zero entries for those types even though generate_pair_graphs.py's
   pair graphs use them. nodes.jsonl alone isn't a complete source of truth.
   Fixed by additionally scanning data/pair_graphs/*.json once and adding
   any (type, normalized_id) not already covered -- this also acts as a
   safety net for target/enzyme/transporter in case nodes.jsonl is stale
   relative to the current drug_targets.jsonl/drug_enzymes.jsonl/etc.

"drug" ids are left untouched (no lower/strip) since build_hetero_graph.py
uses raw `str(row["drug"])` for those, and the very first run of this script
confirmed drug=19,830 matches CLAUDE.md's expected count exactly.

Usage:
    python scripts/build_node_vocab.py \
        --nodes-file data/processed/nodes.jsonl \
        --pair-graphs-dir data/pair_graphs \
        --out data/processed/node_vocab.json
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def normalize(node_id, node_type):
    if node_type == "drug":
        return str(node_id)
    return str(node_id).strip().lower()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-file", default="data/processed/nodes.jsonl")
    parser.add_argument("--pair-graphs-dir", default="data/pair_graphs")
    parser.add_argument("--out", default="data/processed/node_vocab.json")
    parser.add_argument(
        "--skip-pair-graph-scan",
        action="store_true",
        help="Skip the backfill pass over pair_graphs (faster, but will "
        "miss any node type/id not already present in nodes.jsonl -- only "
        "safe if you've already confirmed nodes.jsonl is complete).",
    )
    args = parser.parse_args()

    node_ids_by_type = {}  # type -> set of normalized ids

    print(f"Reading {args.nodes_file} ...")

    with open(args.nodes_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            node = json.loads(line)
            norm_id = normalize(node["id"], node["type"])
            node_ids_by_type.setdefault(node["type"], set()).add(norm_id)

    print("From nodes.jsonl:")
    for node_type, ids in sorted(node_ids_by_type.items()):
        print(f"  {node_type:15s} {len(ids)}")

    if not args.skip_pair_graph_scan:

        pair_graph_files = sorted(Path(args.pair_graphs_dir).glob("*.json"))
        print(
            f"\nBackfilling from {len(pair_graph_files):,} pair-graph files "
            f"(catches types/ids missing from nodes.jsonl)..."
        )

        added = {}

        for path in tqdm(pair_graph_files):
            with open(path, encoding="utf-8") as f:
                graph = json.load(f)

            for node in graph["nodes"]:
                norm_id = normalize(node["id"], node["type"])
                bucket = node_ids_by_type.setdefault(node["type"], set())
                if norm_id not in bucket:
                    bucket.add(norm_id)
                    added[node["type"]] = added.get(node["type"], 0) + 1

        if added:
            print("\nAdded from pair_graphs (missing from nodes.jsonl):")
            for node_type, count in sorted(added.items()):
                print(f"  {node_type:15s} +{count}")
        else:
            print("\nNothing new found in pair_graphs -- nodes.jsonl was complete.")

    node_vocab = {
        node_type: {
            node_id: idx for idx, node_id in enumerate(sorted(ids))
        }
        for node_type, ids in node_ids_by_type.items()
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(node_vocab, f)

    print("\nFinal global vocabulary sizes:")
    for node_type, mapping in sorted(node_vocab.items()):
        print(f"  {node_type:15s} {len(mapping)}")

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
