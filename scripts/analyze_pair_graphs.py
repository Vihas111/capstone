import os
import json
from collections import Counter

PAIR_DIR = "data/pair_graphs"

files = [
    f for f in os.listdir(PAIR_DIR)
    if f.endswith(".json")
]

print(f"\nTotal Graphs: {len(files):,}")

node_counts = []
edge_counts = []

node_types = Counter()
edge_types = Counter()

for i, file in enumerate(files):

    path = os.path.join(PAIR_DIR, file)

    with open(path, encoding="utf-8") as f:
        graph = json.load(f)

    node_counts.append(
        len(graph["nodes"])
    )

    edge_counts.append(
        len(graph["edges"])
    )

    for node in graph["nodes"]:
        node_types[node["type"]] += 1

    for edge in graph["edges"]:
        edge_types[edge["relation"]] += 1

    if i % 10000 == 0 and i > 0:
        print(f"Processed {i:,}")

print("\n===== GRAPH SIZE =====\n")

print(
    "Average Nodes:",
    round(sum(node_counts) / len(node_counts), 2)
)

print(
    "Average Edges:",
    round(sum(edge_counts) / len(edge_counts), 2)
)

print(
    "Max Nodes:",
    max(node_counts)
)

print(
    "Max Edges:",
    max(edge_counts)
)

print("\n===== NODE TYPES =====\n")

for k, v in node_types.most_common():
    print(f"{k:<20}{v:,}")

print("\n===== EDGE TYPES =====\n")

for k, v in edge_types.most_common():
    print(f"{k:<25}{v:,}")