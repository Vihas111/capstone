# =========================================================
# graph_stats.py
# =========================================================

import json

from collections import Counter

# =========================================================
# COUNTERS
# =========================================================

node_types = Counter()
edge_types = Counter()

total_nodes = 0
total_edges = 0

# =========================================================
# READ NODES
# =========================================================

print("Reading nodes...")

with open(
    "data/processed/nodes.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        node = json.loads(line)

        total_nodes += 1

        node_types[node["type"]] += 1

# =========================================================
# READ EDGES
# =========================================================

print("Reading edges...")

with open(
    "data/processed/edges.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        edge = json.loads(line)

        total_edges += 1

        edge_types[edge["relation"]] += 1

# =========================================================
# RESULTS
# =========================================================

print("\n===== GRAPH STATS =====\n")

print(f"Total Nodes: {total_nodes:,}")
print(f"Total Edges: {total_edges:,}")

# =========================================================
# NODE TYPES
# =========================================================

print("\n===== NODE TYPES =====\n")

for node_type, count in sorted(
    node_types.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{node_type:<20} {count:,}")

# =========================================================
# EDGE TYPES
# =========================================================

print("\n===== EDGE TYPES =====\n")

for edge_type, count in sorted(
    edge_types.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{edge_type:<20} {count:,}")

# =========================================================
# SIMPLE DENSITY STATS
# =========================================================

print("\n===== DENSITY STATS =====\n")

avg_degree = (
    (2 * total_edges) / total_nodes
    if total_nodes > 0 else 0
)

print(f"Average Degree: {avg_degree:.2f}")

# =========================================================
# TOP RELATIONS
# =========================================================

print("\n===== TOP 5 RELATIONS =====\n")

for relation, count in edge_types.most_common(5):

    percentage = (
        count / total_edges * 100
    )

    print(
        f"{relation:<20}"
        f"{count:,} "
        f"({percentage:.2f}%)"
    )

print("\nDone.")