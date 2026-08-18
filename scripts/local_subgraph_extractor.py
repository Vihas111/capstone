# =========================================================
# local_subgraph_extractor.py
# =========================================================

import json
import networkx as nx

from collections import Counter, deque

# =========================================================
# LOAD GRAPH
# =========================================================

print("Loading graph...")

G = nx.Graph()

node_types = {}

# =========================================================
# LOAD NODES
# =========================================================

print("Loading nodes...")

with open(
    "data/processed/nodes.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        node = json.loads(line)

        node_id = node["id"]

        node_types[node_id] = node["type"]

        G.add_node(
            node_id,
            node_type=node["type"]
        )

# =========================================================
# LOAD EDGES
# =========================================================

print("Loading edges...")

with open(
    "data/processed/edges.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        edge = json.loads(line)

        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"]
        )

# =========================================================
# GRAPH STATS
# =========================================================

print("\n===== GRAPH LOADED =====\n")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# =========================================================
# BIOLOGY-ONLY RELATIONS
# =========================================================

ALLOWED_RELATIONS = {

    "inhibits",
    "induces",
    "substrate_of",
    "targets"

}

# =========================================================
# FILTERED BFS
# =========================================================

def filtered_bfs(
    start_node,
    max_hops=2
):

    visited = set()

    queue = deque()

    queue.append(
        (start_node, 0)
    )

    visited.add(start_node)

    while queue:

        current_node, depth = queue.popleft()

        if depth >= max_hops:
            continue

        for neighbor in G.neighbors(current_node):

            edge_data = G.get_edge_data(
                current_node,
                neighbor
            )

            relation = edge_data["relation"]

            # =============================================
            # RELATION FILTERING
            # =============================================

            if relation not in ALLOWED_RELATIONS:
                continue

            neighbor_type = node_types.get(
                neighbor,
                "unknown"
            )

            # =============================================
            # BLOCK NON-BIOLOGY
            # =============================================

            if neighbor_type in {

                "interaction",
                "ade"

            }:
                continue

            # =============================================
            # VISIT
            # =============================================

            if neighbor not in visited:

                visited.add(neighbor)

                queue.append(
                    (
                        neighbor,
                        depth + 1
                    )
                )

    return visited

# =========================================================
# EXTRACT SHARED-BIOLOGY SUBGRAPH
# =========================================================

def extract_subgraph(
    drug1,
    drug2,
    hops=2
):

    # =====================================================
    # BIOLOGICAL NEIGHBORHOODS
    # =====================================================

    nodes_1 = filtered_bfs(
        drug1,
        hops
    )

    nodes_2 = filtered_bfs(
        drug2,
        hops
    )

    # =====================================================
    # KEEP ONLY SHARED BIOLOGY
    # =====================================================

    shared_nodes = (
        nodes_1
        &
        nodes_2
    )

    # always keep source drugs

    shared_nodes.add(drug1)
    shared_nodes.add(drug2)

    # =====================================================
    # INDUCED SUBGRAPH
    # =====================================================

    subgraph = G.subgraph(
        shared_nodes
    ).copy()

    return subgraph

# =========================================================
# TEST EXTRACTION
# =========================================================

print("\n===== TEST EXTRACTION =====\n")

drug1 = "DB00186"
drug2 = "DB00945"

subgraph = extract_subgraph(
    drug1,
    drug2,
    hops=2
)

print("Subgraph nodes:", subgraph.number_of_nodes())
print("Subgraph edges:", subgraph.number_of_edges())

# =========================================================
# NODE TYPE ANALYSIS
# =========================================================

counter = Counter()

for node in subgraph.nodes():

    node_type = node_types.get(
        node,
        "unknown"
    )

    counter[node_type] += 1

print("\n===== NODE TYPES =====\n")

for node_type, count in counter.items():

    print(f"{node_type:<20} {count}")

# =========================================================
# SAMPLE NODES
# =========================================================

print("\n===== SAMPLE NODES =====\n")

sample = list(subgraph.nodes())[:50]

for node in sample:

    print(
        f"{node:<45}"
        f"{node_types.get(node)}"
    )

print("\nDone.")