# =========================================================
# inspect_pair_graph.py
# =========================================================
#
# PURPOSE:
#   Inspect heterogeneous mechanistic pair graphs
#
# GOALS:
#   - analyze node distributions
#   - analyze edge distributions
#   - detect PK motifs
#   - detect PD motifs
#   - analyze ADE dominance
#   - estimate graph quality
#
# =========================================================

import json

from collections import Counter
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================

PAIR_GRAPH_PATH = (
    "data/pair_graphs/DB00186__DB00945.json"
)

TOP_K = 20

# =========================================================
# LOAD GRAPH
# =========================================================

print("Loading graph...")

with open(
    PAIR_GRAPH_PATH,
    encoding="utf-8"
) as f:

    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]

drug1 = graph["drug1"]
drug2 = graph["drug2"]

# =========================================================
# BASIC INFO
# =========================================================

print("\n===== PAIR GRAPH REPORT =====\n")

print("Drug 1:", drug1)
print("Drug 2:", drug2)

print()

print("Total Nodes:", len(nodes))
print("Total Edges:", len(edges))

# =========================================================
# NODE TYPE ANALYSIS
# =========================================================

print("\n===== NODE TYPE DISTRIBUTION =====\n")

node_type_counter = Counter()

for node in nodes:

    node_type_counter[
        node["type"]
    ] += 1

for node_type, count in sorted(

    node_type_counter.items(),

    key=lambda x: x[1],

    reverse=True

):

    percentage = (
        count / len(nodes)
    ) * 100

    print(
        f"{node_type:<20}"
        f"{count:<10}"
        f"({percentage:.2f}%)"
    )

# =========================================================
# EDGE TYPE ANALYSIS
# =========================================================

print("\n===== EDGE TYPE DISTRIBUTION =====\n")

edge_type_counter = Counter()

for edge in edges:

    edge_type_counter[
        edge["relation"]
    ] += 1

for relation, count in sorted(

    edge_type_counter.items(),

    key=lambda x: x[1],

    reverse=True

):

    percentage = (
        count / len(edges)
    ) * 100

    print(
        f"{relation:<25}"
        f"{count:<10}"
        f"({percentage:.2f}%)"
    )

# =========================================================
# BUILD EDGE LOOKUPS
# =========================================================

outgoing = defaultdict(list)

incoming = defaultdict(list)

for edge in edges:

    source = edge["source"]

    target = edge["target"]

    relation = edge["relation"]

    outgoing[source].append(edge)

    incoming[target].append(edge)

# =========================================================
# PK MOTIF DETECTION
# =========================================================

print("\n===== PK MOTIF DETECTION =====\n")

pk_motifs = []

# ---------------------------------------------------------
# Detect inhibitor-substrate motifs
# ---------------------------------------------------------

for node in nodes:

    if node["type"] != "enzyme":
        continue

    enzyme = node["id"]

    incoming_edges = incoming[enzyme]

    inhibitors = []
    substrates = []
    inducers = []

    for edge in incoming_edges:

        relation = edge["relation"]

        source = edge["source"]

        if relation == "inhibitor":

            inhibitors.append(source)

        elif relation == "substrate":

            substrates.append(source)

        elif relation == "inducer":

            inducers.append(source)

    # -----------------------------------------------------
    # inhibitor-substrate
    # -----------------------------------------------------

    for inhibitor in inhibitors:

        for substrate in substrates:

            if inhibitor == substrate:
                continue

            pk_motifs.append({

                "type":
                    "inhibitor_substrate",

                "enzyme":
                    enzyme,

                "description":
                    f"{inhibitor} inhibits "
                    f"{enzyme} used by "
                    f"{substrate}"

            })

    # -----------------------------------------------------
    # inducer-substrate
    # -----------------------------------------------------

    for inducer in inducers:

        for substrate in substrates:

            if inducer == substrate:
                continue

            pk_motifs.append({

                "type":
                    "inducer_substrate",

                "enzyme":
                    enzyme,

                "description":
                    f"{inducer} induces "
                    f"{enzyme} affecting "
                    f"{substrate}"

            })

    # -----------------------------------------------------
    # substrate competition
    # -----------------------------------------------------

    if len(substrates) >= 2:

        pk_motifs.append({

            "type":
                "substrate_competition",

            "enzyme":
                enzyme,

            "description":
                f"Multiple drugs use "
                f"{enzyme}"

        })

# ---------------------------------------------------------
# Print PK motifs
# ---------------------------------------------------------

print(
    "Detected PK motifs:",
    len(pk_motifs)
)

print()

if pk_motifs:

    for motif in pk_motifs[:TOP_K]:

        print(
            f"[{motif['type']}] "
            f"{motif['description']}"
        )

else:

    print(
        "No PK motifs detected."
    )

# =========================================================
# PD ANALYSIS
# =========================================================

print("\n===== PD ANALYSIS =====\n")

target_nodes = [

    n for n in nodes

    if n["type"] == "target"

]

pathway_nodes = [

    n for n in nodes

    if n["type"] == "pathway"

]

category_nodes = [

    n for n in nodes

    if n["type"] == "category"

]

print(
    "Targets:",
    len(target_nodes)
)

print(
    "Pathways:",
    len(pathway_nodes)
)

print(
    "Categories:",
    len(category_nodes)
)

# ---------------------------------------------------------
# Shared target detection
# ---------------------------------------------------------

drug_targets = defaultdict(set)

for edge in edges:

    if edge["relation"] in [

        "targets",
        "agonist",
        "antagonist",
        "binder",
        "ligand",
        "modulator",
        "activator",
        "downregulator",
        "upregulator",
        "positive_allosteric_modulator"

    ]:

        source = edge["source"]
        target = edge["target"]

        drug_targets[source].add(target)

shared_targets = (

    drug_targets[drug1]
    &
    drug_targets[drug2]

)

print()

print(
    "Shared Targets:",
    len(shared_targets)
)

if shared_targets:

    print()

    for target in list(shared_targets)[:TOP_K]:

        print("-", target)

# =========================================================
# ADE ANALYSIS
# =========================================================

print("\n===== ADE ANALYSIS =====\n")

ade_nodes = [

    n for n in nodes

    if n["type"] == "ade"

]

print(
    "ADE Nodes:",
    len(ade_nodes)
)

ade_percentage = (
    len(ade_nodes)
    /
    len(nodes)
) * 100

print(
    "ADE Percentage:",
    f"{ade_percentage:.2f}%"
)

# ---------------------------------------------------------
# ADE dominance warning
# ---------------------------------------------------------

if ade_percentage > 80:

    print()

    print(
        "WARNING: ADE domination detected."
    )

    print(
        "Graph is heavily observational."
    )

    print(
        "Mechanistic structure may be drowned out."
    )

# =========================================================
# TOP PRR ADEs
# =========================================================

print("\n===== TOP ADE SIGNALS =====\n")

ade_edges = []

for edge in edges:

    if (
        edge["relation"]
        ==
        "associated_with"
    ):

        if "prr" in edge:

            ade_edges.append(edge)

ade_edges_sorted = sorted(

    ade_edges,

    key=lambda x:
        x.get("prr", 0),

    reverse=True

)

for edge in ade_edges_sorted[:TOP_K]:

    print(

        f"{edge['target']}"
        f" | PRR={edge['prr']}"

    )

# =========================================================
# GRAPH CONNECTIVITY
# =========================================================

print("\n===== GRAPH CONNECTIVITY =====\n")

isolated_nodes = 0

for node in nodes:

    node_id = node["id"]

    degree = (

        len(outgoing[node_id])
        +
        len(incoming[node_id])

    )

    if degree == 0:

        isolated_nodes += 1

print(
    "Isolated Nodes:",
    isolated_nodes
)

density = (

    len(edges)
    /
    len(nodes)

)

print(
    "Edge/Node Ratio:",
    f"{density:.4f}"
)

# =========================================================
# GRAPH QUALITY SCORING
# =========================================================

print("\n===== GRAPH QUALITY =====\n")

score = 0

# ---------------------------------------------------------
# PK richness
# ---------------------------------------------------------

if len(pk_motifs) > 0:

    score += 1

# ---------------------------------------------------------
# PD richness
# ---------------------------------------------------------

if len(shared_targets) > 0:

    score += 1

# ---------------------------------------------------------
# Mechanistic edges
# ---------------------------------------------------------

mechanistic_relations = [

    "inhibitor",
    "substrate",
    "inducer",
    "agonist",
    "antagonist",
    "modulator"

]

mechanistic_edge_count = 0

for relation in mechanistic_relations:

    mechanistic_edge_count += (

        edge_type_counter.get(
            relation,
            0
        )

    )

if mechanistic_edge_count > 10:

    score += 1

# ---------------------------------------------------------
# ADE domination penalty
# ---------------------------------------------------------

if ade_percentage < 60:

    score += 1

# =========================================================
# INTERPRETATION
# =========================================================

if score == 4:

    quality = "EXCELLENT"

elif score == 3:

    quality = "GOOD"

elif score == 2:

    quality = "MODERATE"

else:

    quality = "WEAK"

print(
    "Quality Score:",
    score
)

print(
    "Overall Quality:",
    quality
)

# =========================================================
# FINAL INTERPRETATION
# =========================================================

print("\n===== FINAL INTERPRETATION =====\n")

if pk_motifs:

    print(
        "Mechanistic PK pathways detected."
    )

if shared_targets:

    print(
        "Potential PD overlap detected."
    )

if mechanistic_edge_count > 10:

    print(
        "Graph contains strong typed "
        "mechanistic relations."
    )

if ade_percentage > 80:

    print(
        "ADE node domination is excessive."
    )

    print(
        "Consider converting ADEs into "
        "labels or top-K evidence nodes."
    )

print("\nDone.")