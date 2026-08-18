# =========================================================
# prototype_pair_graph.py
# =========================================================
#
# PURPOSE:
#   Build ONE mechanistic heterogeneous graph
#   for a drug pair.
#
# UPDATED DESIGN:
#   - NO ADE nodes
#   - ADEs stored as graph-level labels/evidence
#   - mechanism-centric graph
#   - cleaner SumGNN-compatible structure
#
# OUTPUT:
#   data/pair_graphs/
#
#       DBxxxx__DByyyy.json
#
# =========================================================

import json
import os

# =========================================================
# CONFIG
# =========================================================

DRUG_1 = "DB00186"
DRUG_2 = "DB00945"

TOP_K_ADE = 25

# =========================================================
# PATHS
# =========================================================

DATA_DIR = "data/processed/"

PAIR_GRAPH_DIR = "data/pair_graphs/"

os.makedirs(
    PAIR_GRAPH_DIR,
    exist_ok=True
)

# =========================================================
# FILES
# =========================================================

TARGETS_FILE = (
    DATA_DIR +
    "drug_targets.jsonl"
)

ENZYMES_FILE = (
    DATA_DIR +
    "drug_enzymes.jsonl"
)

TRANSPORTERS_FILE = (
    DATA_DIR +
    "drug_transporters.jsonl"
)

CARRIERS_FILE = (
    DATA_DIR +
    "drug_carriers.jsonl"
)

PATHWAYS_FILE = (
    DATA_DIR +
    "drug_pathways.jsonl"
)

CATEGORIES_FILE = (
    DATA_DIR +
    "drug_categories.jsonl"
)

TWOSIDES_FILE = (
    DATA_DIR +
    "twosides_interactions.jsonl"
)

RXNORM_MAPPING_FILE = (
    DATA_DIR +
    "drugbank_to_rxnorm.json"
)

# =========================================================
# HELPERS
# =========================================================

def read_jsonl(path):

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            yield json.loads(line)

# =========================================================
# LOAD RXNORM MAP
# =========================================================

print("Loading mappings...")

with open(
    RXNORM_MAPPING_FILE,
    encoding="utf-8"
) as f:

    drugbank_to_rxnorm = json.load(f)

rx1 = drugbank_to_rxnorm.get(DRUG_1)
rx2 = drugbank_to_rxnorm.get(DRUG_2)

print("Drug 1 RxNorm:", rx1)
print("Drug 2 RxNorm:", rx2)

# =========================================================
# GRAPH STRUCTURE
# =========================================================

graph = {

    "drug1": DRUG_1,
    "drug2": DRUG_2,

    "nodes": [],
    "edges": [],

    # -----------------------------------------------------
    # ADEs become graph-level evidence
    # -----------------------------------------------------

    "ade_evidence": []

}

# =========================================================
# DEDUPLICATION
# =========================================================

node_set = set()

edge_set = set()

# =========================================================
# NODE ADDER
# =========================================================

def add_node(node_id, node_type):

    key = (node_id, node_type)

    if key in node_set:
        return

    node_set.add(key)

    graph["nodes"].append({

        "id": node_id,
        "type": node_type

    })

# =========================================================
# EDGE ADDER
# =========================================================

def add_edge(source, target, relation):

    key = (
        source,
        target,
        relation
    )

    if key in edge_set:
        return

    edge_set.add(key)

    graph["edges"].append({

        "source": source,
        "target": target,
        "relation": relation

    })

# =========================================================
# CORE DRUG NODES
# =========================================================

add_node(DRUG_1, "drug")
add_node(DRUG_2, "drug")

# =========================================================
# ENZYMES
# =========================================================

print("\nProcessing enzymes...")

for row in read_jsonl(ENZYMES_FILE):

    drug = row["drug"]

    if (
        drug != DRUG_1
        and
        drug != DRUG_2
    ):

        continue

    enzyme = row["enzyme"]

    actions = row.get(
        "actions",
        []
    )

    add_node(
        enzyme,
        "enzyme"
    )

    if not actions:

        add_edge(
            drug,
            enzyme,
            "associated_with"
        )

    for action in actions:

        relation = (

            action
            .lower()
            .replace(" ", "_")

        )

        add_edge(
            drug,
            enzyme,
            relation
        )

# =========================================================
# TARGETS
# =========================================================

print("Processing targets...")

for row in read_jsonl(TARGETS_FILE):

    drug = row["drug"]

    if (
        drug != DRUG_1
        and
        drug != DRUG_2
    ):

        continue

    target = row["target"]

    actions = row.get(
        "actions",
        []
    )

    add_node(
        target,
        "target"
    )

    if not actions:

        add_edge(
            drug,
            target,
            "targets"
        )

    for action in actions:

        relation = (

            action
            .lower()
            .replace(" ", "_")

        )

        add_edge(
            drug,
            target,
            relation
        )

# =========================================================
# TRANSPORTERS
# =========================================================

print("Processing transporters...")

for row in read_jsonl(TRANSPORTERS_FILE):

    drug = row["drug"]

    if (
        drug != DRUG_1
        and
        drug != DRUG_2
    ):

        continue

    transporter = row["transporter"]

    actions = row.get(
        "actions",
        []
    )

    add_node(
        transporter,
        "transporter"
    )

    if not actions:

        add_edge(
            drug,
            transporter,
            "transported_by"
        )

    for action in actions:

        relation = (

            action
            .lower()
            .replace(" ", "_")

        )

        add_edge(
            drug,
            transporter,
            relation
        )

# =========================================================
# CARRIERS
# =========================================================

print("Processing carriers...")

for row in read_jsonl(CARRIERS_FILE):

    drug = row["drug"]

    if (
        drug != DRUG_1
        and
        drug != DRUG_2
    ):

        continue

    carrier = row["carrier"]

    add_node(
        carrier,
        "carrier"
    )

    add_edge(
        drug,
        carrier,
        "carrier_association"
    )

# =========================================================
# PATHWAYS
# =========================================================

print("Processing pathways...")

for row in read_jsonl(PATHWAYS_FILE):

    drug = row["drug"]

    if (
        drug != DRUG_1
        and
        drug != DRUG_2
    ):

        continue

    pathway = row.get(
        "pathway"
    )

    if not pathway:
        continue

    add_node(
        pathway,
        "pathway"
    )

    add_edge(
        drug,
        pathway,
        "participates_in"
    )

# =========================================================
# CATEGORIES
# =========================================================

print("Processing categories...")

for row in read_jsonl(CATEGORIES_FILE):

    drug = row["drug"]

    if (
        drug != DRUG_1
        and
        drug != DRUG_2
    ):

        continue

    category = row["category"]

    add_node(
        category,
        "category"
    )

    add_edge(
        drug,
        category,
        "belongs_to"
    )

# =========================================================
# TWOSIDES ADE EVIDENCE
# =========================================================

print("Processing TWOSIDES evidence...")

ade_evidence = []

if rx1 and rx2:

    target_pair = set([rx1, rx2])

    for i, row in enumerate(
        read_jsonl(TWOSIDES_FILE)
    ):

        pair = set([

            row["rxnorm_1"],
            row["rxnorm_2"]

        ])

        if pair != target_pair:
            continue

        ade_evidence.append({

            "ade":
                row["ade"],

            "prr":
                row["prr"],

            "prr_error":
                row["prr_error"],

            "mean_reporting_frequency":
                row[
                    "mean_reporting_frequency"
                ]

        })

        if i % 5_000_000 == 0 and i > 0:

            print(
                f"Scanned {i:,} TWOSIDES rows"
            )

# =========================================================
# KEEP ONLY TOP-K ADEs
# =========================================================

ade_evidence = sorted(

    ade_evidence,

    key=lambda x:
        x.get("prr", 0),

    reverse=True

)

graph["ade_evidence"] = (
    ade_evidence[:TOP_K_ADE]
)

# =========================================================
# SAVE GRAPH
# =========================================================

output_path = (

    PAIR_GRAPH_DIR +
    f"{DRUG_1}__{DRUG_2}.json"

)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        graph,
        f,
        indent=2
    )

# =========================================================
# SUMMARY
# =========================================================

print("\n===== GRAPH SUMMARY =====\n")

print("Drug 1:", DRUG_1)
print("Drug 2:", DRUG_2)

print()

print(
    "Total Nodes:",
    len(graph["nodes"])
)

print(
    "Total Edges:",
    len(graph["edges"])
)

print(
    "ADE Evidence Entries:",
    len(graph["ade_evidence"])
)

# =========================================================
# NODE TYPES
# =========================================================

node_type_counts = {}

for node in graph["nodes"]:

    node_type = node["type"]

    node_type_counts[
        node_type
    ] = (

        node_type_counts.get(
            node_type,
            0
        ) + 1

    )

print("\n===== NODE TYPES =====\n")

for node_type, count in sorted(
    node_type_counts.items()
):

    print(
        f"{node_type:<20}"
        f"{count:,}"
    )

# =========================================================
# EDGE TYPES
# =========================================================

edge_type_counts = {}

for edge in graph["edges"]:

    relation = edge["relation"]

    edge_type_counts[
        relation
    ] = (

        edge_type_counts.get(
            relation,
            0
        ) + 1

    )

print("\n===== EDGE TYPES =====\n")

for relation, count in sorted(
    edge_type_counts.items()
):

    print(
        f"{relation:<25}"
        f"{count:,}"
    )

# =========================================================
# TOP ADE SIGNALS
# =========================================================

print("\n===== TOP ADE SIGNALS =====\n")

for ade in graph["ade_evidence"]:

    print(

        f"{ade['ade']}"
        f" | PRR={ade['prr']}"

    )

print("\nSaved graph:")
print(output_path)

print("\nDone.")