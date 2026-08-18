# =========================================================
# generate_pair_graphs.py
# =========================================================
#
# PURPOSE:
#   Generate scalable mechanistic pair graphs
#
# OUTPUT:
#   data/pair_graphs/
#
# =========================================================

import json
import os

from collections import defaultdict

# =========================================================
# PATHS
# =========================================================

DATA_DIR = "data/processed/"
INDEX_DIR = "data/indexed/"
OUTPUT_DIR = "data/pair_graphs/"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =========================================================
# FILES
# =========================================================

PAIR_INDEX_FILE = (
    INDEX_DIR +
    "pair_to_ades.json"
)

RXNORM_TO_DRUGBANK = (
    DATA_DIR +
    "rxnorm_to_drugbank.json"
)

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
# LOAD PAIR INDEX
# =========================================================

print("Loading pair index...")

with open(
    PAIR_INDEX_FILE,
    encoding="utf-8"
) as f:

    pair_index = json.load(f)

print(
    "Pairs loaded:",
    len(pair_index)
)

# =========================================================
# LOAD RXNORM MAP
# =========================================================

print("Loading RxNorm mappings...")

with open(
    RXNORM_TO_DRUGBANK,
    encoding="utf-8"
) as f:

    rxnorm_to_drugbank = json.load(f)

# =========================================================
# BUILD FEATURE LOOKUPS
# =========================================================

print("Building feature lookups...")

enzymes = defaultdict(list)
targets = defaultdict(list)
transporters = defaultdict(list)
carriers = defaultdict(list)
pathways = defaultdict(list)
categories = defaultdict(list)

# ---------------------------------------------------------
# ENZYMES
# ---------------------------------------------------------

for row in read_jsonl(ENZYMES_FILE):

    enzymes[
        row["drug"]
    ].append(row)

# ---------------------------------------------------------
# TARGETS
# ---------------------------------------------------------

for row in read_jsonl(TARGETS_FILE):

    targets[
        row["drug"]
    ].append(row)

# ---------------------------------------------------------
# TRANSPORTERS
# ---------------------------------------------------------

for row in read_jsonl(TRANSPORTERS_FILE):

    transporters[
        row["drug"]
    ].append(row)

# ---------------------------------------------------------
# CARRIERS
# ---------------------------------------------------------

for row in read_jsonl(CARRIERS_FILE):

    carriers[
        row["drug"]
    ].append(row)

# ---------------------------------------------------------
# PATHWAYS
# ---------------------------------------------------------

for row in read_jsonl(PATHWAYS_FILE):

    pathways[
        row["drug"]
    ].append(row)

# ---------------------------------------------------------
# CATEGORIES
# ---------------------------------------------------------

for row in read_jsonl(CATEGORIES_FILE):

    categories[
        row["drug"]
    ].append(row)

# =========================================================
# GRAPH HELPERS
# =========================================================

def add_node(
    graph,
    node_set,
    node_id,
    node_type
):

    key = (
        node_id,
        node_type
    )

    if key in node_set:
        return

    node_set.add(key)

    graph["nodes"].append({

        "id": node_id,
        "type": node_type

    })

# ---------------------------------------------------------

def add_edge(
    graph,
    edge_set,
    source,
    target,
    relation
):

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
# GENERATE GRAPHS
# =========================================================

print("\nGenerating pair graphs...\n")

generated = 0
skipped = 0

for i, (pair_key, ades) in enumerate(
    pair_index.items()
):

    # -----------------------------------------------------
    # RXNORM PAIR
    # -----------------------------------------------------

    rx1, rx2 = pair_key.split("__")

    drug1 = rxnorm_to_drugbank.get(rx1)
    drug2 = rxnorm_to_drugbank.get(rx2)

    # -----------------------------------------------------
    # skip unmapped pairs
    # -----------------------------------------------------

    if not drug1 or not drug2:

        skipped += 1
        continue

    # -----------------------------------------------------
    # GRAPH
    # -----------------------------------------------------

    graph = {

        "drug1": drug1,
        "drug2": drug2,

        "nodes": [],
        "edges": [],

        "ade_evidence": sorted(

            ades,

            key=lambda x:
                x.get("prr", 0),

            reverse=True

        )[:25]

    }

    node_set = set()
    edge_set = set()

    # -----------------------------------------------------
    # CORE DRUGS
    # -----------------------------------------------------

    add_node(
        graph,
        node_set,
        drug1,
        "drug"
    )

    add_node(
        graph,
        node_set,
        drug2,
        "drug"
    )

    # =====================================================
    # FEATURE INJECTION FUNCTION
    # =====================================================

    def inject_features(
        drug
    ):

        # -------------------------------------------------
        # ENZYMES
        # -------------------------------------------------

        for row in enzymes[drug]:

            enzyme = row["enzyme"]

            add_node(
                graph,
                node_set,
                enzyme,
                "enzyme"
            )

            actions = row.get(
                "actions",
                []
            )

            if not actions:

                add_edge(
                    graph,
                    edge_set,
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
                    graph,
                    edge_set,
                    drug,
                    enzyme,
                    relation
                )

        # -------------------------------------------------
        # TARGETS
        # -------------------------------------------------

        for row in targets[drug]:

            target = row["target"]

            add_node(
                graph,
                node_set,
                target,
                "target"
            )

            actions = row.get(
                "actions",
                []
            )

            if not actions:

                add_edge(
                    graph,
                    edge_set,
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
                    graph,
                    edge_set,
                    drug,
                    target,
                    relation
                )

        # -------------------------------------------------
        # TRANSPORTERS
        # -------------------------------------------------

        for row in transporters[drug]:

            transporter = row["transporter"]

            add_node(
                graph,
                node_set,
                transporter,
                "transporter"
            )

            actions = row.get(
                "actions",
                []
            )

            if not actions:

                add_edge(
                    graph,
                    edge_set,
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
                    graph,
                    edge_set,
                    drug,
                    transporter,
                    relation
                )

        # -------------------------------------------------
        # CARRIERS
        # -------------------------------------------------

        for row in carriers[drug]:

            carrier = row["carrier"]

            add_node(
                graph,
                node_set,
                carrier,
                "carrier"
            )

            add_edge(
                graph,
                edge_set,
                drug,
                carrier,
                "carrier_association"
            )

        # -------------------------------------------------
        # PATHWAYS
        # -------------------------------------------------

        for row in pathways[drug]:

            pathway = row.get(
                "pathway"
            )

            if not pathway:
                continue

            add_node(
                graph,
                node_set,
                pathway,
                "pathway"
            )

            add_edge(
                graph,
                edge_set,
                drug,
                pathway,
                "participates_in"
            )

        # -------------------------------------------------
        # CATEGORIES
        # -------------------------------------------------

        for row in categories[drug]:

            category = row["category"]

            add_node(
                graph,
                node_set,
                category,
                "category"
            )

            add_edge(
                graph,
                edge_set,
                drug,
                category,
                "belongs_to"
            )

    # -----------------------------------------------------
    # inject both drugs
    # -----------------------------------------------------

    inject_features(drug1)
    inject_features(drug2)

    # =====================================================
    # SAVE GRAPH
    # =====================================================

    output_file = (

        OUTPUT_DIR +
        f"{drug1}__{drug2}.json"

    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            graph,
            f
        )

    generated += 1

    # =====================================================
    # LOGGING
    # =====================================================

    if generated % 1000 == 0:

        print(
            f"Generated {generated:,} graphs"
        )

# =========================================================
# SUMMARY
# =========================================================

print("\n===== GENERATION COMPLETE =====\n")

print(
    "Generated graphs:",
    generated
)

print(
    "Skipped pairs:",
    skipped
)

print(
    "\nSaved to:"
)

print(OUTPUT_DIR)

print("\nDone.")