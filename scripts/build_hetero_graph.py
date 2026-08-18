# =========================================================
# build_hetero_graph.py
# =========================================================

import csv
import gzip
import json

# =========================================================
# LOAD GROUPED ADEs
# =========================================================

with open(
    "data/processed/grouped_ades.json",
    encoding="utf-8"
) as f:

    grouped_ades = json.load(f)

# =========================================================
# LOAD RXNORM -> DRUGBANK
# =========================================================

with open(
    "data/processed/rxnorm_to_drugbank.json",
    encoding="utf-8"
) as f:

    rxnorm_to_drugbank = json.load(f)

# =========================================================
# STORAGE
# =========================================================

nodes = {}
edges = set()

# =========================================================
# HELPERS
# =========================================================

def add_node(node_id, node_type):

    key = f"{node_type}:{node_id}"

    if key not in nodes:

        nodes[key] = {
            "id": node_id,
            "type": node_type
        }

def add_edge(source, target, relation):

    edges.add((
        str(source),
        str(target),
        str(relation)
    ))

# =========================================================
# DRUG NODES
# =========================================================

print("Loading drugs...")

with open(
    "data/processed/drugs.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        row = json.loads(line)

        drug_id = row["drugbank_id"]

        add_node(drug_id, "drug")

# =========================================================
# ENZYMES
# =========================================================

print("Loading enzymes...")

with open(
    "data/processed/drug_enzymes.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        row = json.loads(line)

        drug = str(row["drug"])

        enzyme = (
            row["enzyme"]
            .lower()
            .strip()
        )

        add_node(enzyme, "enzyme")

        actions = row.get("actions", [])

        for action in actions:

            action = action.lower()

            if action == "inhibitor":
                relation = "inhibits"

            elif action == "inducer":
                relation = "induces"

            elif action == "substrate":
                relation = "substrate_of"

            else:
                relation = action

            add_edge(
                drug,
                enzyme,
                relation
            )

# =========================================================
# TRANSPORTERS
# =========================================================

print("Loading transporters...")

with open(
    "data/processed/drug_transporters.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        row = json.loads(line)

        drug = str(row["drug"])

        transporter = (
            row["transporter"]
            .lower()
            .strip()
        )

        add_node(transporter, "transporter")

        actions = row.get("actions", [])

        for action in actions:

            action = action.lower()

            if action == "inhibitor":
                relation = "inhibits"

            elif action == "substrate":
                relation = "substrate_of"

            else:
                relation = action

            add_edge(
                drug,
                transporter,
                relation
            )

# =========================================================
# TARGETS
# =========================================================

print("Loading targets...")

with open(
    "data/processed/drug_targets.jsonl",
    encoding="utf-8"
) as f:

    for line in f:

        row = json.loads(line)

        drug = str(row["drug"])

        target = (
            row["target"]
            .lower()
            .strip()
        )

        add_node(target, "target")

        add_edge(
            drug,
            target,
            "targets"
        )

# =========================================================
# TWOSIDES INTERACTIONS
# =========================================================

print("Processing TWOSIDES...")

matched = 0
missing_ade = 0
mapping_failed = 0

with gzip.open(
    "../dataset_explore/twosides/raw/TWOSIDES.csv.gz",
    "rt",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for i, row in enumerate(reader):

        try:

            # IMPORTANT:
            # TWOSIDES typo is REAL
            rx1 = row["drug_1_rxnorn_id"]
            rx2 = row["drug_2_rxnorm_id"]

            ade = (
                row["condition_concept_name"]
                .lower()
                .strip()
            )

        except Exception as e:

            print("\nTWOSIDES parsing error:")
            print(e)
            break

        # =================================================
        # ADE FILTERING
        # =================================================

        if ade not in grouped_ades:

            missing_ade += 1
            continue

        ade_group = grouped_ades[ade]

        # =================================================
        # RXNORM -> DRUGBANK
        # =================================================

        db1_entry = rxnorm_to_drugbank.get(rx1)
        db2_entry = rxnorm_to_drugbank.get(rx2)

        if not db1_entry or not db2_entry:

            mapping_failed += 1
            continue

        # support both:
        # direct string
        # nested dict

        if isinstance(db1_entry, dict):
            db1 = db1_entry["drugbank_id"]
        else:
            db1 = db1_entry

        if isinstance(db2_entry, dict):
            db2 = db2_entry["drugbank_id"]
        else:
            db2 = db2_entry

        # =================================================
        # CANONICAL ORDERING
        # =================================================

        pair = sorted([db1, db2])

        db1 = pair[0]
        db2 = pair[1]

        # =================================================
        # INTERACTION NODE
        # =================================================

        interaction_id = (
            f"interaction_{db1}_{db2}"
        )

        add_node(
            interaction_id,
            "interaction"
        )

        add_node(
            ade_group,
            "ade"
        )

        # =================================================
        # DRUG -> INTERACTION
        # =================================================

        add_edge(
            db1,
            interaction_id,
            "participates_in"
        )

        add_edge(
            db2,
            interaction_id,
            "participates_in"
        )

        # =================================================
        # INTERACTION -> ADE
        # =================================================

        add_edge(
            interaction_id,
            ade_group,
            "associated_with"
        )

        matched += 1

        if i % 1_000_000 == 0 and i > 0:
            print(f"Processed {i:,} rows")

# =========================================================
# DEBUG STATS
# =========================================================

print("\n===== TWOSIDES DEBUG =====\n")

print(f"Matched rows: {matched:,}")
print(f"Missing ADE mappings: {missing_ade:,}")
print(f"RxNorm mapping failures: {mapping_failed:,}")

# =========================================================
# SAVE NODES
# =========================================================

print("Saving nodes...")

with open(
    "data/processed/nodes.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for node in nodes.values():

        f.write(
            json.dumps(node) + "\n"
        )

# =========================================================
# SAVE EDGES
# =========================================================

print("Saving edges...")

with open(
    "data/processed/edges.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for source, target, relation in edges:

        edge = {
            "source": source,
            "target": target,
            "relation": relation
        }

        f.write(
            json.dumps(edge) + "\n"
        )

# =========================================================
# FINAL STATS
# =========================================================

print("\n===== GRAPH STATS =====\n")

print(f"Nodes: {len(nodes):,}")
print(f"Edges: {len(edges):,}")

print("\nDone.")