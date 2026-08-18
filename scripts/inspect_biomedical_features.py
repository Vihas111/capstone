# =========================================================
# inspect_biomedical_features.py
# =========================================================
#
# PURPOSE:
#   Inspect extracted biomedical JSONL files
#   before pair-graph construction.
#
# GOALS:
#   - understand schema quality
#   - inspect action distributions
#   - inspect pathway richness
#   - inspect target/enzyme structure
#   - validate extraction quality
#
# =========================================================

import json

from collections import Counter, defaultdict

# =========================================================
# PATHS
# =========================================================

DATA_DIR = "data/processed/"

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

MECHANISMS_FILE = (
    DATA_DIR +
    "drug_mechanisms.jsonl"
)

DRUG_INTERACTIONS_FILE = (
    DATA_DIR +
    "drugbank_interactions.jsonl"
)

TWOSIDES_FILE = (
    DATA_DIR +
    "twosides_interactions.jsonl"
)

# =========================================================
# TARGET ANALYSIS
# =========================================================

print("\n===== TARGET ANALYSIS =====\n")

target_action_counter = Counter()

target_counter = Counter()

sample_targets = []

for row in read_jsonl(TARGETS_FILE):

    target = row["target"]

    target_counter[target] += 1

    actions = row.get(
        "actions",
        []
    )

    for action in actions:

        target_action_counter[action] += 1

    if len(sample_targets) < 10:

        sample_targets.append(row)

print("Unique Targets:", len(target_counter))

print("\nTop 20 Target Actions:\n")

for action, count in target_action_counter.most_common(20):

    print(f"{action:<25} {count:,}")

print("\nSample Target Rows:\n")

for row in sample_targets:

    print(json.dumps(row, indent=2))

# =========================================================
# ENZYME ANALYSIS
# =========================================================

print("\n===== ENZYME ANALYSIS =====\n")

enzyme_action_counter = Counter()

enzyme_counter = Counter()

sample_enzymes = []

for row in read_jsonl(ENZYMES_FILE):

    enzyme = row["enzyme"]

    enzyme_counter[enzyme] += 1

    actions = row.get(
        "actions",
        []
    )

    for action in actions:

        enzyme_action_counter[action] += 1

    if len(sample_enzymes) < 10:

        sample_enzymes.append(row)

print("Unique Enzymes:", len(enzyme_counter))

print("\nTop Enzyme Actions:\n")

for action, count in enzyme_action_counter.most_common(20):

    print(f"{action:<25} {count:,}")

print("\nSample Enzyme Rows:\n")

for row in sample_enzymes:

    print(json.dumps(row, indent=2))

# =========================================================
# TRANSPORTER ANALYSIS
# =========================================================

print("\n===== TRANSPORTER ANALYSIS =====\n")

transporter_action_counter = Counter()

transporter_counter = Counter()

for row in read_jsonl(TRANSPORTERS_FILE):

    transporter = row["transporter"]

    transporter_counter[transporter] += 1

    actions = row.get(
        "actions",
        []
    )

    for action in actions:

        transporter_action_counter[action] += 1

print("Unique Transporters:", len(transporter_counter))

print("\nTop Transporter Actions:\n")

for action, count in transporter_action_counter.most_common(20):

    print(f"{action:<25} {count:,}")

# =========================================================
# CARRIER ANALYSIS
# =========================================================

print("\n===== CARRIER ANALYSIS =====\n")

carrier_counter = Counter()

for row in read_jsonl(CARRIERS_FILE):

    carrier_counter[
        row["carrier"]
    ] += 1

print("Unique Carriers:", len(carrier_counter))

print("\nTop 20 Carriers:\n")

for carrier, count in carrier_counter.most_common(20):

    print(f"{carrier:<40} {count:,}")

# =========================================================
# PATHWAY ANALYSIS
# =========================================================

print("\n===== PATHWAY ANALYSIS =====\n")

pathway_counter = Counter()

sample_pathways = []

for row in read_jsonl(PATHWAYS_FILE):

    pathway = row.get(
        "pathway"
    )

    if pathway:

        pathway_counter[pathway] += 1

    if len(sample_pathways) < 10:

        sample_pathways.append(row)

print("Unique Pathways:", len(pathway_counter))

print("\nTop 20 Pathways:\n")

for pathway, count in pathway_counter.most_common(20):

    print(f"{pathway:<60} {count:,}")

print("\nSample Pathway Rows:\n")

for row in sample_pathways:

    print(json.dumps(row, indent=2))

# =========================================================
# CATEGORY ANALYSIS
# =========================================================

print("\n===== CATEGORY ANALYSIS =====\n")

category_counter = Counter()

for row in read_jsonl(CATEGORIES_FILE):

    category_counter[
        row["category"]
    ] += 1

print("Unique Categories:", len(category_counter))

print("\nTop 30 Categories:\n")

for category, count in category_counter.most_common(30):

    print(f"{category:<50} {count:,}")

# =========================================================
# MECHANISM ANALYSIS
# =========================================================

print("\n===== MECHANISM ANALYSIS =====\n")

mechanism_lengths = []

sample_mechanisms = []

for row in read_jsonl(MECHANISMS_FILE):

    mechanism = row["mechanism"]

    mechanism_lengths.append(
        len(mechanism)
    )

    if len(sample_mechanisms) < 5:

        sample_mechanisms.append(row)

avg_length = (
    sum(mechanism_lengths)
    /
    len(mechanism_lengths)
)

print("Mechanism Entries:", len(mechanism_lengths))

print("Average Length:", round(avg_length, 2))

print("\nSample Mechanisms:\n")

for row in sample_mechanisms:

    print(json.dumps(row, indent=2))

# =========================================================
# DRUGBANK INTERACTION ANALYSIS
# =========================================================

print("\n===== DRUGBANK INTERACTION ANALYSIS =====\n")

interaction_counter = 0

sample_interactions = []

for row in read_jsonl(DRUG_INTERACTIONS_FILE):

    interaction_counter += 1

    if len(sample_interactions) < 10:

        sample_interactions.append(row)

print("Total DrugBank Interactions:",
      f"{interaction_counter:,}")

print("\nSample DrugBank Interactions:\n")

for row in sample_interactions:

    print(json.dumps(row, indent=2))

# =========================================================
# TWOSIDES ANALYSIS
# =========================================================

print("\n===== TWOSIDES ANALYSIS =====\n")

ade_counter = Counter()

prr_values = []

sample_twosides = []

MAX_ROWS = 1_000_000

for i, row in enumerate(
    read_jsonl(TWOSIDES_FILE)
):

    ade = row["ade"]

    ade_counter[ade] += 1

    try:

        prr = float(row["prr"])

        prr_values.append(prr)

    except:
        pass

    if len(sample_twosides) < 10:

        sample_twosides.append(row)

    if i >= MAX_ROWS:

        break

print("Sampled Rows:", f"{MAX_ROWS:,}")

print("Unique ADEs:", len(ade_counter))

if prr_values:

    avg_prr = (
        sum(prr_values)
        /
        len(prr_values)
    )

    print("Average PRR:",
          round(avg_prr, 4))

print("\nTop 30 ADEs:\n")

for ade, count in ade_counter.most_common(30):

    print(f"{ade:<40} {count:,}")

print("\nSample TWOSIDES Rows:\n")

for row in sample_twosides:

    print(json.dumps(row, indent=2))

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n===== INSPECTION COMPLETE =====\n")

print("Now inspect:")
print("- action quality")
print("- biological sparsity")
print("- PK vs PD richness")
print("- pathway usefulness")
print("- interaction descriptions")
print("- ADE distributions")

print("\nDone.")