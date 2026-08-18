# =========================================================
# validate_pair_mapping.py
# =========================================================
#
# PURPOSE:
#   Validate TWOSIDES ↔ DrugBank alignment
#
# GOALS:
#   - determine usable TWOSIDES coverage
#   - count mapped drug pairs
#   - count unique drug pairs
#   - inspect ADE diversity
#   - estimate final training scale
#
# =========================================================

import json

from collections import Counter

# =========================================================
# PATHS
# =========================================================

RXNORM_MAPPING = (
    "data/processed/rxnorm_to_drugbank.json"
)

TWOSIDES_FILE = (
    "data/processed/twosides_interactions.jsonl"
)

# =========================================================
# LOAD MAPPING
# =========================================================

print("Loading RxNorm mapping...")

with open(
    RXNORM_MAPPING,
    encoding="utf-8"
) as f:

    rxnorm_to_drugbank = json.load(f)

print(
    "Loaded mappings:",
    f"{len(rxnorm_to_drugbank):,}"
)

# =========================================================
# COUNTERS
# =========================================================

total_rows = 0

mapped_rows = 0

unmapped_rows = 0

unique_pairs = set()

ade_counter = Counter()

drug_counter = Counter()

# =========================================================
# PROCESS TWOSIDES
# =========================================================

print("\nProcessing TWOSIDES...\n")

with open(
    TWOSIDES_FILE,
    encoding="utf-8"
) as f:

    for i, line in enumerate(f):

        total_rows += 1

        try:

            row = json.loads(line)

            rx1 = row["rxnorm_1"]
            rx2 = row["rxnorm_2"]

            db1 = rxnorm_to_drugbank.get(rx1)
            db2 = rxnorm_to_drugbank.get(rx2)

            # =============================================
            # VALID PAIR
            # =============================================

            if db1 and db2:

                mapped_rows += 1

                # canonical ordering

                pair = tuple(sorted([db1, db2]))

                unique_pairs.add(pair)

                drug_counter[db1] += 1
                drug_counter[db2] += 1

                ade = row["ade"]

                ade_counter[ade] += 1

            else:

                unmapped_rows += 1

        except Exception:
            continue

        # =============================================
        # PROGRESS
        # =============================================

        if i % 1_000_000 == 0 and i > 0:

            print(
                f"Processed {i:,} rows"
            )

# =========================================================
# RESULTS
# =========================================================

print("\n===== VALIDATION RESULTS =====\n")

print(
    "Total TWOSIDES Rows:",
    f"{total_rows:,}"
)

print(
    "Mapped Rows:",
    f"{mapped_rows:,}"
)

print(
    "Unmapped Rows:",
    f"{unmapped_rows:,}"
)

coverage = (
    mapped_rows
    /
    total_rows
) * 100

print(
    "Coverage:",
    f"{coverage:.2f}%"
)

print()

print(
    "Unique Drug Pairs:",
    f"{len(unique_pairs):,}"
)

print(
    "Unique Drugs Used:",
    f"{len(drug_counter):,}"
)

print(
    "Unique ADEs:",
    f"{len(ade_counter):,}"
)

# =========================================================
# TOP DRUGS
# =========================================================

print("\n===== TOP 30 MOST COMMON DRUGS =====\n")

for drug, count in drug_counter.most_common(30):

    print(f"{drug:<20} {count:,}")

# =========================================================
# TOP ADEs
# =========================================================

print("\n===== TOP 30 ADEs =====\n")

for ade, count in ade_counter.most_common(30):

    print(f"{ade:<40} {count:,}")

# =========================================================
# SAMPLE PAIRS
# =========================================================

print("\n===== SAMPLE PAIRS =====\n")

sample_pairs = list(unique_pairs)[:30]

for pair in sample_pairs:

    print(pair)

# =========================================================
# FINAL NOTES
# =========================================================

print("\n===== INTERPRETATION =====\n")

print(
    "Mapped rows represent TWOSIDES interactions"
)

print(
    "that can be connected to DrugBank PK/PD biology."
)

print()

print(
    "Unique pairs estimate the approximate"
)

print(
    "mechanistic graph construction workload."
)

print("\nDone.")