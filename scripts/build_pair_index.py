# =========================================================
# build_pair_index.py
# =========================================================
#
# PURPOSE:
#   Build fast pair -> ADE lookup index
#
# OUTPUT:
#   data/indexed/pair_to_ades.json
#
# =========================================================

import json
import os

from collections import defaultdict

# =========================================================
# PATHS
# =========================================================

DATA_DIR = "data/processed/"

OUTPUT_DIR = "data/indexed/"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

TWOSIDES_FILE = (
    DATA_DIR +
    "twosides_interactions.jsonl"
)

OUTPUT_FILE = (
    OUTPUT_DIR +
    "pair_to_ades.json"
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
# BUILD INDEX
# =========================================================

print("Building pair index...")

pair_index = defaultdict(list)

for i, row in enumerate(
    read_jsonl(TWOSIDES_FILE)
):

    rx1 = row["rxnorm_1"]
    rx2 = row["rxnorm_2"]

    pair_key = "__".join(
        sorted([rx1, rx2])
    )

    pair_index[pair_key].append({

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

    if i % 1_000_000 == 0 and i > 0:

        print(
            f"Processed {i:,} rows"
        )

# =========================================================
# SAVE
# =========================================================

print("\nSaving index...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        pair_index,
        f
    )

# =========================================================
# SUMMARY
# =========================================================

print("\n===== INDEX SUMMARY =====\n")

print(
    "Unique pairs:",
    len(pair_index)
)

print(
    "Saved:"
)

print(OUTPUT_FILE)

print("\nDone.")