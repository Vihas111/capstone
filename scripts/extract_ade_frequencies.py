# =========================================================
# extract_ade_frequencies.py
# =========================================================

import csv
import gzip
import json

from collections import Counter

# =========================================================
# COUNTER
# =========================================================

ade_counter = Counter()

# =========================================================
# READ TWOSIDES
# =========================================================

print("Reading TWOSIDES...")

with gzip.open(
    "../dataset_explore/twosides/raw/TWOSIDES.csv.gz",
    "rt",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for i, row in enumerate(reader):

        try:

            ade = (
                row["condition_concept_name"]
                .lower()
                .strip()
            )

            if ade:
                ade_counter[ade] += 1

        except Exception:
            continue

        # progress update
        if i % 1_000_000 == 0 and i > 0:
            print(f"Processed {i:,} rows")

# =========================================================
# RESULTS
# =========================================================

print("\n===== TOP 50 ADEs =====\n")

for ade, count in ade_counter.most_common(50):

    print(f"{ade:<50} {count:,}")

# =========================================================
# UNIQUE ADE COUNT
# =========================================================

print("\n===== TOTAL UNIQUE ADEs =====\n")

print(len(ade_counter))

# =========================================================
# SAVE OUTPUT
# =========================================================

with open("ade_frequencies.json", "w") as f:

    json.dump(
        dict(ade_counter.most_common()),
        f,
        indent=2
    )

print("\nSaved ade_frequencies.json")
print("\nDone.")