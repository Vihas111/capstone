from pathlib import Path
import orjson
from collections import Counter

PROCESSED = Path("data/processed")


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


files = [
    "drugs.jsonl",
    "drug_enzymes.jsonl",
    "drug_transporters.jsonl",
    "drug_targets.jsonl",
    "drug_text.jsonl"
]

print("\n========== FILE COUNTS ==========\n")

for file in files:
    path = PROCESSED / file

    if path.exists():
        print(f"{file}: {count_lines(path):,}")
    else:
        print(f"{file}: MISSING")


# -----------------------------------
# ENZYME ACTION ANALYSIS
# -----------------------------------

enzyme_actions = Counter()

with open(PROCESSED / "drug_enzymes.jsonl", "rb") as f:
    for line in f:
        row = orjson.loads(line)

        for action in row.get("actions", []):
            enzyme_actions[action] += 1

print("\n========== ENZYME ACTIONS ==========\n")

for k, v in enzyme_actions.most_common(20):
    print(f"{k}: {v:,}")


# -----------------------------------
# TRANSPORTER ACTION ANALYSIS
# -----------------------------------

transporter_actions = Counter()
missing_transporter_actions = 0
total_transporters = 0

with open(PROCESSED / "drug_transporters.jsonl", "rb") as f:
    for line in f:
        row = orjson.loads(line)

        total_transporters += 1

        actions = row.get("actions", [])

        if not actions:
            missing_transporter_actions += 1

        for action in actions:
            transporter_actions[action] += 1

print("\n========== TRANSPORTER ACTIONS ==========\n")

print(f"Total transporter rows: {total_transporters:,}")
print(f"Missing actions: {missing_transporter_actions:,}")

print("\nTop transporter actions:\n")

for k, v in transporter_actions.most_common(20):
    print(f"{k}: {v:,}")


# -----------------------------------
# SAMPLE RECORDS
# -----------------------------------

print("\n========== SAMPLE RECORDS ==========\n")

sample_files = [
    "drugs.jsonl",
    "drug_enzymes.jsonl",
    "drug_transporters.jsonl",
    "drug_targets.jsonl",
]

for sf in sample_files:

    print(f"\n--- {sf} ---\n")

    with open(PROCESSED / sf, "rb") as f:
        for i, line in enumerate(f):
            print(orjson.loads(line))

            if i >= 2:
                break