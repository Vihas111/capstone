import orjson
from pathlib import Path

INPUT_PATH = Path(
    "data/processed/drugbank_external_ids.jsonl"
)

total = 0
mapped_rxnorm = 0
missing_rxnorm = 0

examples = []

with open(INPUT_PATH, "rb") as f:

    for line in f:

        row = orjson.loads(line)

        total += 1

        if row.get("rxnorm"):

            mapped_rxnorm += 1

            if len(examples) < 10:
                examples.append(row)

        else:
            missing_rxnorm += 1

print("\n========== RXNORM COVERAGE ==========\n")

print(f"Total drugs: {total:,}")
print(f"Mapped RxNorm IDs: {mapped_rxnorm:,}")
print(f"Missing RxNorm IDs: {missing_rxnorm:,}")

coverage = mapped_rxnorm / total * 100

print(f"\nCoverage: {coverage:.2f}%")

print("\n========== SAMPLE MAPPINGS ==========\n")

for ex in examples:

    print({
        "drugbank_id": ex["drugbank_id"],
        "name": ex["name"],
        "rxnorm": ex["rxnorm"]
    })