import gzip
import csv
import orjson
from pathlib import Path

# -----------------------------------
# LOAD DRUGBANK RXNORM MAP
# -----------------------------------

MAP_PATH = Path(
    "data/processed/drugbank_external_ids.jsonl"
)

rxnorm_to_drugbank = {}

with open(MAP_PATH, "rb") as f:

    for line in f:

        row = orjson.loads(line)

        rx = row.get("rxnorm")

        if rx:

            rxnorm_to_drugbank[str(rx)] = {
                "drugbank_id": row["drugbank_id"],
                "name": row["name"]
            }

print(
    f"\nLoaded {len(rxnorm_to_drugbank):,} "
    f"RxNorm mappings"
)

# -----------------------------------
# LOAD TWOSIDES SAMPLE
# -----------------------------------

TWOSIDES_PATH = Path(
    "../dataset_explore/twosides/raw/TWOSIDES.csv.gz"
)

total_rows = 0
fully_mapped = 0
partial_mapped = 0
unmapped = 0

sample_matches = []

seen_drugs = set()
mapped_drugs = set()

with gzip.open(TWOSIDES_PATH, "rt", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        total_rows += 1

        d1 = row["drug_1_rxnorn_id"]
        d2 = row["drug_2_rxnorm_id"]

        seen_drugs.add(d1)
        seen_drugs.add(d2)

        d1_mapped = d1 in rxnorm_to_drugbank
        d2_mapped = d2 in rxnorm_to_drugbank

        if d1_mapped:
            mapped_drugs.add(d1)

        if d2_mapped:
            mapped_drugs.add(d2)

        if d1_mapped and d2_mapped:

            fully_mapped += 1

            if len(sample_matches) < 10:

                sample_matches.append({
                    "drug1": row["drug_1_concept_name"],
                    "drug2": row["drug_2_concept_name"],
                    "drug1_db": rxnorm_to_drugbank[d1],
                    "drug2_db": rxnorm_to_drugbank[d2]
                })

        elif d1_mapped or d2_mapped:
            partial_mapped += 1

        else:
            unmapped += 1

        # limit full scan for speed initially
        if total_rows >= 1_000_000:
            break

# -----------------------------------
# RESULTS
# -----------------------------------

print("\n========== MAPPING RESULTS ==========\n")

print(f"Rows checked: {total_rows:,}")
print(f"Fully mapped pairs: {fully_mapped:,}")
print(f"Partially mapped pairs: {partial_mapped:,}")
print(f"Unmapped pairs: {unmapped:,}")

coverage = (
    fully_mapped / total_rows * 100
)

print(f"\nFull pair coverage: {coverage:.2f}%")

print(
    f"\nTWOSIDES unique drugs seen: "
    f"{len(seen_drugs):,}"
)

print(
    f"Mapped TWOSIDES drugs: "
    f"{len(mapped_drugs):,}"
)

drug_coverage = (
    len(mapped_drugs) / len(seen_drugs) * 100
)

print(
    f"\nDrug mapping coverage: "
    f"{drug_coverage:.2f}%"
)

print("\n========== SAMPLE MATCHES ==========\n")

for s in sample_matches:

    print("\n------------------")

    print(
        f"{s['drug1']} "
        f"-> "
        f"{s['drug1_db']['drugbank_id']}"
    )

    print(
        f"{s['drug2']} "
        f"-> "
        f"{s['drug2_db']['drugbank_id']}"
    )