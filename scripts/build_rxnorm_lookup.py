import orjson
from pathlib import Path

INPUT_PATH = Path(
    "data/processed/drugbank_external_ids.jsonl"
)

OUTPUT_DIR = Path(
    "data/processed"
)

drugbank_to_rxnorm = {}
rxnorm_to_drugbank = {}

with open(INPUT_PATH, "rb") as f:

    for line in f:

        row = orjson.loads(line)

        dbid = row["drugbank_id"]
        rx = row.get("rxnorm")

        if rx:

            drugbank_to_rxnorm[dbid] = rx

            rxnorm_to_drugbank[rx] = {
                "drugbank_id": dbid,
                "name": row["name"]
            }

# -----------------------------------
# SAVE
# -----------------------------------

with open(
    OUTPUT_DIR / "drugbank_to_rxnorm.json",
    "wb"
) as f:

    f.write(
        orjson.dumps(
            drugbank_to_rxnorm
        )
    )

with open(
    OUTPUT_DIR / "rxnorm_to_drugbank.json",
    "wb"
) as f:

    f.write(
        orjson.dumps(
            rxnorm_to_drugbank
        )
    )

print("\nLookup tables built successfully.")