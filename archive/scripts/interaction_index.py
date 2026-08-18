import orjson
from pathlib import Path
from collections import defaultdict

PROCESSED = Path("data/processed")


class InteractionIndex:

    def __init__(self):

        self.drug_to_enzymes = defaultdict(list)
        self.drug_to_transporters = defaultdict(list)
        self.drug_to_targets = defaultdict(list)

        self.name_to_drugbank = {}

        self._load_drugs()
        self._load_enzymes()
        self._load_transporters()
        self._load_targets()

    # -----------------------------------
    # DRUG NAME INDEX
    # -----------------------------------

    def _load_drugs(self):

        path = PROCESSED / "drugs.jsonl"

        with open(path, "rb") as f:

            for line in f:

                row = orjson.loads(line)

                dbid = row["drugbank_id"]
                name = row["name"]

                self.name_to_drugbank[
                    name.lower()
                ] = dbid

    # -----------------------------------
    # ENZYMES
    # -----------------------------------

    def _load_enzymes(self):

        path = PROCESSED / "drug_enzymes.jsonl"

        with open(path, "rb") as f:

            for line in f:

                row = orjson.loads(line)

                self.drug_to_enzymes[
                    row["drug"]
                ].append({
                    "enzyme": row["enzyme"],
                    "actions": row["actions"]
                })

    # -----------------------------------
    # TRANSPORTERS
    # -----------------------------------

    def _load_transporters(self):

        path = PROCESSED / "drug_transporters.jsonl"

        with open(path, "rb") as f:

            for line in f:

                row = orjson.loads(line)

                self.drug_to_transporters[
                    row["drug"]
                ].append({
                    "transporter": row["transporter"],
                    "actions": row["actions"]
                })

    # -----------------------------------
    # TARGETS
    # -----------------------------------

    def _load_targets(self):

        path = PROCESSED / "drug_targets.jsonl"

        with open(path, "rb") as f:

            for line in f:

                row = orjson.loads(line)

                self.drug_to_targets[
                    row["drug"]
                ].append(row["target"])

    # -----------------------------------
    # LOOKUP HELPERS
    # -----------------------------------

    def resolve_drug(self, name_or_id):

        if name_or_id.startswith("DB"):
            return name_or_id

        return self.name_to_drugbank.get(
            name_or_id.lower()
        )