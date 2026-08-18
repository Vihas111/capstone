import gzip
import csv
import orjson
from pathlib import Path


class TWOSIDESLookup:

    def __init__(self):

        # -----------------------------------
        # PATHS
        # -----------------------------------

        self.twosides_path = Path(
            "../dataset_explore/twosides/raw/TWOSIDES.csv.gz"
        )

        # -----------------------------------
        # LOAD LOOKUPS
        # -----------------------------------

        with open(
            "data/processed/drugbank_to_rxnorm.json",
            "rb"
        ) as f:

            self.drugbank_to_rxnorm = orjson.loads(
                f.read()
            )

    # -----------------------------------
    # MAIN LOOKUP
    # -----------------------------------

    def lookup_pair(
        self,
        drug1_dbid,
        drug2_dbid,
        min_prr=1.0,
        top_k=20
    ):

        rx1 = self.drugbank_to_rxnorm.get(
            drug1_dbid
        )

        rx2 = self.drugbank_to_rxnorm.get(
            drug2_dbid
        )

        if not rx1 or not rx2:

            return {
                "error": "RxNorm mapping missing"
            }

        matches = []

        with gzip.open(
            self.twosides_path,
            "rt",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                d1 = row["drug_1_rxnorn_id"]
                d2 = row["drug_2_rxnorm_id"]

                # -----------------------------------
                # MATCH BOTH DIRECTIONS
                # -----------------------------------

                pair_match = (
                    (d1 == rx1 and d2 == rx2)
                    or
                    (d1 == rx2 and d2 == rx1)
                )

                if not pair_match:
                    continue

                # -----------------------------------
                # PARSE PRR
                # -----------------------------------

                try:
                    prr = float(row["PRR"])
                except:
                    continue

                if prr < min_prr:
                    continue

                matches.append({
                    "effect":
                        row["condition_concept_name"],

                    "meddra_id":
                        row["condition_meddra_id"],

                    "PRR":
                        prr,

                    "frequency":
                        row[
                            "mean_reporting_frequency"
                        ]
                })

        # -----------------------------------
        # SORT BY PRR
        # -----------------------------------

        matches = sorted(
            matches,
            key=lambda x: x["PRR"],
            reverse=True
        )

        return matches[:top_k]


# -----------------------------------
# TEST
# -----------------------------------

if __name__ == "__main__":

    lookup = TWOSIDESLookup()

    results = lookup.lookup_pair(
        "DB00682",  # Warfarin
        "DB00196"   # Fluconazole
    )

    import pprint
    pprint.pprint(results)