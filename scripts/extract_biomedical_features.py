# =========================================================
# extract_biomedical_features.py
# =========================================================
#
# PURPOSE:
#   Extract rich PK + PD biomedical data from:
#
#   1. DrugBank XML
#   2. TWOSIDES
#
# OUTPUT:
#   data/processed/
#
#       drugs.jsonl
#       drug_targets.jsonl
#       drug_enzymes.jsonl
#       drug_transporters.jsonl
#       drug_carriers.jsonl
#       drug_categories.jsonl
#       drug_mechanisms.jsonl
#       drug_pathways.jsonl
#       drug_indications.jsonl
#       drugbank_interactions.jsonl
#       twosides_interactions.jsonl
#
# =========================================================

import csv
import gzip
import json
import xml.etree.ElementTree as ET

# =========================================================
# PATHS
# =========================================================

DRUGBANK_XML = (
    "data/raw/drugdatabase.xml"
)

TWOSIDES_PATH = (
    "../dataset_explore/twosides/raw/TWOSIDES.csv.gz"
)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUT_DIR = "data/processed/"

# =========================================================
# XML NAMESPACE
# =========================================================

NS = {
    "db": "http://www.drugbank.ca"
}

# =========================================================
# HELPERS
# =========================================================

def get_text(element):

    if element is None:
        return None

    if element.text is None:
        return None

    return element.text.strip()

# =========================================================
# STORAGE
# =========================================================

drugs = []
targets = []
enzymes = []
transporters = []
carriers = []
categories = []
mechanisms = []
pathways = []
indications = []
drugbank_interactions = []

# =========================================================
# LOAD DRUGBANK
# =========================================================

print("Parsing DrugBank XML...")

tree = ET.parse(DRUGBANK_XML)

root = tree.getroot()

# =========================================================
# PARSE DRUGS
# =========================================================

for i, drug in enumerate(root.findall("db:drug", NS)):

    try:

        # =================================================
        # DRUG ID
        # =================================================

        dbid_elem = drug.find(
            "db:drugbank-id[@primary='true']",
            NS
        )

        drugbank_id = get_text(dbid_elem)

        if not drugbank_id:
            continue

        # =================================================
        # DRUG NAME
        # =================================================

        name = get_text(
            drug.find("db:name", NS)
        )

        # =================================================
        # BASIC DRUG OBJECT
        # =================================================

        drugs.append({

            "drugbank_id": drugbank_id,
            "name": name

        })

        # =================================================
        # TARGETS
        # =================================================

        target_root = drug.find(
            "db:targets",
            NS
        )

        if target_root is not None:

            for target in target_root.findall(
                "db:target",
                NS
            ):

                polypeptide = target.find(
                    "db:polypeptide",
                    NS
                )

                target_name = None

                if polypeptide is not None:

                    target_name = get_text(
                        polypeptide.find(
                            "db:name",
                            NS
                        )
                    )

                actions = []

                actions_root = target.find(
                    "db:actions",
                    NS
                )

                if actions_root is not None:

                    for action in actions_root.findall(
                        "db:action",
                        NS
                    ):

                        txt = get_text(action)

                        if txt:
                            actions.append(txt)

                if target_name:

                    targets.append({

                        "drug": drugbank_id,
                        "target": target_name,
                        "actions": actions

                    })

        # =================================================
        # ENZYMES
        # =================================================

        enzyme_root = drug.find(
            "db:enzymes",
            NS
        )

        if enzyme_root is not None:

            for enzyme in enzyme_root.findall(
                "db:enzyme",
                NS
            ):

                enzyme_name = get_text(
                    enzyme.find(
                        "db:name",
                        NS
                    )
                )

                actions = []

                actions_root = enzyme.find(
                    "db:actions",
                    NS
                )

                if actions_root is not None:

                    for action in actions_root.findall(
                        "db:action",
                        NS
                    ):

                        txt = get_text(action)

                        if txt:
                            actions.append(txt)

                if enzyme_name:

                    enzymes.append({

                        "drug": drugbank_id,
                        "enzyme": enzyme_name,
                        "actions": actions

                    })

        # =================================================
        # TRANSPORTERS
        # =================================================

        transporter_root = drug.find(
            "db:transporters",
            NS
        )

        if transporter_root is not None:

            for transporter in transporter_root.findall(
                "db:transporter",
                NS
            ):

                transporter_name = get_text(
                    transporter.find(
                        "db:name",
                        NS
                    )
                )

                actions = []

                actions_root = transporter.find(
                    "db:actions",
                    NS
                )

                if actions_root is not None:

                    for action in actions_root.findall(
                        "db:action",
                        NS
                    ):

                        txt = get_text(action)

                        if txt:
                            actions.append(txt)

                if transporter_name:

                    transporters.append({

                        "drug": drugbank_id,
                        "transporter": transporter_name,
                        "actions": actions

                    })

        # =================================================
        # CARRIERS
        # =================================================

        carrier_root = drug.find(
            "db:carriers",
            NS
        )

        if carrier_root is not None:

            for carrier in carrier_root.findall(
                "db:carrier",
                NS
            ):

                carrier_name = get_text(
                    carrier.find(
                        "db:name",
                        NS
                    )
                )

                if carrier_name:

                    carriers.append({

                        "drug": drugbank_id,
                        "carrier": carrier_name

                    })

        # =================================================
        # CATEGORIES
        # =================================================

        category_root = drug.find(
            "db:categories",
            NS
        )

        if category_root is not None:

            for category in category_root.findall(
                "db:category",
                NS
            ):

                category_name = get_text(
                    category.find(
                        "db:category",
                        NS
                    )
                )

                if category_name:

                    categories.append({

                        "drug": drugbank_id,
                        "category": category_name

                    })

        # =================================================
        # MECHANISM OF ACTION
        # =================================================

        moa = get_text(
            drug.find(
                "db:mechanism-of-action",
                NS
            )
        )

        if moa:

            mechanisms.append({

                "drug": drugbank_id,
                "mechanism": moa

            })

        # =================================================
        # INDICATIONS
        # =================================================

        indication = get_text(
            drug.find(
                "db:indication",
                NS
            )
        )

        if indication:

            indications.append({

                "drug": drugbank_id,
                "indication": indication

            })

        # =================================================
        # PATHWAYS
        # =================================================

        pathway_root = drug.find(
            "db:pathways",
            NS
        )

        if pathway_root is not None:

            for pathway in pathway_root.findall(
                "db:pathway",
                NS
            ):

                pathway_name = get_text(
                    pathway.find(
                        "db:name",
                        NS
                    )
                )

                smpdb_id = get_text(
                    pathway.find(
                        "db:smpdb-id",
                        NS
                    )
                )

                pathways.append({

                    "drug": drugbank_id,
                    "pathway": pathway_name,
                    "smpdb_id": smpdb_id

                })

        # =================================================
        # DRUG INTERACTIONS
        # =================================================

        interaction_root = drug.find(
            "db:drug-interactions",
            NS
        )

        if interaction_root is not None:

            for interaction in interaction_root.findall(
                "db:drug-interaction",
                NS
            ):

                other_drug = get_text(
                    interaction.find(
                        "db:drugbank-id",
                        NS
                    )
                )

                description = get_text(
                    interaction.find(
                        "db:description",
                        NS
                    )
                )

                if other_drug:

                    drugbank_interactions.append({

                        "drug1": drugbank_id,
                        "drug2": other_drug,
                        "description": description

                    })

        # =================================================
        # PROGRESS
        # =================================================

        if i % 1000 == 0 and i > 0:

            print(f"Processed {i:,} drugs")

    except Exception as e:

        print("Drug parse failure:", e)

# =========================================================
# SAVE JSONL
# =========================================================

def save_jsonl(filename, data):

    path = OUT_DIR + filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for row in data:

            f.write(
                json.dumps(row) + "\n"
            )

    print(f"Saved {filename}")

# =========================================================
# SAVE DRUGBANK OUTPUTS
# =========================================================

print("\nSaving DrugBank outputs...")

save_jsonl("drugs.jsonl", drugs)

save_jsonl("drug_targets.jsonl", targets)

save_jsonl("drug_enzymes.jsonl", enzymes)

save_jsonl("drug_transporters.jsonl", transporters)

save_jsonl("drug_carriers.jsonl", carriers)

save_jsonl("drug_categories.jsonl", categories)

save_jsonl("drug_mechanisms.jsonl", mechanisms)

save_jsonl("drug_pathways.jsonl", pathways)

save_jsonl("drug_indications.jsonl", indications)

save_jsonl(
    "drugbank_interactions.jsonl",
    drugbank_interactions
)

# =========================================================
# TWOSIDES EXTRACTION
# =========================================================

print("\nProcessing TWOSIDES...")

output_path = (
    OUT_DIR +
    "twosides_interactions.jsonl"
)

count = 0

with gzip.open(
    TWOSIDES_PATH,
    "rt",
    encoding="utf-8"
) as infile, open(
    output_path,
    "w",
    encoding="utf-8"
) as outfile:

    reader = csv.DictReader(infile)

    for i, row in enumerate(reader):

        try:

            output_row = {

                "rxnorm_1":
                    row["drug_1_rxnorn_id"],

                "rxnorm_2":
                    row["drug_2_rxnorm_id"],

                "drug_1":
                    row["drug_1_concept_name"],

                "drug_2":
                    row["drug_2_concept_name"],

                "ade":
                    row["condition_concept_name"],

                "prr":
                    row["PRR"],

                "prr_error":
                    row["PRR_error"],

                "mean_reporting_frequency":
                    row["mean_reporting_frequency"]

            }

            outfile.write(
                json.dumps(output_row) + "\n"
            )

            count += 1

        except Exception:
            continue

        if i % 1_000_000 == 0 and i > 0:

            print(
                f"Processed {i:,} TWOSIDES rows"
            )

print(
    "\nSaved twosides_interactions.jsonl"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n===== EXTRACTION COMPLETE =====\n")

print(f"Drugs: {len(drugs):,}")
print(f"Targets: {len(targets):,}")
print(f"Enzymes: {len(enzymes):,}")
print(f"Transporters: {len(transporters):,}")
print(f"Carriers: {len(carriers):,}")
print(f"Pathways: {len(pathways):,}")
print(f"Drug Interactions: {len(drugbank_interactions):,}")
print(f"TWOSIDES Rows: {count:,}")

print("\nDone.")