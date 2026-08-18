# =========================================================
# build_rxnorm_mapping.py
# =========================================================
#
# PURPOSE:
#   Build RxNorm ↔ DrugBank mappings
#
# OUTPUT:
#   data/processed/
#
#       rxnorm_to_drugbank.json
#       drugbank_to_rxnorm.json
#
# =========================================================

import json
import xml.etree.ElementTree as ET

# =========================================================
# PATHS
# =========================================================

DRUGBANK_XML = (
    "data/raw/drugdatabase.xml"
)

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

rxnorm_to_drugbank = {}

drugbank_to_rxnorm = {}

# =========================================================
# LOAD XML
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
        # DRUGBANK ID
        # =================================================

        dbid_elem = drug.find(
            "db:drugbank-id[@primary='true']",
            NS
        )

        drugbank_id = get_text(dbid_elem)

        if not drugbank_id:
            continue

        # =================================================
        # EXTERNAL IDENTIFIERS
        # =================================================

        external_ids = drug.find(
            "db:external-identifiers",
            NS
        )

        if external_ids is None:
            continue

        rxnorm_id = None

        for ext in external_ids.findall(
            "db:external-identifier",
            NS
        ):

            resource = get_text(
                ext.find(
                    "db:resource",
                    NS
                )
            )

            identifier = get_text(
                ext.find(
                    "db:identifier",
                    NS
                )
            )

            # =============================================
            # RXNORM / RXCUI MATCH
            # =============================================

            if resource is None:
                continue

            resource_lower = resource.lower()

            if (

                "rxnorm" in resource_lower
                or
                "rxcui" in resource_lower

            ):

                rxnorm_id = identifier
                break

        # =================================================
        # STORE MAPPING
        # =================================================

        if rxnorm_id:

            rxnorm_to_drugbank[
                rxnorm_id
            ] = drugbank_id

            drugbank_to_rxnorm[
                drugbank_id
            ] = rxnorm_id

        # =================================================
        # PROGRESS
        # =================================================

        if i % 1000 == 0 and i > 0:

            print(
                f"Processed {i:,} drugs"
            )

    except Exception as e:

        print("Parse failure:", e)

# =========================================================
# SAVE JSON
# =========================================================

print("\nSaving mappings...")

with open(
    OUT_DIR + "rxnorm_to_drugbank.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rxnorm_to_drugbank,
        f,
        indent=2
    )

with open(
    OUT_DIR + "drugbank_to_rxnorm.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        drugbank_to_rxnorm,
        f,
        indent=2
    )

# =========================================================
# SUMMARY
# =========================================================

print("\n===== MAPPING SUMMARY =====\n")

print(
    "RxNorm → DrugBank mappings:",
    f"{len(rxnorm_to_drugbank):,}"
)

print(
    "DrugBank → RxNorm mappings:",
    f"{len(drugbank_to_rxnorm):,}"
)

# =========================================================
# SAMPLE OUTPUT
# =========================================================

print("\n===== SAMPLE MAPPINGS =====\n")

sample_items = list(
    rxnorm_to_drugbank.items()
)[:20]

for rxnorm, drugbank in sample_items:

    print(
        f"{rxnorm:<15}"
        f"{drugbank}"
    )

print("\nDone.")