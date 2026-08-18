# =========================================================
# extract_smiles.py
# =========================================================
#
# PURPOSE:
#   Extract each drug's SMILES (chemical structure string) from the raw
#   DrugBank XML. This feeds scripts/build_fingerprints.py, which needs a
#   structure to compute a molecular fingerprint from -- the input feature
#   for the mechanism gap-filler model (see CLAUDE.md section 3b).
#
#   SMILES lives under db:calculated-properties/db:property where
#   db:kind == "SMILES" (confirmed directly against data/raw/drugdatabase.xml;
#   NOT under db:experimental-properties, which holds unrelated properties
#   like logP/melting point). Only ~73.7% of DrugBank's ~19,830 drugs have
#   one -- biologics (proteins, antibodies) generally don't, since SMILES
#   represents small-molecule structure.
#
#   Deliberately a standalone script, NOT an extension of
#   scripts/parse_drugbank.py -- that script is dead code and is not the
#   source of the data/processed/*.jsonl files currently on disk (verified
#   via mtime + write-list mismatch against scripts/extract_biomedical_features.py,
#   which is the actual live ingestion script). Mirrors that script's
#   ET.parse()/NS/get_text() conventions instead, since that's the proven,
#   already-working way to walk this exact XML file.
#
# OUTPUT:
#   data/processed/drug_smiles.jsonl  -- {"drugbank_id": ..., "smiles": ...}
#   one row per drug that HAS a SMILES value (absence from the file is the
#   signal for "no SMILES", same convention drug_enzymes.jsonl etc. use).
#
# =========================================================

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"db": "http://www.drugbank.ca"}


def get_text(element):

    if element is None:
        return None

    if element.text is None:
        return None

    return element.text.strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", default="data/raw/drugdatabase.xml")
    p.add_argument("--out", default="data/processed/drug_smiles.jsonl")
    return p.parse_args()


def main():

    args = parse_args()

    print(f"Parsing {args.xml} for SMILES...")

    tree = ET.parse(args.xml)
    root = tree.getroot()

    rows = []
    n_drugs = 0
    n_with_smiles = 0

    for drug in root.findall("db:drug", NS):

        n_drugs += 1

        dbid_elem = drug.find("db:drugbank-id[@primary='true']", NS)
        drugbank_id = get_text(dbid_elem)

        if not drugbank_id:
            continue

        smiles = None

        calc_props = drug.find("db:calculated-properties", NS)

        if calc_props is not None:

            for prop in calc_props.findall("db:property", NS):

                kind = get_text(prop.find("db:kind", NS))

                if kind == "SMILES":
                    smiles = get_text(prop.find("db:value", NS))
                    break

        if smiles:
            rows.append({"drugbank_id": drugbank_id, "smiles": smiles})
            n_with_smiles += 1

        if n_drugs % 2000 == 0:
            print(f"  ...{n_drugs} drugs scanned")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nScanned {n_drugs} drugs, {n_with_smiles} had a SMILES value "
          f"({100 * n_with_smiles / n_drugs:.1f}%).")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
