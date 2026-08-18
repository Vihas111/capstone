import xml.etree.ElementTree as ET
import orjson
from pathlib import Path
from tqdm import tqdm

XML_PATH = Path("data/raw/drugdatabase.xml")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NS = {"db": "http://www.drugbank.ca"}

drugs_file = open(OUTPUT_DIR / "drugs.jsonl", "wb")
enzymes_file = open(OUTPUT_DIR / "drug_enzymes.jsonl", "wb")
transporters_file = open(OUTPUT_DIR / "drug_transporters.jsonl", "wb")
targets_file = open(OUTPUT_DIR / "drug_targets.jsonl", "wb")
text_file = open(OUTPUT_DIR / "drug_text.jsonl", "wb")


def write_jsonl(file_obj, data):
    file_obj.write(orjson.dumps(data))
    file_obj.write(b"\n")


context = ET.iterparse(XML_PATH, events=("end",))

for event, elem in tqdm(context):

    if elem.tag.endswith("drug"):

        try:
            drugbank_id = None

            id_elem = elem.find("db:drugbank-id[@primary='true']", NS)
            if id_elem is not None:
                drugbank_id = id_elem.text

            name_elem = elem.find("db:name", NS)
            drug_name = name_elem.text if name_elem is not None else None

            if not drugbank_id:
                elem.clear()
                continue

            # -------------------------
            # BASIC DRUG INFO
            # -------------------------

            write_jsonl(drugs_file, {
                "drugbank_id": drugbank_id,
                "name": drug_name
            })

            # -------------------------
            # TEXT SECTIONS
            # -------------------------

            def get_text(tag):
                t = elem.find(f"db:{tag}", NS)
                return t.text if t is not None else ""

            write_jsonl(text_file, {
                "drugbank_id": drugbank_id,
                "name": drug_name,
                "description": get_text("description"),
                "pharmacodynamics": get_text("pharmacodynamics"),
                "toxicity": get_text("toxicity"),
                "indication": get_text("indication")
            })

            # -------------------------
            # ENZYMES
            # -------------------------

            enzymes = elem.findall("db:enzymes/db:enzyme", NS)

            for enzyme in enzymes:

                enzyme_name = enzyme.find("db:name", NS)

                actions = [
                    a.text
                    for a in enzyme.findall("db:actions/db:action", NS)
                    if a.text
                ]

                write_jsonl(enzymes_file, {
                    "drug": drugbank_id,
                    "enzyme": enzyme_name.text if enzyme_name is not None else None,
                    "actions": actions
                })

            # -------------------------
            # TRANSPORTERS
            # -------------------------

            transporters = elem.findall(
                "db:transporters/db:transporter",
                NS
            )

            for transporter in transporters:

                transporter_name = transporter.find("db:name", NS)

                actions = [
                    a.text
                    for a in transporter.findall("db:actions/db:action", NS)
                    if a.text
                ]

                write_jsonl(transporters_file, {
                    "drug": drugbank_id,
                    "transporter": (
                        transporter_name.text
                        if transporter_name is not None else None
                    ),
                    "actions": actions
                })

            # -------------------------
            # TARGETS
            # -------------------------

            targets = elem.findall("db:targets/db:target", NS)

            for target in targets:

                target_name = target.find("db:name", NS)

                write_jsonl(targets_file, {
                    "drug": drugbank_id,
                    "target": (
                        target_name.text
                        if target_name is not None else None
                    )
                })

        except Exception as e:
            print("ERROR:", e)

        finally:
            elem.clear()

drugs_file.close()
enzymes_file.close()
transporters_file.close()
targets_file.close()
text_file.close()

print("DrugBank parsing complete.")