import xml.etree.ElementTree as ET
import orjson
from pathlib import Path
from tqdm import tqdm

XML_PATH = Path("data/raw/drugdatabase.xml")

OUTPUT_PATH = Path(
    "data/processed/drugbank_external_ids.jsonl"
)

NS = {"db": "http://www.drugbank.ca"}

output_file = open(OUTPUT_PATH, "wb")


def write_jsonl(file_obj, data):
    file_obj.write(orjson.dumps(data))
    file_obj.write(b"\n")


context = ET.iterparse(XML_PATH, events=("end",))

for event, elem in tqdm(context):

    if elem.tag.endswith("drug"):

        try:

            # -------------------------
            # BASIC INFO
            # -------------------------

            id_elem = elem.find(
                "db:drugbank-id[@primary='true']",
                NS
            )

            if id_elem is None:
                elem.clear()
                continue

            drugbank_id = id_elem.text

            name_elem = elem.find("db:name", NS)

            drug_name = (
                name_elem.text
                if name_elem is not None
                else None
            )

            # -------------------------
            # RXNORM EXTRACTION
            # -------------------------

            rxnorm_id = None

            external_ids = elem.findall(
                "db:external-identifiers/db:external-identifier",
                NS
            )

            for ext in external_ids:

                resource = ext.find("db:resource", NS)
                identifier = ext.find("db:identifier", NS)

                if (
                    resource is not None
                    and identifier is not None
                ):

                    if resource.text == "RxCUI":

                        rxnorm_id = identifier.text
                        break

            # -------------------------
            # SYNONYMS
            # -------------------------

            synonyms = []

            synonym_elems = elem.findall(
                "db:synonyms/db:synonym",
                NS
            )

            for syn in synonym_elems:

                if syn.text:
                    synonyms.append(syn.text)

            # -------------------------
            # WRITE OUTPUT
            # -------------------------

            write_jsonl(output_file, {
                "drugbank_id": drugbank_id,
                "name": drug_name,
                "rxnorm": rxnorm_id,
                "synonyms": synonyms
            })

        except Exception as e:
            print("ERROR:", e)

        finally:
            elem.clear()

output_file.close()

print("\nExternal ID extraction complete.")