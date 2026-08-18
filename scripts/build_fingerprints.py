"""
Computes a Morgan/ECFP molecular fingerprint per drug from
data/processed/drug_smiles.jsonl (built by scripts/extract_smiles.py). This
is the input feature for the mechanism gap-filler model
(models/mechanism_mlp.py) -- structure in, likely mechanism labels out.

Design choices (see CLAUDE.md section 3b for the reasoning):
  - Uses the newer rdFingerprintGenerator API, not the older
    AllChem.GetMorganFingerprintAsBitVect (being deprecated upstream).
  - includeChirality=True (NOT the default False) -- otherwise stereoisomer
    pairs (pharmacologically distinct, separate DrugBank entries) collapse
    onto identical fingerprints.
  - Multi-fragment SMILES (salts, e.g. "CC(=O)O.[Na]") are fingerprinted as
    the whole disconnected mol, not reduced to the largest fragment. Simpler,
    and DrugBank's calculated SMILES are for the specific salt/form DrugBank
    documents -- reducing to the largest fragment would throw that away.
  - Failures (unparseable SMILES, sanitization errors) are collected and
    reported, not silently dropped -- consistent with this project's general
    habit (see scripts/mechanism_lookup.py) of surfacing gaps rather than
    hiding them.

Usage:
    python scripts/build_fingerprints.py \
        --smiles-path data/processed/drug_smiles.jsonl \
        --radius 2 --n-bits 1024 --chirality \
        --out data/processed/drug_fingerprints.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.error")  # we check MolFromSmiles() -> None ourselves


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles-path", default="data/processed/drug_smiles.jsonl")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=1024)
    p.add_argument("--chirality", action="store_true", default=True)
    p.add_argument("--out", default="data/processed/drug_fingerprints.pt")
    p.add_argument(
        "--failed-out", default="data/processed/drug_fingerprints_failed.json"
    )
    return p.parse_args()


def main():

    args = parse_args()

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius, fpSize=args.n_bits, includeChirality=args.chirality
    )

    drug_ids = []
    fingerprints = []
    failed = {}

    with open(args.smiles_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Computing fingerprints for {len(rows)} drugs...")

    for i, row in enumerate(rows):

        drugbank_id = row["drugbank_id"]
        smiles = row["smiles"]

        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                failed[drugbank_id] = "MolFromSmiles returned None (unparseable SMILES)"
                continue

            fp = gen.GetFingerprint(mol)
            arr = np.zeros((args.n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)

        except Exception as e:
            failed[drugbank_id] = f"{type(e).__name__}: {e}"
            continue

        drug_ids.append(drugbank_id)
        fingerprints.append(arr)

        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(rows)} processed")

    fingerprints = torch.tensor(np.stack(fingerprints), dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "drug_ids": drug_ids,
            "fingerprints": fingerprints,
            "radius": args.radius,
            "n_bits": args.n_bits,
            "chirality": args.chirality,
        },
        out_path,
    )

    with open(args.failed_out, "w") as f:
        json.dump(failed, f, indent=2)

    print(
        f"\nSucceeded: {len(drug_ids)}/{len(rows)}. Failed: {len(failed)} "
        f"(see {args.failed_out})."
    )
    print(f"Fingerprint tensor shape: {tuple(fingerprints.shape)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
