"""
Computes a fixed set of RDKit physicochemical descriptors per drug from
data/processed/drug_smiles.jsonl (same source scripts/build_fingerprints.py
reads) -- tested because findings/findings.md's ORIGINAL hypothesis (before
the ChemBERTa detour, see scripts/run_mechanism_embedding_kfold.py) was that
transporter/PK substrate recognition may depend more on global physicochemical
properties (size, polarity, H-bonding) than the LOCAL 2D substructure
patterns Morgan/ECFP fingerprints encode -- e.g. P-gp substrate recognition
is well-documented in the pharmacology literature to correlate with exactly
these properties. That hypothesis was never actually tested; ChemBERTa
tested a different fix (pretrained representation) instead and didn't help.

A fixed, named, interpretable descriptor set (not RDKit's full ~217-descriptor
"kitchen sink" via Descriptors.CalcMolDescriptors) -- ~25 well-established
ADMET/QSAR descriptors, low-dimensional by design so there's no meaningful
overfitting risk even concatenated onto the 1024-bit fingerprint at this
project's ~3,100-training-drug scale.

NOTE: descriptors are NOT normalized here (raw RDKit values, wildly
different scales -- MolWt ~100-1000, FractionCSP3 ~0-1). Normalization
(z-score, train-split statistics only) happens at train/eval time in
scripts/run_mechanism_physchem_kfold.py, per-fold, to avoid baking
leakage-prone global statistics into a shared artifact.

Usage:
    python scripts/build_physchem_descriptors.py \
        --smiles-path data/processed/drug_smiles.jsonl \
        --out data/processed/drug_physchem_descriptors.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.error")

# Fixed, named descriptor list -- order matters (defines the output tensor's
# column order, recorded in the saved artifact's "descriptor_names").
DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "NumSaturatedRings": Lipinski.NumSaturatedRings,
    "RingCount": Lipinski.RingCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumSaturatedHeterocycles": Lipinski.NumSaturatedHeterocycles,
    "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "NumStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters,
    "FormalCharge": Chem.GetFormalCharge,
}
DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCS.keys())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles-path", default="data/processed/drug_smiles.jsonl")
    p.add_argument("--out", default="data/processed/drug_physchem_descriptors.pt")
    p.add_argument("--failed-out", default="data/processed/drug_physchem_descriptors_failed.json")
    return p.parse_args()


def main():

    args = parse_args()

    with open(args.smiles_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Computing {len(DESCRIPTOR_NAMES)} descriptors for {len(rows)} drugs...")

    drug_ids = []
    descriptors = []
    failed = {}

    for i, row in enumerate(rows):

        drugbank_id = row["drugbank_id"]
        smiles = row["smiles"]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed[drugbank_id] = "MolFromSmiles returned None (unparseable SMILES)"
                continue

            values = [DESCRIPTOR_FUNCS[name](mol) for name in DESCRIPTOR_NAMES]
            arr = np.array(values, dtype=np.float32)

            if not np.all(np.isfinite(arr)):
                failed[drugbank_id] = f"non-finite descriptor value(s): {arr.tolist()}"
                continue

        except Exception as e:
            failed[drugbank_id] = f"{type(e).__name__}: {e}"
            continue

        drug_ids.append(drugbank_id)
        descriptors.append(arr)

        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(rows)} processed")

    descriptors = torch.tensor(np.stack(descriptors), dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "drug_ids": drug_ids,
            "descriptors": descriptors,
            "descriptor_names": DESCRIPTOR_NAMES,
        },
        out_path,
    )

    with open(args.failed_out, "w") as f:
        json.dump(failed, f, indent=2)

    print(f"\nSucceeded: {len(drug_ids)}/{len(rows)}. Failed: {len(failed)} (see {args.failed_out}).")
    print(f"Descriptor tensor shape: {tuple(descriptors.shape)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
