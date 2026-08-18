"""
Builds the label space + per-drug multi-hot label vectors for the mechanism
gap-filler model, from data/processed/drug_enzymes.jsonl / drug_targets.jsonl
/ drug_transporters.jsonl (drug_carriers.jsonl is excluded -- it has no
`actions` field, so it doesn't fit the "kind:protein:action" scheme below).

Label scheme: "<kind>:<protein>:<action>", e.g.
"enzyme:Cytochrome P450 3A4:inhibitor" -- kind is included (not just
protein:action) so predictions can be routed straight back into
scripts/mechanism_lookup.py's existing enzymes/targets/transporters
structure without ambiguity about which table a protein name came from.

IMPORTANT: population and frequency counts are restricted to drug IDs
already present in --fingerprints-path (NOT the full DrugBank population).
Counting frequency over the full population and only intersecting with
fingerprints afterward gives the wrong --min-freq cutoff, since some of a
label's "support" may come from drugs with no usable structure to train on.
Restricting first is what yields the real ~422-label / ~3,882-drug trainable
pool documented in CLAUDE.md section 3b (vs. an inflated ~443/~6,345 if you
count against the full DrugBank population).

Usage:
    python scripts/build_mechanism_labels.py \
        --fingerprints-path data/processed/drug_fingerprints.pt \
        --min-freq 10 \
        --out data/processed/mechanism_dataset.pt \
        --vocab-out data/processed/mechanism_label_vocab.json \
        --pos-weight-out data/processed/mechanism_pos_weight.pt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import orjson
import torch

DATA_DIR = Path("data/processed")

SOURCE_TABLES = [
    ("enzyme", DATA_DIR / "drug_enzymes.jsonl"),
    ("target", DATA_DIR / "drug_targets.jsonl"),
    ("transporter", DATA_DIR / "drug_transporters.jsonl"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--min-freq", type=int, default=10)
    p.add_argument("--out", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--vocab-out", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--pos-weight-out", default="data/processed/mechanism_pos_weight.pt")
    return p.parse_args()


def load_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def main():

    args = parse_args()

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fingerprint_drugs = set(fp_data["drug_ids"])
    print(f"Restricting to {len(fingerprint_drugs)} drugs with a usable fingerprint.")

    # drug_id -> set of "kind:protein:action" labels
    drug_labels = defaultdict(set)

    for kind, path in SOURCE_TABLES:
        for row in load_jsonl(path):

            drug = row["drug"]
            if drug not in fingerprint_drugs:
                continue

            actions = row.get("actions") or []
            protein = row[kind]

            for action in actions:
                drug_labels[drug].add(f"{kind}:{protein}:{action}")

    print(f"{len(drug_labels)} fingerprint-bearing drugs have >=1 actioned mechanism label.")

    # Frequency of each label, counted only over fingerprint-bearing drugs.
    label_freq = defaultdict(int)
    for labels in drug_labels.values():
        for label in labels:
            label_freq[label] += 1

    vocab = sorted(l for l, freq in label_freq.items() if freq >= args.min_freq)
    label_to_idx = {label: i for i, label in enumerate(vocab)}
    num_labels = len(vocab)

    print(f"{len(label_freq)} distinct labels total; {num_labels} kept at "
          f"--min-freq {args.min_freq}.")

    # Final drug population: fingerprint-bearing drugs with >=1 label in vocab.
    drug_ids = []
    rows = []

    for drug, labels in drug_labels.items():
        kept = [label_to_idx[l] for l in labels if l in label_to_idx]
        if not kept:
            continue
        vec = torch.zeros(num_labels)
        vec[kept] = 1.0
        drug_ids.append(drug)
        rows.append(vec)

    label_matrix = torch.stack(rows) if rows else torch.zeros((0, num_labels))

    print(f"Final trainable population: {len(drug_ids)} drugs x {num_labels} labels.")
    print(f"Mean labels/drug: {label_matrix.sum(dim=1).mean().item():.2f}, "
          f"median: {label_matrix.sum(dim=1).median().item():.0f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"drug_ids": drug_ids, "labels": label_matrix}, args.out)

    with open(args.vocab_out, "w") as f:
        json.dump(
            {"labels": vocab, "min_freq": args.min_freq, "num_labels": num_labels},
            f, indent=2,
        )

    # pos_weight -- same formula as scripts/compute_pos_weight.py.
    positives = label_matrix.sum(dim=0)
    negatives = len(drug_ids) - positives
    pos_weight = torch.clamp(negatives / (positives + 1.0), min=1.0, max=100.0)
    torch.save(pos_weight, args.pos_weight_out)

    print(f"\nSaved:")
    print(f"  {args.out}")
    print(f"  {args.vocab_out}")
    print(f"  {args.pos_weight_out}")


if __name__ == "__main__":
    main()
