"""
Computes a pretrained ChemBERTa embedding per drug from
data/processed/drug_smiles.jsonl (same source scripts/build_fingerprints.py
reads) -- an alternative structure representation for the mechanism
gap-filler, tried because findings/findings.md's "Item 1 diagnosis"
established the model's weakness is a REPRESENTATION problem, not a
data-volume one: raw Morgan/ECFP bits (this project's only representation
so far) correlate well with target/PD binding but poorly with enzyme/
transporter/PK recognition, and within-kind more training data barely
moves AP. A pretrained encoder (seyonec/ChemBERTa-zinc-base-v1, a RoBERTa
model trained on ~250k ZINC15 SMILES) carries structure-activity signal
learned from a much larger and more diverse chemical space than this
project's ~3,880-drug labeled population, which is the actual point of
trying it: better generalization to genuinely novel drugs/scaffolds that
don't resemble anything already in DrugBank, not just a better score on
drugs already similar to the training pool (which is what the existing
Tanimoto k-NN ensemble already covers well).

Embedding = masked mean-pool over the last hidden layer's token
embeddings (standard choice absent a task-specific pooling head; the
model's own pooler_output is trained for next-sentence-prediction-style
objectives, not molecule-level similarity, so mean-pooling is preferred
here, consistent with how ChemBERTa is used in the property-prediction
literature).

Usage:
    python scripts/build_chemberta_embeddings.py \
        --smiles-path data/processed/drug_smiles.jsonl \
        --out data/processed/drug_chemberta_embeddings.pt
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles-path", default="data/processed/drug_smiles.jsonl")
    p.add_argument("--model-name", default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--out", default="data/processed/drug_chemberta_embeddings.pt")
    p.add_argument("--failed-out", default="data/processed/drug_chemberta_embeddings_failed.json")
    return p.parse_args()


def mean_pool(last_hidden_state, attention_mask):
    """[B, T, H], [B, T] -> [B, H], excluding padding tokens from the mean."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-8)
    return summed / counts


def main():

    args = parse_args()

    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    with open(args.smiles_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Embedding {len(rows)} drugs (batch_size={args.batch_size})...")

    drug_ids = []
    embeddings = []
    failed = {}

    for start in range(0, len(rows), args.batch_size):

        batch = rows[start:start + args.batch_size]
        smiles_list = [r["smiles"] for r in batch]

        try:
            inputs = tokenizer(
                smiles_list, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
            batch_embeddings = mean_pool(out.last_hidden_state, inputs["attention_mask"]).cpu()
        except Exception as e:
            for r in batch:
                failed[r["drugbank_id"]] = f"{type(e).__name__}: {e}"
            continue

        for r, emb in zip(batch, batch_embeddings):
            drug_ids.append(r["drugbank_id"])
            embeddings.append(emb)

        if (start // args.batch_size) % 20 == 0:
            print(f"  ...{start + len(batch)}/{len(rows)} processed")

    embeddings = torch.stack(embeddings)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "drug_ids": drug_ids,
            "embeddings": embeddings,
            "model_name": args.model_name,
            "pooling": "masked_mean",
        },
        out_path,
    )

    with open(args.failed_out, "w") as f:
        json.dump(failed, f, indent=2)

    print(f"\nSucceeded: {len(drug_ids)}/{len(rows)}. Failed: {len(failed)} (see {args.failed_out}).")
    print(f"Embedding tensor shape: {tuple(embeddings.shape)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
