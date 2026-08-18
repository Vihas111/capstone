"""
CLI wrapper around scripts/mechanism_thresholds.py's tune_thresholds().
Computes per-label thresholds from the VAL split's predictions (never test
-- avoid leaking test info into threshold selection) and saves them.

Two distinct uses in this pipeline (see scripts/mechanism_thresholds.py's
docstring for why the objectives differ):
  --objective f1              -> the DEPLOYED thresholds used by
                                  scripts/mechanism_lookup.py --predict.
  --objective precision-floor -> the GATE for accepting self-training
                                  pseudo-labels (stricter: a wrong guess
                                  here becomes a hard-coded fake label).

Usage:
    python scripts/tune_mechanism_thresholds.py --objective f1 \
        --checkpoint checkpoints/mechanism_predictor.pt --hidden-dims \
        --out data/processed/mechanism_label_thresholds.json

    python scripts/tune_mechanism_thresholds.py --objective precision-floor \
        --precision-floor 0.5 --checkpoint checkpoints/mechanism_predictor.pt \
        --hidden-dims --out data/processed/mechanism_label_thresholds_bootstrap.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.mechanism_mlp import MechanismMLP
from scripts.mechanism_thresholds import tune_thresholds
from scripts.train_mechanism_predictor import build_xy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/mechanism_predictor.pt")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[])
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--split-path", default="data/processed/mechanism_drug_split.json")
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--objective", choices=["f1", "precision-floor"], default="f1")
    p.add_argument("--precision-floor", type=float, default=0.5)
    p.add_argument("--min-support", type=int, default=10)
    p.add_argument("--out", default="data/processed/mechanism_label_thresholds.json")
    return p.parse_args()


def main():

    args = parse_args()

    with open(args.vocab_path) as f:
        label_vocab = json.load(f)["labels"]

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids, label_matrix = dataset["drug_ids"], dataset["labels"]
    label_by_drug = dict(zip(drug_ids, label_matrix))

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    with open(args.split_path) as f:
        split = json.load(f)

    x_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, split["val"])

    model = MechanismMLP(
        in_dim=x_val.shape[1], hidden_dims=tuple(args.hidden_dims),
        out_dim=len(label_vocab), dropout=args.dropout,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    model.eval()

    with torch.no_grad():
        y_score = torch.sigmoid(model(x_val.to(DEVICE))).cpu()

    objective = "f1" if args.objective == "f1" else "precision_floor"
    thresholds, tiers = tune_thresholds(
        y_val, y_score, label_vocab,
        objective=objective, min_support=args.min_support,
        precision_floor=args.precision_floor,
    )

    n_per_label = sum(1 for t in tiers if t == "per-label")
    n_never_fires = int((thresholds > 1.0).sum())
    print(f"Tuned {len(label_vocab)} thresholds from {x_val.shape[0]} val drugs "
          f"(objective={args.objective}).")
    print(f"  {n_per_label}/{len(label_vocab)} got a real per-label threshold "
          f"(val positives >= {args.min_support}), rest fell back to per-kind pooled.")
    print(f"  {n_never_fires}/{len(label_vocab)} labels never fire (threshold > 1.0 -- "
          f"no signal met the objective for that label's kind pool).")

    out = {
        "objective": args.objective,
        "min_support": args.min_support,
        "precision_floor": args.precision_floor if args.objective == "precision-floor" else None,
        "checkpoint": args.checkpoint,
        "label_vocab": label_vocab,
        "thresholds": thresholds.tolist(),
        "tiers": tiers,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
