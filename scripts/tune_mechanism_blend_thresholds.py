"""
Same job as scripts/tune_mechanism_thresholds.py --objective f1, but tunes
against the k-NN/linear BLENDED probability distribution
(scripts/mechanism_knn_transfer.py + averaging with the linear model),
since findings/findings.md's "Item 2 result" showed that blend beats the
linear model alone in 5/5 folds and is now the default for
scripts/mechanism_lookup.py --predict. The existing per-label thresholds
(data/processed/mechanism_label_thresholds.json) were fit to the LINEAR
model's own output distribution specifically -- reusing them for a
differently-shaped blended distribution would silently miscalibrate
--predict's confidence cutoffs, so this is a separate, blend-specific file
rather than overwriting the original.

k-NN pool is the main train split ONLY (data/processed/mechanism_drug_split.json's
"train" key, 3103 drugs) -- same population the linear model itself was
trained on, and never val/test, for the same reason the linear model
doesn't get retrained on val/test after evaluation: val/test are meant to
stay a permanently honest "drugs never used to build this" measurement.

Usage:
    python scripts/tune_mechanism_blend_thresholds.py \
        --out data/processed/mechanism_label_thresholds_blend.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.mechanism_mlp import MechanismMLP
from scripts.mechanism_knn_transfer import knn_transfer_probs
from scripts.mechanism_thresholds import tune_thresholds
from scripts.train_mechanism_predictor import build_xy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/mechanism_predictor.pt")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[])
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--split-path", default="data/processed/mechanism_drug_split.json")
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--knn-k", type=int, default=40,
                    help="Matches the K chosen by scripts/run_mechanism_knn_ensemble_kfold.py's "
                    "fold-0 val sweep.")
    p.add_argument("--min-support", type=int, default=10)
    p.add_argument("--out", default="data/processed/mechanism_label_thresholds_blend.json")
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

    x_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, split["train"])
    x_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, split["val"])

    model = MechanismMLP(
        in_dim=x_val.shape[1], hidden_dims=tuple(args.hidden_dims),
        out_dim=len(label_vocab),
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    model.eval()

    with torch.no_grad():
        linear_probs = torch.sigmoid(model(x_val.to(DEVICE))).cpu()

    knn_probs = knn_transfer_probs(x_val, x_train, y_train, k=args.knn_k)
    blend_probs = (linear_probs + knn_probs) / 2

    thresholds, tiers = tune_thresholds(
        y_val, blend_probs, label_vocab, objective="f1", min_support=args.min_support,
    )

    n_per_label = sum(1 for t in tiers if t == "per-label")
    n_never_fires = int((thresholds > 1.0).sum())
    print(f"Tuned {len(label_vocab)} blend thresholds from {x_val.shape[0]} val drugs.")
    print(f"  {n_per_label}/{len(label_vocab)} got a real per-label threshold "
          f"(val positives >= {args.min_support}), rest fell back to per-kind pooled.")
    print(f"  {n_never_fires}/{len(label_vocab)} labels never fire.")

    out = {
        "objective": "f1",
        "min_support": args.min_support,
        "precision_floor": None,
        "checkpoint": args.checkpoint,
        "knn_k": args.knn_k,
        "blend": "average(linear, knn_transfer)",
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
