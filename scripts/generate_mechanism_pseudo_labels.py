"""
Self-training step 1: run the current best mechanism-predictor model over
every fingerprint-bearing drug EXCLUDED from data/processed/mechanism_dataset.pt
(didn't clear --min-freq 10 in scripts/build_mechanism_labels.py), gate
predictions by the bootstrap precision-floor thresholds
(scripts/tune_mechanism_thresholds.py --objective precision-floor), and keep
only drugs where >=1 label clears the gate as pseudo-labeled training data.

ONE batched forward pass over data/processed/drug_fingerprints.pt directly
-- deliberately NOT scripts/mechanism_lookup.py's per-drug SMILES->RDKit
path, which would be redundant re-computation of fingerprints already
sitting in that tensor.

Expect a modest yield (low hundreds of drugs, not thousands) -- the
precision-floor gate is intentionally strict (a wrong pseudo-label becomes
a hard-coded fake label for training), and findings/findings.md's own
calibration analysis showed even the model's most confident bucket only
reaches ~46% aggregate precision. Report the real number, don't assume.

Usage:
    python scripts/generate_mechanism_pseudo_labels.py \
        --checkpoint checkpoints/mechanism_predictor.pt --hidden-dims \
        --thresholds-path data/processed/mechanism_label_thresholds_bootstrap.json \
        --out data/processed/mechanism_pseudo_labels.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from models.mechanism_mlp import MechanismMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/mechanism_predictor.pt")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[])
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--thresholds-path", default="data/processed/mechanism_label_thresholds_bootstrap.json")
    p.add_argument("--min-labels-per-drug", type=int, default=1)
    p.add_argument("--out", default="data/processed/mechanism_pseudo_labels.pt")
    p.add_argument("--report-out", default="data/processed/mechanism_pseudo_labels_report.json")
    return p.parse_args()


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    real_drug_ids = set(dataset["drug_ids"])

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    all_ids, all_fps = fp_data["drug_ids"], fp_data["fingerprints"]

    pool_ids = [d for d in all_ids if d not in real_drug_ids]
    pool_idx = [i for i, d in enumerate(all_ids) if d not in real_drug_ids]
    pool_fps = all_fps[pool_idx]

    print(f"Pseudo-label candidate pool: {len(pool_ids)} drugs "
          f"(fingerprint-bearing, excluded from the {len(real_drug_ids)}-drug real training set).")

    with open(args.thresholds_path) as f:
        th_data = json.load(f)
    label_vocab = th_data["label_vocab"]
    thresholds = torch.tensor(th_data["thresholds"])

    model = MechanismMLP(
        in_dim=pool_fps.shape[1], hidden_dims=tuple(args.hidden_dims),
        out_dim=len(label_vocab), dropout=args.dropout,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(pool_fps.to(DEVICE))).cpu()

    accepted = probs >= thresholds.unsqueeze(0)
    n_labels_per_drug = accepted.sum(dim=1)
    keep_mask = n_labels_per_drug >= args.min_labels_per_drug

    kept_ids = [pool_ids[i] for i in range(len(pool_ids)) if keep_mask[i]]
    kept_labels = accepted[keep_mask].float()

    print(f"Accepted {len(kept_ids)}/{len(pool_ids)} pool drugs "
          f"(>= {args.min_labels_per_drug} label(s) cleared the precision-floor gate).")

    if kept_ids:
        mean_labels = kept_labels.sum(dim=1).mean().item()
        print(f"Mean pseudo-labels per accepted drug: {mean_labels:.2f}")

    torch.save({"drug_ids": kept_ids, "labels": kept_labels}, args.out)

    report = {
        "pool_size": len(pool_ids),
        "n_accepted": len(kept_ids),
        "min_labels_per_drug": args.min_labels_per_drug,
        "thresholds_source": args.thresholds_path,
        "mean_labels_per_accepted_drug": kept_labels.sum(dim=1).mean().item() if kept_ids else 0.0,
    }
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved: {args.out}")
    print(f"Saved: {args.report_out}")


if __name__ == "__main__":
    main()
