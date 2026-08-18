"""
Evaluates the mechanism gap-filler (models/mechanism_mlp.py,
checkpoints/mechanism_predictor.pt) against the held-out test split.

Also supports --baseline linear: trains a single Linear(in_dim -> num_labels)
model (no hidden layers, no prior_logits) on the SAME train split and reports
its test macro/micro-AP alongside the MLP's. At ~3,100 training drugs with
median-2-positives/label sparsity, a well-regularized linear model can
plausibly match a deeper MLP -- this project's own convention
(CLAUDE.md section 5) is to report real, unflattering numbers rather than
assume the fancier model is better, so check rather than assume here too.

Usage:
    python scripts/evaluate_mechanism_predictor.py
    python scripts/evaluate_mechanism_predictor.py --baseline linear
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from models.mechanism_mlp import MechanismMLP
from scripts.losses import build_criterion
from scripts.metrics import macro_and_micro_ap
from scripts.train_mechanism_predictor import build_xy, run_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--checkpoint", default="checkpoints/mechanism_predictor.pt")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--split-path", default="data/processed/mechanism_drug_split.json")
    p.add_argument("--pos-weight-path", default="data/processed/mechanism_pos_weight.pt")
    p.add_argument("--baseline", choices=["none", "linear"], default="none")
    p.add_argument("--baseline-epochs", type=int, default=100)
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--per-kind", action="store_true",
                    help="Also report macro-AP separately for enzyme/target/"
                    "transporter labels (via scripts/metrics.py's class_mask).")
    p.add_argument("--metrics-out", default=None,
                    help="Dump {split, loss, macro_ap, micro_ap, n_samples[, per_kind]} "
                    "as JSON to this path, for programmatic collection (e.g. k-fold).")
    return p.parse_args()


def evaluate(model, x, y, batch_size=128, kind_masks=None):
    """kind_masks: optional dict[str, BoolTensor[C]] (e.g. {"enzyme": ...,
    "target": ..., "transporter": ...}) -- if given, also computes macro/
    micro-AP restricted to each kind's columns, added under metrics["per_kind"]."""

    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    criterion = build_criterion("bce")

    all_targets, all_probs, total_loss, steps = [], [], 0.0, 0
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            steps += 1
            all_targets.append(yb.cpu())
            all_probs.append(torch.sigmoid(logits).cpu())

    y_true = torch.cat(all_targets)
    y_score = torch.cat(all_probs)
    macro_ap, micro_ap = macro_and_micro_ap(y_true, y_score)

    metrics = {"loss": total_loss / max(steps, 1), "macro_ap": macro_ap, "micro_ap": micro_ap}

    if kind_masks:
        metrics["per_kind"] = {}
        for kind, mask in kind_masks.items():
            k_macro, k_micro = macro_and_micro_ap(y_true, y_score, class_mask=mask)
            metrics["per_kind"][kind] = {"macro_ap": k_macro, "micro_ap": k_micro, "n_labels": int(mask.sum())}

    return metrics


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids, label_matrix = dataset["drug_ids"], dataset["labels"]
    label_by_drug = dict(zip(drug_ids, label_matrix))

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    with open(args.split_path) as f:
        split = json.load(f)

    num_labels = label_matrix.shape[1]

    x_eval, y_eval = build_xy(drug_ids, fp_by_drug, label_by_drug, split[args.split])
    print(f"Evaluating on {args.split}: {x_eval.shape[0]} drugs, {num_labels} labels")

    kind_masks = None
    if args.per_kind:
        with open(args.vocab_path) as f:
            vocab = json.load(f)["labels"]
        kinds = [label.split(":", 1)[0] for label in vocab]
        kind_masks = {
            kind: torch.tensor([k == kind for k in kinds], dtype=torch.bool)
            for kind in ("enzyme", "target", "transporter")
        }

    model = MechanismMLP(
        in_dim=x_eval.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        out_dim=num_labels,
        dropout=args.dropout,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=False))

    x_eval_dev = x_eval.to(DEVICE)
    y_eval_dev = y_eval.to(DEVICE)
    metrics = evaluate(model, x_eval_dev, y_eval_dev, kind_masks=kind_masks)

    print(f"\nMechanismMLP -- {args.split} loss: {metrics['loss']:.4f} | "
          f"macro-AP: {metrics['macro_ap']:.4f} | micro-AP: {metrics['micro_ap']:.4f}")
    print(f"(n_samples={x_eval.shape[0]}, n_classes={num_labels})")
    if kind_masks:
        for kind, m in metrics["per_kind"].items():
            print(f"  [{kind:12s}] n_labels={m['n_labels']:3d}  macro-AP: {m['macro_ap']:.4f}  micro-AP: {m['micro_ap']:.4f}")

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({"split": args.split, "n_samples": x_eval.shape[0], **metrics}, f, indent=2)

    if args.baseline == "linear":

        x_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, split["train"])

        pos_weight = None
        if Path(args.pos_weight_path).exists():
            pos_weight = torch.load(args.pos_weight_path, weights_only=False).to(DEVICE)

        linear_model = MechanismMLP(
            in_dim=x_train.shape[1], hidden_dims=(), out_dim=num_labels, dropout=0.0
        ).to(DEVICE)
        criterion = build_criterion("bce", pos_weight=pos_weight)
        optimizer = AdamW(linear_model.parameters(), lr=1e-3, weight_decay=1e-3)

        train_loader = DataLoader(
            TensorDataset(x_train.to(DEVICE), y_train.to(DEVICE)), batch_size=128, shuffle=True
        )

        print(f"\nTraining linear baseline ({args.baseline_epochs} epochs)...")
        for _ in range(args.baseline_epochs):
            run_epoch(linear_model, train_loader, criterion, optimizer)

        baseline_metrics = evaluate(linear_model, x_eval_dev, y_eval_dev)
        print(f"\nLinear baseline -- {args.split} loss: {baseline_metrics['loss']:.4f} | "
              f"macro-AP: {baseline_metrics['macro_ap']:.4f} | "
              f"micro-AP: {baseline_metrics['micro_ap']:.4f}")


if __name__ == "__main__":
    main()
