"""
Trains MechanismMLP (models/mechanism_mlp.py) to predict a drug's likely
mechanism-label profile from its fingerprint alone -- the gap-filler for
scripts/mechanism_lookup.py's --predict flag.

Deliberately uses a plain in-memory TensorDataset/DataLoader, NOT
data_utils/shard_dataset_v2.py's shard-streaming machinery: the whole
dataset here is ~3,880 x 1024 floats (~16MB), trivially fits in memory, so
sharding would be pure overhead. This is a deliberate deviation from
scripts/train_v2.py's conventions, not an oversight -- everything else
(argparse CLI shape, run_epoch, checkpoint-on-best-val-macro-AP, JSON
history every epoch, --patience early stopping, scripts/losses.py +
scripts/metrics.py reuse) mirrors train_v2.py exactly.

prior_logits (final-layer bias init to each label's empirical base rate --
see scripts/build_label_priors.py for the original rationale, same base-rate
-collapse risk applies here given median-2-positives/drug sparsity) is
computed from the TRAIN split only, not the full dataset -- an intentional
tightening vs. build_label_priors.py's whole-dataset scan, since here it's
cheap to do correctly and avoids leaking val/test statistics into model init.

Usage:
    python scripts/train_mechanism_predictor.py --epochs 200 --patience 20
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 128],
                    help="Pass with no values (--hidden-dims) for a plain "
                    "linear model (no hidden layers) -- see evaluate_mechanism_"
                    "predictor.py's --baseline linear comparison.")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--no-pos-weight", action="store_true")
    p.add_argument("--no-prior-logits", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--split-path", default="data/processed/mechanism_drug_split.json")
    p.add_argument("--pos-weight-path", default="data/processed/mechanism_pos_weight.pt")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint-name", default="mechanism_predictor.pt")
    p.add_argument("--history-name", default="mechanism_training_history.json")
    return p.parse_args()


def build_xy(drug_ids, fp_by_drug, label_by_drug, split_ids):
    keep = [d for d in split_ids if d in label_by_drug]
    x = torch.stack([fp_by_drug[d] for d in keep])
    y = torch.stack([label_by_drug[d] for d in keep])
    return x, y


def run_epoch(model, loader, criterion, optimizer=None):

    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    all_targets = []
    all_probs = []
    total_loss = 0.0
    steps = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()

    with context:

        for x, y in loader:

            x, y = x.to(DEVICE), y.to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            steps += 1

            all_targets.append(y.detach().cpu())
            all_probs.append(torch.sigmoid(logits).detach().cpu())

    y_true = torch.cat(all_targets)
    y_score = torch.cat(all_probs)

    macro_ap, micro_ap = macro_and_micro_ap(y_true, y_score)

    return {"loss": total_loss / max(steps, 1), "macro_ap": macro_ap, "micro_ap": micro_ap}


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

    x_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, split["train"])
    x_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, split["val"])

    print(f"Train: {x_train.shape[0]} drugs, Val: {x_val.shape[0]} drugs, "
          f"{num_labels} labels")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False
    )

    pos_weight = None
    if not args.no_pos_weight and Path(args.pos_weight_path).exists():
        pos_weight = torch.load(args.pos_weight_path, weights_only=False).to(DEVICE)
        print(f"Loaded pos_weight from {args.pos_weight_path}")

    prior_logits = None
    if not args.no_prior_logits:
        p = (y_train.sum(dim=0) + 1) / (y_train.shape[0] + 2)
        prior_logits = torch.log(p / (1 - p))
        print(f"Computed prior_logits from train split "
              f"(range {prior_logits.min():.3f} to {prior_logits.max():.3f})")

    model = MechanismMLP(
        in_dim=x_train.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        out_dim=num_labels,
        dropout=args.dropout,
        prior_logits=prior_logits,
    ).to(DEVICE)

    criterion = build_criterion(args.loss, pos_weight=pos_weight, focal_gamma=args.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_macro_ap = -1.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):

        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train loss {train_metrics['loss']:.4f} macro-AP {train_metrics['macro_ap']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} macro-AP {val_metrics['macro_ap']:.4f} "
            f"micro-AP {val_metrics['micro_ap']:.4f}"
        )

        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        with open(ckpt_dir / args.history_name, "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["macro_ap"] > best_macro_ap:
            best_macro_ap = val_metrics["macro_ap"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / args.checkpoint_name)
            print("  -> saved new best checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}.")
                break

    print(f"\nTraining finished. Best val macro-AP: {best_macro_ap:.4f}")


if __name__ == "__main__":
    main()
