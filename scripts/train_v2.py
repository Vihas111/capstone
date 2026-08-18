"""
Enhanced training script, replacing train_full.py, that wires in every fix
described in models/rgcn_v2.py, data_utils/shard_dataset_v2.py, and
scripts/losses.py / metrics.py:

  1. Real per-node-type embedding sizes (scripts/build_vocab_sizes.py)
     instead of the baseline's hardcoded 5000-slot table that silently
     collided ~19,830 drugs down to 5000 buckets.
  2. Cold-start-by-drug split (scripts/build_drug_split.py) instead of
     create_splits.py's random-by-shard split, which leaked drug identity
     between train/val/test.
  3. A dataset that doesn't 8x-duplicate itself under multi-worker loading
     and actually shuffles.
  4. pos_weight (already computed by compute_pos_weight.py, previously never
     loaded by any training script) and/or focal loss for the class imbalance.
  5. Macro AND micro AUPR every epoch, computed in a vectorized way so it's
     not the >100s/epoch sklearn bottleneck CLAUDE.md documents. Checkpoints
     on macro-AP (the metric that actually reflects long-tail performance),
     not micro-AP.

Run from the Capstone/ directory:
    python scripts/train_v2.py --epochs 30 --batch-size 128 --loss focal

Prerequisites (one-time, run once against your real data/tensorized/ shards):
    python scripts/build_vocab_sizes.py
    python scripts/build_drug_split.py
    python scripts/compute_pos_weight.py   # already existed; unchanged
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from data_utils.shard_dataset_v2 import ShardDatasetV2
from models.rgcn_v2 import RGCNv2
from scripts.losses import build_criterion
from scripts.metrics import macro_and_micro_ap


NUM_ADE_CLASSES = 4486
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--head", choices=["mean", "bilinear"], default="bilinear")
    p.add_argument("--loss", choices=["bce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--no-pos-weight", action="store_true")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--shard-list-dir",
        default="data/splits",
        help="Directory containing train.txt/val.txt listing ALL relevant "
        "shard paths -- cold-start filtering (which graphs actually belong "
        "to train vs val) is applied on top of these via drug_split.json, "
        "so it's fine (in fact expected) for these lists to overlap.",
    )
    p.add_argument("--vocab-sizes", default="data/processed/vocab_sizes.json")
    p.add_argument("--drug-split", default="data/processed/drug_split.json")
    p.add_argument("--pos-weight-path", default="data/processed/pos_weight.pt")
    p.add_argument("--global-metadata", default="data/processed/global_metadata.pt")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument(
        "--label-mask",
        default=None,
        help="Path to a label_mask.json from scripts/build_label_mask.py -- "
        "restricts macro-AP (and best-checkpoint selection) to classes with "
        "enough dataset-wide positive examples to be learnable. The model "
        "still outputs all classes; this only changes what counts toward "
        "the reported/selected metric.",
    )
    p.add_argument(
        "--label-priors",
        default=None,
        help="Path to label_priors.pt from scripts/build_label_priors.py -- "
        "initializes the classifier's final bias to each class's empirical "
        "base rate instead of zero. Addresses base-rate collapse (confirmed "
        "via prediction-variance check: trained model was outputting nearly "
        "identical predictions regardless of input).",
    )
    return p.parse_args()


def build_loader(args, split_name, drug_split, shuffle):

    shard_list_file = Path(args.shard_list_dir) / f"{split_name}.txt"

    dataset = ShardDatasetV2(
        shard_list_file,
        drug_split=drug_split,
        split_name=split_name,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def run_epoch(model, loader, criterion, optimizer=None, class_mask=None):

    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    all_targets = []
    all_probs = []
    total_loss = 0.0
    steps = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()

    with context:

        for batch in loader:

            batch = batch.to(DEVICE, non_blocking=True)

            if train_mode:
                optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            steps += 1

            all_targets.append(batch.y.detach().cpu())
            all_probs.append(torch.sigmoid(logits).detach().cpu())

    y_true = torch.cat(all_targets)
    y_score = torch.cat(all_probs)

    macro_ap, micro_ap = macro_and_micro_ap(y_true, y_score, class_mask=class_mask)

    return {
        "loss": total_loss / max(steps, 1),
        "macro_ap": macro_ap,
        "micro_ap": micro_ap,
    }


def main():

    args = parse_args()

    with open(args.vocab_sizes) as f:
        vocab_sizes = json.load(f)

    with open(args.drug_split) as f:
        raw_drug_split = json.load(f)
        drug_split = {k: set(v) for k, v in raw_drug_split.items()}

    metadata = torch.load(args.global_metadata, weights_only=False)

    class_mask = None
    if args.label_mask:
        with open(args.label_mask) as f:
            mask_data = json.load(f)
        class_mask = torch.tensor(mask_data["mask"], dtype=torch.bool)
        print(
            f"Loaded label mask: scoring macro-AP over "
            f"{class_mask.sum().item()}/{class_mask.numel()} classes "
            f"(>= {mask_data['min_freq']} positive examples dataset-wide)."
        )

    pos_weight = None
    if not args.no_pos_weight and Path(args.pos_weight_path).exists():
        pos_weight = torch.load(
            args.pos_weight_path, weights_only=False
        ).to(DEVICE)
        print(f"Loaded pos_weight from {args.pos_weight_path}")
    elif not args.no_pos_weight:
        print(
            f"WARNING: {args.pos_weight_path} not found -- training WITHOUT "
            f"pos_weight. Run scripts/compute_pos_weight.py first, or pass "
            f"--no-pos-weight to silence this."
        )

    print("Building data loaders (cold-start split enforced via drug_split.json)...")
    train_loader = build_loader(args, "train", drug_split, shuffle=True)
    val_loader = build_loader(args, "val", drug_split, shuffle=False)

    print("Building model...")
    prior_logits = None
    if args.label_priors:
        prior_logits = torch.load(args.label_priors, weights_only=False)
        print(f"Loaded label priors from {args.label_priors}")

    model = RGCNv2(
        metadata,
        vocab_sizes=vocab_sizes,
        hidden_dim=args.hidden_dim,
        out_dim=NUM_ADE_CLASSES,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head=args.head,
        prior_logits=prior_logits,
    ).to(DEVICE)

    criterion = build_criterion(
        args.loss, pos_weight=pos_weight, focal_gamma=args.focal_gamma
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_macro_ap = -1.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):

        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, class_mask=class_mask)
        val_metrics = run_epoch(model, val_loader, criterion, class_mask=class_mask)

        print(
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train macro-AP {train_metrics['macro_ap']:.4f}"
        )
        print(
            f"Val   loss {val_metrics['loss']:.4f} | "
            f"Val   macro-AP {val_metrics['macro_ap']:.4f} | "
            f"Val   micro-AP {val_metrics['micro_ap']:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        with open(ckpt_dir / "training_history_v2.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["macro_ap"] > best_macro_ap:

            best_macro_ap = val_metrics["macro_ap"]
            patience_counter = 0

            torch.save(model.state_dict(), ckpt_dir / "best_rgcn_v2.pt")
            print("Saved new best checkpoint (by val macro-AP).")

        else:

            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print("\nEarly stopping triggered.")
                break

    print(f"\nTraining finished. Best val macro-AP: {best_macro_ap:.4f}")


if __name__ == "__main__":
    main()
