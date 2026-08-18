"""
Evaluation script for the CURRENT pipeline (RGCNv2 + ShardDatasetV2 +
checkpoints/best_rgcn_v2.pt). Fills a gap left by train_v2.py: the only
evaluators that existed (evaluate_rgcn.py, evaluate_test_rgcn.py) were still
hardcoded to the old RGCNBaseline/ShardDataset/best_rgcn.pt, so running "the"
eval script would silently evaluate the wrong (superseded) model.

Mirrors train_v2.py's build_loader/run_epoch exactly (same cold-start
drug_split filtering, same macro/micro AP via scripts/metrics.py, same
optional label_mask) so numbers here are directly comparable to what
train_v2.py logged during training.

Usage (from the Capstone/ directory):
    python scripts/evaluate_v2.py --split val
    python scripts/evaluate_v2.py --split test --checkpoint checkpoints/best_rgcn_v2.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch_geometric.loader import DataLoader

from data_utils.shard_dataset_v2 import ShardDatasetV2
from models.rgcn_v2 import RGCNv2
from scripts.losses import build_criterion
from scripts.metrics import macro_and_micro_ap


NUM_ADE_CLASSES = 4486
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--shard-list-dir", default="data/splits")
    p.add_argument("--vocab-sizes", default="data/processed/vocab_sizes.json")
    p.add_argument("--drug-split", default="data/processed/drug_split.json")
    p.add_argument("--pos-weight-path", default="data/processed/pos_weight.pt")
    p.add_argument("--global-metadata", default="data/processed/global_metadata.pt")
    p.add_argument("--label-mask", default="data/processed/label_mask.json")
    p.add_argument("--checkpoint", default="checkpoints/best_rgcn_v2.pt")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--head", choices=["mean", "bilinear"], default="bilinear")
    p.add_argument("--loss", choices=["bce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--no-pos-weight", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-shards", type=int, default=None,
                    help="Only read the first N shards -- for a quick smoke run.")
    p.add_argument("--save-preds", default=None,
                    help="If given, save {'preds','targets'} tensors to this "
                    "path. Off by default -- the old evaluate_rgcn.py always "
                    "dumped a 694MB val_predictions.pt, which is exactly the "
                    "kind of regenerable large file this cleanup removed.")
    return p.parse_args()


def main():

    args = parse_args()

    with open(args.vocab_sizes) as f:
        vocab_sizes = json.load(f)

    with open(args.drug_split) as f:
        raw_drug_split = json.load(f)
        drug_split = {k: set(v) for k, v in raw_drug_split.items()}

    metadata = torch.load(args.global_metadata, weights_only=False)

    class_mask = None
    if args.label_mask and Path(args.label_mask).exists():
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
        pos_weight = torch.load(args.pos_weight_path, weights_only=False).to(DEVICE)
        print(f"Loaded pos_weight from {args.pos_weight_path}")

    shard_list_file = Path(args.shard_list_dir) / f"{args.split}.txt"
    with open(shard_list_file) as f:
        shard_paths = [line.strip() for line in f if line.strip()]
    if args.limit_shards:
        shard_paths = shard_paths[: args.limit_shards]

    tmp_list = Path(f"/tmp/_evaluate_v2_{args.split}_shards.txt")
    tmp_list.write_text("\n".join(shard_paths) + "\n")

    dataset = ShardDatasetV2(
        tmp_list,
        drug_split=drug_split,
        split_name=args.split,
        shuffle=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Building model (device={DEVICE})...")
    model = RGCNv2(
        metadata,
        vocab_sizes=vocab_sizes,
        hidden_dim=args.hidden_dim,
        out_dim=NUM_ADE_CLASSES,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head=args.head,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    )
    model.eval()

    criterion = build_criterion(args.loss, pos_weight=pos_weight, focal_gamma=args.focal_gamma)

    all_targets = []
    all_probs = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():

        for step, batch in enumerate(loader):

            batch = batch.to(DEVICE)

            logits = model(batch)
            loss = criterion(logits, batch.y)

            total_loss += loss.item()
            steps += 1

            all_targets.append(batch.y.cpu())
            all_probs.append(torch.sigmoid(logits).cpu())

            if step % 20 == 0:
                print(f"Processed batch {step}")

    if steps == 0:
        raise SystemExit(
            f"No batches produced for split={args.split!r} -- check "
            f"{shard_list_file} and --drug-split filtering."
        )

    y_true = torch.cat(all_targets)
    y_score = torch.cat(all_probs)

    macro_ap, micro_ap = macro_and_micro_ap(y_true, y_score, class_mask=class_mask)

    print(f"\n{args.split} loss: {total_loss / steps:.4f}")
    print(f"{args.split} macro-AP: {macro_ap:.4f}")
    print(f"{args.split} micro-AP: {micro_ap:.4f}")
    print(f"(n_samples={y_true.shape[0]}, n_classes={y_true.shape[1]})")

    if args.save_preds:
        torch.save({"preds": y_score, "targets": y_true}, args.save_preds)
        print(f"\nSaved predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
