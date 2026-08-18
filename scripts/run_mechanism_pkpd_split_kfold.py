"""
k-fold evaluation of PK/PD (kind-specific) model splitting --
findings/findings.md item 4, the last of the four originally-proposed
interventions. Trains THREE separate linear models (one each for enzyme,
target, transporter labels) instead of one joint 422-label linear model,
then concatenates their predictions back for evaluation.

IMPORTANT: this tests a genuinely DIFFERENT mechanism than the already-
tried-and-failed per-category pos_weight ablation
(checkpoints/mechanism_kfold_pcw_results.json). A plain linear layer
already has a fully separate weight vector per output label -- there's no
shared-capacity bottleneck between labels at the parameter level, so
splitting into 3 models doesn't change what functions are representable.
What DOES change: each kind-specific model gets its own EARLY-STOPPING
criterion (best val macro-AP for THAT KIND alone), not one epoch chosen to
jointly optimize the full 422-label average. If the weak kinds (enzyme,
transporter) have a different optimal training schedule than target
(which dominates the joint average, being the strongest and most numerous
kind), joint early-stopping could be picking a suboptimal epoch for them
specifically -- that's the hypothesis this script actually tests.

Same 5-fold splits, comparable to the existing baselines (fp linear alone:
0.258 ± 0.014; deployed fp Tanimoto-knn blend: 0.293 ± 0.016).

Variants:
  - split_linear : 3 separate per-kind linear models, predictions concatenated
  - split_blend  : average(split_linear, fp_knn) -- same ensemble recipe as
                   the deployed blend, but with the split model instead of
                   the joint one

Usage:
    python scripts/run_mechanism_pkpd_split_kfold.py \
        --out checkpoints/mechanism_kfold_pkpd_split_results.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from models.mechanism_mlp import MechanismMLP
from scripts.losses import build_criterion
from scripts.mechanism_knn_transfer import knn_transfer_probs, tanimoto_similarity
from scripts.metrics import macro_and_micro_ap
from scripts.train_mechanism_predictor import build_xy, run_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kfold-splits-path", default="data/processed/mechanism_kfold_splits.json")
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--clamp-min", type=float, default=1.0)
    p.add_argument("--clamp-max", type=float, default=100.0)
    p.add_argument("--fp-knn-k", type=int, default=40)
    p.add_argument("--out", default="checkpoints/mechanism_kfold_pkpd_split_results.json")
    return p.parse_args()


def compute_pos_weight(y_train, clamp_min, clamp_max):
    n = y_train.shape[0]
    positives = y_train.sum(dim=0)
    negatives = n - positives
    raw = negatives / (positives + 1.0)
    return torch.clamp(raw, min=clamp_min, max=clamp_max)


def train_linear(x_train, y_train, x_val, y_val, pos_weight, epochs, patience):
    """Same training loop as the other kfold scripts, but early-stopping is
    on THIS call's y_train/y_val -- when called per-kind, that means the
    kind-specific macro-AP, not the joint 422-label average."""

    num_labels = y_train.shape[1]
    p = (y_train.sum(dim=0) + 1) / (y_train.shape[0] + 2)
    prior_logits = torch.log(p / (1 - p))

    model = MechanismMLP(
        in_dim=x_train.shape[1], hidden_dims=(), out_dim=num_labels,
        dropout=0.0, prior_logits=prior_logits,
    ).to(DEVICE)

    criterion = build_criterion("bce", pos_weight=pos_weight.to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=128, shuffle=False)

    best_state, best_macro_ap, patience_counter, best_epoch = None, -1.0, 0, 0

    for epoch in range(epochs):
        run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion)
        if val_metrics["macro_ap"] > best_macro_ap:
            best_macro_ap = val_metrics["macro_ap"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def eval_variant(y_true, probs, kind_masks):
    macro_ap, micro_ap = macro_and_micro_ap(y_true, probs)
    metrics = {"macro_ap": macro_ap, "micro_ap": micro_ap}
    metrics["per_kind"] = {}
    for kind, mask in kind_masks.items():
        k_macro, k_micro = macro_and_micro_ap(y_true, probs, class_mask=mask)
        metrics["per_kind"][kind] = {"macro_ap": k_macro, "micro_ap": k_micro, "n_labels": int(mask.sum())}
    return metrics


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids, label_matrix = dataset["drug_ids"], dataset["labels"]
    label_by_drug = dict(zip(drug_ids, label_matrix))

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    with open(args.vocab_path) as f:
        vocab = json.load(f)["labels"]
    kinds = [label.split(":", 1)[0] for label in vocab]
    kind_names = ("enzyme", "target", "transporter")
    kind_masks = {kind: torch.tensor([k == kind for k in kinds], dtype=torch.bool) for kind in kind_names}
    kind_idx = {kind: mask.nonzero(as_tuple=True)[0] for kind, mask in kind_masks.items()}

    with open(args.kfold_splits_path) as f:
        kfold = json.load(f)

    variants = ["split_linear", "split_blend"]
    fold_metrics = {v: [] for v in variants}
    best_epochs_by_kind = {kind: [] for kind in kind_names}
    joint_best_epochs = []

    for i, fold in enumerate(kfold["folds"]):

        print(f"\n===== Fold {i + 1}/{kfold['k']} =====")

        fp_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["train"])
        fp_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["val"])
        fp_test, y_test = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["test"])

        pos_weight = compute_pos_weight(y_train, args.clamp_min, args.clamp_max)

        # Joint model, for reference on this exact fold (same recipe as
        # run_mechanism_kfold.py, re-trained here so the best-epoch
        # comparison below is apples-to-apples on identical data/pos_weight).
        _, joint_epoch = train_linear(
            fp_train.to(DEVICE), y_train.to(DEVICE), fp_val.to(DEVICE), y_val.to(DEVICE),
            pos_weight, args.epochs, args.patience,
        )
        joint_best_epochs.append(joint_epoch)

        split_test_probs = torch.zeros(fp_test.shape[0], len(vocab))

        for kind in kind_names:
            idx = kind_idx[kind]
            model, best_epoch = train_linear(
                fp_train.to(DEVICE), y_train[:, idx].to(DEVICE),
                fp_val.to(DEVICE), y_val[:, idx].to(DEVICE),
                pos_weight[idx], args.epochs, args.patience,
            )
            best_epochs_by_kind[kind].append(best_epoch)
            model.eval()
            with torch.no_grad():
                split_test_probs[:, idx] = torch.sigmoid(model(fp_test.to(DEVICE))).cpu()

        fp_knn_probs = knn_transfer_probs(
            fp_test, fp_train, y_train, k=args.fp_knn_k, similarity_fn=tanimoto_similarity
        )
        split_blend_probs = (split_test_probs + fp_knn_probs) / 2

        results = {"split_linear": split_test_probs, "split_blend": split_blend_probs}

        for variant, probs in results.items():
            m = eval_variant(y_test, probs, kind_masks)
            fold_metrics[variant].append(m)
            print(f"  [{variant:14s}] macro-AP {m['macro_ap']:.4f}  micro-AP {m['micro_ap']:.4f}  "
                  + "  ".join(f"{k}={v['macro_ap']:.3f}" for k, v in m["per_kind"].items()))

        print(f"  best epochs: joint={joint_epoch}  " + "  ".join(f"{k}={best_epochs_by_kind[k][-1]}" for k in kind_names))

    def mean_std(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {
        "k": kfold["k"], "fp_knn_k": args.fp_knn_k,
        "joint_best_epoch_mean": statistics.mean(joint_best_epochs),
        "per_kind_best_epoch_mean": {k: statistics.mean(v) for k, v in best_epochs_by_kind.items()},
        "variants": {},
    }

    print(f"\n===== Summary (fp K={args.fp_knn_k}, k={kfold['k']}-fold) =====")
    print("  Reference: fp linear alone 0.2581+-0.0137, fp Tanimoto-knn blend 0.2927+-0.0156 (from prior results)")
    print(f"  Best-epoch comparison: joint model mean={summary['joint_best_epoch_mean']:.1f}  "
          + "  ".join(f"{k}={v:.1f}" for k, v in summary["per_kind_best_epoch_mean"].items()))
    for variant in variants:
        macro_mean, macro_std = mean_std([m["macro_ap"] for m in fold_metrics[variant]])
        micro_mean, micro_std = mean_std([m["micro_ap"] for m in fold_metrics[variant]])
        per_kind_summary = {}
        for kind in kind_masks:
            k_macro_mean, k_macro_std = mean_std([m["per_kind"][kind]["macro_ap"] for m in fold_metrics[variant]])
            per_kind_summary[kind] = {"macro_ap_mean": k_macro_mean, "macro_ap_std": k_macro_std}
        summary["variants"][variant] = {
            "macro_ap_mean": macro_mean, "macro_ap_std": macro_std,
            "micro_ap_mean": micro_mean, "micro_ap_std": micro_std,
            "per_kind": per_kind_summary,
            "per_fold": fold_metrics[variant],
        }
        print(f"  [{variant:14s}] macro-AP {macro_mean:.4f} +- {macro_std:.4f}  "
              f"micro-AP {micro_mean:.4f} +- {micro_std:.4f}  |  "
              + "  ".join(f"{k}={v['macro_ap_mean']:.3f}+-{v['macro_ap_std']:.3f}" for k, v in per_kind_summary.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
