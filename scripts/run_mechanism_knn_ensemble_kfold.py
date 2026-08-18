"""
k-fold evaluation of the k-NN Tanimoto-transfer / linear-model ensemble --
findings/findings.md item 2. Trains the SAME linear model as
scripts/run_mechanism_kfold.py (in-process here, not via subprocess, since
k-NN needs the trained model's probabilities anyway) on each fold's train
split, computes k-NN transfer probabilities (scripts/mechanism_knn_transfer.py)
from the same train split, and reports linear-alone / knn-alone / blend
(mean and max) macro/micro-AP -- overall and per-kind -- mean +- std across
folds, directly comparable to checkpoints/mechanism_kfold_baseline_results.json.

K (number of neighbors) is chosen ONCE via a val-only sweep on fold 0
(never touching any fold's test split), then reused fixed across all 5
folds' final test evaluation -- consistent with how this project already
treats pos_weight's clamp_max (a fixed global choice, not per-fold-tuned).

Usage:
    python scripts/run_mechanism_knn_ensemble_kfold.py \
        --out checkpoints/mechanism_kfold_knn_blend_results.json
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
from scripts.metrics import macro_and_micro_ap
from scripts.mechanism_knn_transfer import knn_transfer_probs
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
    p.add_argument("--k-sweep", type=int, nargs="*", default=[5, 10, 20, 40, 80])
    p.add_argument("--out", default="checkpoints/mechanism_kfold_knn_blend_results.json")
    return p.parse_args()


def compute_pos_weight(y_train, clamp_min, clamp_max):
    n = y_train.shape[0]
    positives = y_train.sum(dim=0)
    negatives = n - positives
    raw = negatives / (positives + 1.0)
    return torch.clamp(raw, min=clamp_min, max=clamp_max)


def train_linear(x_train, y_train, x_val, y_val, pos_weight, epochs, patience):

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

    best_state, best_macro_ap, patience_counter = None, -1.0, 0

    for _ in range(epochs):
        run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion)
        if val_metrics["macro_ap"] > best_macro_ap:
            best_macro_ap = val_metrics["macro_ap"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def eval_variant(y_true, probs, kind_masks):
    macro_ap, micro_ap = macro_and_micro_ap(y_true, probs)
    metrics = {"macro_ap": macro_ap, "micro_ap": micro_ap}
    metrics["per_kind"] = {}
    for kind, mask in kind_masks.items():
        k_macro, k_micro = macro_and_micro_ap(y_true, probs, class_mask=mask)
        metrics["per_kind"][kind] = {"macro_ap": k_macro, "micro_ap": k_micro, "n_labels": int(mask.sum())}
    return metrics


def sweep_k(x_train, y_train, x_val, y_val, k_values):
    print("\n===== Selecting K via fold-0 val sweep (test split untouched) =====")
    best_k, best_ap = None, -1.0
    for k in k_values:
        probs = knn_transfer_probs(x_val, x_train, y_train, k=k)
        macro_ap, _ = macro_and_micro_ap(y_val, probs)
        print(f"  K={k:3d}  val macro-AP={macro_ap:.4f}")
        if macro_ap > best_ap:
            best_ap, best_k = macro_ap, k
    print(f"  -> selected K={best_k} (val macro-AP {best_ap:.4f})")
    return best_k


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
    kind_masks = {
        kind: torch.tensor([k == kind for k in kinds], dtype=torch.bool)
        for kind in ("enzyme", "target", "transporter")
    }

    with open(args.kfold_splits_path) as f:
        kfold = json.load(f)

    variants = ["linear", "knn", "blend_avg", "blend_max"]
    fold_metrics = {v: [] for v in variants}
    chosen_k = None

    for i, fold in enumerate(kfold["folds"]):

        print(f"\n===== Fold {i + 1}/{kfold['k']} =====")

        x_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["train"])
        x_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["val"])
        x_test, y_test = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["test"])

        if chosen_k is None:
            chosen_k = sweep_k(x_train, y_train, x_val, y_val, args.k_sweep)

        pos_weight = compute_pos_weight(y_train, args.clamp_min, args.clamp_max)
        model = train_linear(
            x_train.to(DEVICE), y_train.to(DEVICE), x_val.to(DEVICE), y_val.to(DEVICE),
            pos_weight, args.epochs, args.patience,
        )

        model.eval()
        with torch.no_grad():
            linear_probs = torch.sigmoid(model(x_test.to(DEVICE))).cpu()

        knn_probs = knn_transfer_probs(x_test, x_train, y_train, k=chosen_k)

        blend_avg = (linear_probs + knn_probs) / 2
        blend_max = torch.maximum(linear_probs, knn_probs)

        results = {
            "linear": linear_probs, "knn": knn_probs,
            "blend_avg": blend_avg, "blend_max": blend_max,
        }

        for variant, probs in results.items():
            m = eval_variant(y_test, probs, kind_masks)
            fold_metrics[variant].append(m)
            print(f"  [{variant:10s}] macro-AP {m['macro_ap']:.4f}  micro-AP {m['micro_ap']:.4f}  "
                  + "  ".join(f"{k}={v['macro_ap']:.3f}" for k, v in m["per_kind"].items()))

    def mean_std(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {"k": kfold["k"], "chosen_knn_k": chosen_k, "variants": {}}

    print(f"\n===== Summary (K={chosen_k}, k={kfold['k']}-fold) =====")
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
        print(f"  [{variant:10s}] macro-AP {macro_mean:.4f} +- {macro_std:.4f}  "
              f"micro-AP {micro_mean:.4f} +- {micro_std:.4f}  |  "
              + "  ".join(f"{k}={v['macro_ap_mean']:.3f}+-{v['macro_ap_std']:.3f}" for k, v in per_kind_summary.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
