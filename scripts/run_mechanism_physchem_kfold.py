"""
k-fold evaluation of RDKit physicochemical descriptors for mechanism
prediction -- testing findings/findings.md's ORIGINAL hypothesis (that
transporter/PK recognition depends more on global physicochemical
properties than the local 2D substructure patterns Morgan/ECFP encodes),
which was never actually tested before the ChemBERTa detour (which tested a
different fix -- pretrained representation -- and didn't help; see
checkpoints/mechanism_kfold_embedding_results.json).

Same 5-fold splits, same linear-model training loop as
run_mechanism_kfold.py / run_mechanism_knn_ensemble_kfold.py --
comparable to their saved results (fp linear alone: 0.258+-0.013;
deployed fp Tanimoto-knn blend: 0.293+-0.016).

Descriptors are z-score normalized using TRAIN-split statistics only, per
fold (not baked into the saved artifact -- avoids leaking val/test
statistics into normalization, same discipline as prior_logits/pos_weight).

Variants:
  - physchem_linear : linear model on normalized physchem descriptors alone
  - concat_linear    : linear model on [fingerprint ++ normalized physchem]
  - concat_blend     : average(concat_linear, fp_knn) -- does adding
                       physchem to the linear side improve on the deployed
                       ensemble recipe (fp_linear + fp_knn averaged)?
  - all3_blend       : average(fp_linear, fp_knn, concat_linear)

Usage:
    python scripts/run_mechanism_physchem_kfold.py \
        --out checkpoints/mechanism_kfold_physchem_results.json
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
    p.add_argument("--physchem-path", default="data/processed/drug_physchem_descriptors.pt")
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--clamp-min", type=float, default=1.0)
    p.add_argument("--clamp-max", type=float, default=100.0)
    p.add_argument("--fp-knn-k", type=int, default=40, help="Matches run_mechanism_knn_ensemble_kfold.py's chosen K.")
    p.add_argument("--out", default="checkpoints/mechanism_kfold_physchem_results.json")
    return p.parse_args()


def compute_pos_weight(y_train, clamp_min, clamp_max):
    n = y_train.shape[0]
    positives = y_train.sum(dim=0)
    negatives = n - positives
    raw = negatives / (positives + 1.0)
    return torch.clamp(raw, min=clamp_min, max=clamp_max)


def zscore_normalize(x_train, x_val, x_test):
    """Normalizes val/test using TRAIN-only mean/std -- never fit on val/test."""
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std


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


def main():

    args = parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    drug_ids, label_matrix = dataset["drug_ids"], dataset["labels"]
    label_by_drug = dict(zip(drug_ids, label_matrix))

    fp_data = torch.load(args.fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    pc_data = torch.load(args.physchem_path, weights_only=False)
    pc_by_drug = dict(zip(pc_data["drug_ids"], pc_data["descriptors"]))

    missing = [d for d in drug_ids if d not in pc_by_drug]
    if missing:
        raise SystemExit(f"{len(missing)} labeled drugs have no physchem descriptors: {missing[:5]}...")

    with open(args.vocab_path) as f:
        vocab = json.load(f)["labels"]
    kinds = [label.split(":", 1)[0] for label in vocab]
    kind_masks = {
        kind: torch.tensor([k == kind for k in kinds], dtype=torch.bool)
        for kind in ("enzyme", "target", "transporter")
    }

    with open(args.kfold_splits_path) as f:
        kfold = json.load(f)

    variants = ["physchem_linear", "concat_linear", "concat_blend", "all3_blend"]
    fold_metrics = {v: [] for v in variants}

    for i, fold in enumerate(kfold["folds"]):

        print(f"\n===== Fold {i + 1}/{kfold['k']} =====")

        fp_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["train"])
        fp_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["val"])
        fp_test, y_test = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["test"])

        pc_train_raw, _ = build_xy(drug_ids, pc_by_drug, label_by_drug, fold["train"])
        pc_val_raw, _ = build_xy(drug_ids, pc_by_drug, label_by_drug, fold["val"])
        pc_test_raw, _ = build_xy(drug_ids, pc_by_drug, label_by_drug, fold["test"])

        pc_train, pc_val, pc_test = zscore_normalize(pc_train_raw, pc_val_raw, pc_test_raw)

        concat_train = torch.cat([fp_train, pc_train], dim=1)
        concat_val = torch.cat([fp_val, pc_val], dim=1)
        concat_test = torch.cat([fp_test, pc_test], dim=1)

        pos_weight = compute_pos_weight(y_train, args.clamp_min, args.clamp_max)

        physchem_model = train_linear(
            pc_train.to(DEVICE), y_train.to(DEVICE), pc_val.to(DEVICE), y_val.to(DEVICE),
            pos_weight, args.epochs, args.patience,
        )
        concat_model = train_linear(
            concat_train.to(DEVICE), y_train.to(DEVICE), concat_val.to(DEVICE), y_val.to(DEVICE),
            pos_weight, args.epochs, args.patience,
        )
        fp_model = train_linear(
            fp_train.to(DEVICE), y_train.to(DEVICE), fp_val.to(DEVICE), y_val.to(DEVICE),
            pos_weight, args.epochs, args.patience,
        )

        physchem_model.eval()
        concat_model.eval()
        fp_model.eval()
        with torch.no_grad():
            physchem_linear_probs = torch.sigmoid(physchem_model(pc_test.to(DEVICE))).cpu()
            concat_linear_probs = torch.sigmoid(concat_model(concat_test.to(DEVICE))).cpu()
            fp_linear_probs = torch.sigmoid(fp_model(fp_test.to(DEVICE))).cpu()

        fp_knn_probs = knn_transfer_probs(
            fp_test, fp_train, y_train, k=args.fp_knn_k, similarity_fn=tanimoto_similarity
        )

        concat_blend_probs = (concat_linear_probs + fp_knn_probs) / 2
        all3_blend_probs = (fp_linear_probs + fp_knn_probs + concat_linear_probs) / 3

        results = {
            "physchem_linear": physchem_linear_probs,
            "concat_linear": concat_linear_probs,
            "concat_blend": concat_blend_probs,
            "all3_blend": all3_blend_probs,
        }

        for variant, probs in results.items():
            m = eval_variant(y_test, probs, kind_masks)
            fold_metrics[variant].append(m)
            print(f"  [{variant:16s}] macro-AP {m['macro_ap']:.4f}  micro-AP {m['micro_ap']:.4f}  "
                  + "  ".join(f"{k}={v['macro_ap']:.3f}" for k, v in m["per_kind"].items()))

    def mean_std(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {"k": kfold["k"], "fp_knn_k": args.fp_knn_k, "descriptor_names": pc_data["descriptor_names"], "variants": {}}

    print(f"\n===== Summary (fp K={args.fp_knn_k}, k={kfold['k']}-fold) =====")
    print("  Reference: fp linear alone 0.2581+-0.0137, fp Tanimoto-knn blend 0.2927+-0.0156 (from prior results)")
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
        print(f"  [{variant:16s}] macro-AP {macro_mean:.4f} +- {macro_std:.4f}  "
              f"micro-AP {micro_mean:.4f} +- {micro_std:.4f}  |  "
              + "  ".join(f"{k}={v['macro_ap_mean']:.3f}+-{v['macro_ap_std']:.3f}" for k, v in per_kind_summary.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
