"""
k-fold evaluation of ChemBERTa-embedding-based mechanism prediction --
findings/findings.md item 3 ("pretrained molecular embeddings"), attempted
because item 1's diagnosis established the gap-filler's weakness is a
REPRESENTATION problem (raw Morgan/ECFP bits are a weak enzyme/transporter
predictor regardless of training volume), not something more data would
fix -- so a differently-representation (a pretrained encoder trained on a
much larger, more diverse chemical space) is the natural next lever, and
specifically the lever for generalizing to genuinely novel drugs/scaffolds
that don't resemble anything already in DrugBank -- not just a better score
on drugs already similar to the training pool (which the existing Tanimoto
k-NN ensemble already covers).

Same 5-fold splits, same linear-model training loop, same evaluation
harness as scripts/run_mechanism_kfold.py / run_mechanism_knn_ensemble_kfold.py
-- directly comparable to their saved results
(checkpoints/mechanism_kfold_baseline_results.json: linear-on-fingerprints
0.258+-0.013; checkpoints/mechanism_kfold_knn_blend_results.json:
fingerprint-Tanimoto-knn blend 0.293+-0.016), NOT a new single-split number
(this project's own established mistake to avoid, see findings.md's
"Close-out" section).

Variants evaluated:
  - embed_linear : linear model trained on ChemBERTa embeddings alone
  - embed_knn     : cosine-similarity k-NN vote in embedding space alone
  - embed_blend   : average(embed_linear, embed_knn)
  - concat_linear : linear model on [fingerprint ++ embedding] concatenated
  - all4_blend    : average of ALL FOUR representations' predictions
                    (fp_linear, fp_knn, embed_linear, embed_knn) -- only
                    meaningful if the embedding signal is genuinely
                    complementary to the existing fingerprint ensemble,
                    not redundant with it.

Usage:
    python scripts/run_mechanism_embedding_kfold.py \
        --out checkpoints/mechanism_kfold_embedding_results.json
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
from scripts.mechanism_knn_transfer import cosine_similarity, knn_transfer_probs, tanimoto_similarity
from scripts.metrics import macro_and_micro_ap
from scripts.train_mechanism_predictor import build_xy, run_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kfold-splits-path", default="data/processed/mechanism_kfold_splits.json")
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--embeddings-path", default="data/processed/drug_chemberta_embeddings.pt")
    p.add_argument("--vocab-path", default="data/processed/mechanism_label_vocab.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--clamp-min", type=float, default=1.0)
    p.add_argument("--clamp-max", type=float, default=100.0)
    p.add_argument("--fp-knn-k", type=int, default=40, help="Matches run_mechanism_knn_ensemble_kfold.py's chosen K.")
    p.add_argument("--embed-k-sweep", type=int, nargs="*", default=[5, 10, 20, 40, 80])
    p.add_argument("--out", default="checkpoints/mechanism_kfold_embedding_results.json")
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


def sweep_embed_k(emb_train, y_train, emb_val, y_val, k_values):
    print("\n===== Selecting embedding-kNN K via fold-0 val sweep (test split untouched) =====")
    best_k, best_ap = None, -1.0
    for k in k_values:
        probs = knn_transfer_probs(emb_val, emb_train, y_train, k=k, similarity_fn=cosine_similarity)
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

    emb_data = torch.load(args.embeddings_path, weights_only=False)
    emb_by_drug = dict(zip(emb_data["drug_ids"], emb_data["embeddings"]))

    missing = [d for d in drug_ids if d not in emb_by_drug]
    if missing:
        raise SystemExit(f"{len(missing)} labeled drugs have no ChemBERTa embedding: {missing[:5]}...")

    with open(args.vocab_path) as f:
        vocab = json.load(f)["labels"]
    kinds = [label.split(":", 1)[0] for label in vocab]
    kind_masks = {
        kind: torch.tensor([k == kind for k in kinds], dtype=torch.bool)
        for kind in ("enzyme", "target", "transporter")
    }

    with open(args.kfold_splits_path) as f:
        kfold = json.load(f)

    variants = ["embed_linear", "embed_knn", "embed_blend", "concat_linear", "all4_blend"]
    fold_metrics = {v: [] for v in variants}
    chosen_embed_k = None

    for i, fold in enumerate(kfold["folds"]):

        print(f"\n===== Fold {i + 1}/{kfold['k']} =====")

        fp_train, y_train = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["train"])
        fp_val, y_val = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["val"])
        fp_test, y_test = build_xy(drug_ids, fp_by_drug, label_by_drug, fold["test"])

        emb_train, _ = build_xy(drug_ids, emb_by_drug, label_by_drug, fold["train"])
        emb_val, _ = build_xy(drug_ids, emb_by_drug, label_by_drug, fold["val"])
        emb_test, _ = build_xy(drug_ids, emb_by_drug, label_by_drug, fold["test"])

        concat_train = torch.cat([fp_train, emb_train], dim=1)
        concat_val = torch.cat([fp_val, emb_val], dim=1)
        concat_test = torch.cat([fp_test, emb_test], dim=1)

        if chosen_embed_k is None:
            chosen_embed_k = sweep_embed_k(emb_train, y_train, emb_val, y_val, args.embed_k_sweep)

        pos_weight = compute_pos_weight(y_train, args.clamp_min, args.clamp_max)

        embed_model = train_linear(
            emb_train.to(DEVICE), y_train.to(DEVICE), emb_val.to(DEVICE), y_val.to(DEVICE),
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

        embed_model.eval()
        concat_model.eval()
        fp_model.eval()
        with torch.no_grad():
            embed_linear_probs = torch.sigmoid(embed_model(emb_test.to(DEVICE))).cpu()
            concat_linear_probs = torch.sigmoid(concat_model(concat_test.to(DEVICE))).cpu()
            fp_linear_probs = torch.sigmoid(fp_model(fp_test.to(DEVICE))).cpu()

        embed_knn_probs = knn_transfer_probs(
            emb_test, emb_train, y_train, k=chosen_embed_k, similarity_fn=cosine_similarity
        )
        fp_knn_probs = knn_transfer_probs(
            fp_test, fp_train, y_train, k=args.fp_knn_k, similarity_fn=tanimoto_similarity
        )

        embed_blend_probs = (embed_linear_probs + embed_knn_probs) / 2
        all4_blend_probs = (embed_linear_probs + embed_knn_probs + fp_linear_probs + fp_knn_probs) / 4

        results = {
            "embed_linear": embed_linear_probs,
            "embed_knn": embed_knn_probs,
            "embed_blend": embed_blend_probs,
            "concat_linear": concat_linear_probs,
            "all4_blend": all4_blend_probs,
        }

        for variant, probs in results.items():
            m = eval_variant(y_test, probs, kind_masks)
            fold_metrics[variant].append(m)
            print(f"  [{variant:14s}] macro-AP {m['macro_ap']:.4f}  micro-AP {m['micro_ap']:.4f}  "
                  + "  ".join(f"{k}={v['macro_ap']:.3f}" for k, v in m["per_kind"].items()))

    def mean_std(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    summary = {"k": kfold["k"], "chosen_embed_knn_k": chosen_embed_k, "fp_knn_k": args.fp_knn_k, "variants": {}}

    print(f"\n===== Summary (embed K={chosen_embed_k}, fp K={args.fp_knn_k}, k={kfold['k']}-fold) =====")
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
        print(f"  [{variant:14s}] macro-AP {macro_mean:.4f} +- {macro_std:.4f}  "
              f"micro-AP {micro_mean:.4f} +- {micro_std:.4f}  |  "
              + "  ".join(f"{k}={v['macro_ap_mean']:.3f}+-{v['macro_ap_std']:.3f}" for k, v in per_kind_summary.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
