"""
k-fold cross-validation orchestrator for the mechanism gap-filler. Trains
and evaluates one model per fold (via subprocess calls to the existing,
already-tested scripts/train_mechanism_predictor.py and
scripts/evaluate_mechanism_predictor.py CLIs -- no internal refactor of
either needed), then reports mean+-std macro/micro-AP across folds. This is
the honest headline number findings/findings.md flagged as missing: a
single 388-drug test split is small enough to be noisy on its own.

Supports a --pos-weight-mode {uniform,per-category} sweep so the same
orchestrator covers both the Phase 1 baseline run and the Phase 2/4
per-category-pos_weight ablation without duplicating the fold-loop logic.

Usage:
    python scripts/run_mechanism_kfold.py --tag baseline \
        --out checkpoints/mechanism_kfold_baseline_results.json

    python scripts/run_mechanism_kfold.py --tag pcw_transporter200 \
        --pos-weight-mode per-category --transporter-clamp-max 200 \
        --out checkpoints/mechanism_kfold_pcw_results.json
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kfold-splits-path", default="data/processed/mechanism_kfold_splits.json")
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--pos-weight-mode", choices=["uniform", "per-category"], default="uniform")
    p.add_argument("--clamp-max", type=float, default=100.0)
    p.add_argument("--transporter-clamp-max", type=float, default=None)
    p.add_argument("--enzyme-clamp-max", type=float, default=None)
    p.add_argument("--target-clamp-max", type=float, default=None)
    p.add_argument("--dataset-path", default="data/processed/mechanism_dataset.pt")
    p.add_argument("--fingerprints-path", default="data/processed/drug_fingerprints.pt")
    p.add_argument("--tag", default="run")
    p.add_argument("--scratch-dir", default="/tmp/mechanism_kfold_scratch")
    p.add_argument("--out", default="checkpoints/mechanism_kfold_results.json")
    return p.parse_args()


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return result


def main():

    args = parse_args()

    with open(args.kfold_splits_path) as f:
        kfold = json.load(f)

    scratch = Path(args.scratch_dir) / args.tag
    scratch.mkdir(parents=True, exist_ok=True)

    fold_metrics = []

    for i, fold in enumerate(kfold["folds"]):

        print(f"\n===== Fold {i + 1}/{kfold['k']} ({args.tag}) =====")

        fold_split_path = scratch / f"split_fold{i}.json"
        with open(fold_split_path, "w") as f:
            json.dump(fold, f)

        pos_weight_path = scratch / f"pos_weight_fold{i}.pt"
        pw_cmd = [
            "python", "scripts/compute_mechanism_pos_weight.py",
            "--dataset-path", args.dataset_path,
            "--split-path", str(fold_split_path), "--split", "train",
            "--clamp-max", str(args.clamp_max),
            "--out", str(pos_weight_path),
        ]
        if args.pos_weight_mode == "per-category":
            pw_cmd.append("--per-category")
            if args.transporter_clamp_max:
                pw_cmd += ["--transporter-clamp-max", str(args.transporter_clamp_max)]
            if args.enzyme_clamp_max:
                pw_cmd += ["--enzyme-clamp-max", str(args.enzyme_clamp_max)]
            if args.target_clamp_max:
                pw_cmd += ["--target-clamp-max", str(args.target_clamp_max)]
        run(pw_cmd)

        checkpoint_dir = scratch / f"ckpt_fold{i}"
        checkpoint_name = "model.pt"
        train_cmd = [
            "python", "scripts/train_mechanism_predictor.py",
            "--dataset-path", args.dataset_path,
            "--fingerprints-path", args.fingerprints_path,
            "--split-path", str(fold_split_path),
            "--pos-weight-path", str(pos_weight_path),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--hidden-dims", *[str(h) for h in args.hidden_dims],
            "--dropout", "0.0" if not args.hidden_dims else "0.4",
            "--checkpoint-dir", str(checkpoint_dir),
            "--checkpoint-name", checkpoint_name,
            "--history-name", "history.json",
        ]
        run(train_cmd)

        metrics_path = scratch / f"metrics_fold{i}.json"
        eval_cmd = [
            "python", "scripts/evaluate_mechanism_predictor.py",
            "--split", "test",
            "--checkpoint", str(checkpoint_dir / checkpoint_name),
            "--hidden-dims", *[str(h) for h in args.hidden_dims],
            "--dataset-path", args.dataset_path,
            "--fingerprints-path", args.fingerprints_path,
            "--split-path", str(fold_split_path),
            "--per-kind",
            "--metrics-out", str(metrics_path),
        ]
        run(eval_cmd)

        with open(metrics_path) as f:
            m = json.load(f)
        fold_metrics.append(m)
        print(f"  fold {i}: macro-AP {m['macro_ap']:.4f}  micro-AP {m['micro_ap']:.4f}")

    def mean_std(key):
        vals = [m[key] for m in fold_metrics]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    macro_mean, macro_std = mean_std("macro_ap")
    micro_mean, micro_std = mean_std("micro_ap")

    summary = {
        "tag": args.tag,
        "k": kfold["k"],
        "pos_weight_mode": args.pos_weight_mode,
        "hidden_dims": args.hidden_dims,
        "macro_ap_mean": macro_mean,
        "macro_ap_std": macro_std,
        "micro_ap_mean": micro_mean,
        "micro_ap_std": micro_std,
        "per_fold": fold_metrics,
    }

    print(f"\n===== {args.tag}: macro-AP {macro_mean:.4f} +- {macro_std:.4f} | "
          f"micro-AP {micro_mean:.4f} +- {micro_std:.4f} (k={kfold['k']}) =====")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
