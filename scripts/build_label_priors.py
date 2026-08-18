"""
check_prediction_variance.py confirmed: the trained model outputs a nearly
IDENTICAL prediction vector regardless of input (cosine similarity ~1.0
across different drug pairs), while the untrained model shows healthy
per-example variance. So the model isn't broken -- training is converging to
"predict each class's base rate and ignore the input", a well-documented
failure mode in extreme multi-label imbalance (many classes here have
positive rates around 0.006%, i.e. 10 positives out of 172,754 examples).
It's the cheapest way to minimize loss, and gradient signal for genuinely
input-dependent (rare-positive) learning is comparatively tiny.

Standard fix (from the focal-loss/RetinaNet literature): initialize the
classifier's final-layer BIAS to each class's empirical prior logit
(log(p / (1-p))) instead of zero. The model then starts already at the
base-rate solution, so training gradients immediately encode the residual,
input-dependent signal instead of spending most of the training budget just
discovering priors from scratch.

Usage:
    python scripts/build_label_priors.py --shard-dir data/tensorized_v2 \
        --out data/processed/label_priors.pt
"""

import argparse
from pathlib import Path

import torch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="data/tensorized_v2")
    parser.add_argument("--out", default="data/processed/label_priors.pt")
    args = parser.parse_args()

    shard_files = sorted(Path(args.shard_dir).glob("shard_*.pt"))
    if not shard_files:
        raise SystemExit(f"No shards found under {args.shard_dir}")

    print(f"Scanning {len(shard_files)} shards for per-class positive rates...")

    positives = None
    total = 0

    for shard_path in shard_files:
        graphs = torch.load(shard_path, weights_only=False)
        for g in graphs:
            y = g.y.squeeze(0)
            if positives is None:
                positives = torch.zeros_like(y)
            positives += y
            total += 1

    # Laplace smoothing (+1/-1) so classes with 0 positives don't produce
    # -inf logits, and classes with all positives don't produce +inf.
    p = (positives + 1) / (total + 2)
    prior_logits = torch.log(p / (1 - p))

    torch.save(prior_logits, args.out)

    print(f"\nPositive rate range: {p.min().item():.6f} to {p.max().item():.6f}")
    print(f"Prior logit range: {prior_logits.min().item():.3f} to {prior_logits.max().item():.3f}")
    print(f"Saved: {args.out}")
    print(
        "\nPass this to models/rgcn_v2.py's RGCNv2(..., prior_logits=...) "
        "so the classifier's final bias starts at the empirical base rate "
        "instead of zero."
    )


if __name__ == "__main__":
    main()
