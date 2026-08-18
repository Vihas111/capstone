"""
Tanimoto-similarity k-NN mechanism-label transfer -- findings/findings.md
item 2 ("k-NN similarity transfer, as an ensemble member, not a
replacement"). This is the never-built similarity_gapfiller.py's actual
job, but framed from the start as an ensemble component to blend with
scripts/train_mechanism_predictor.py's linear model, not a standalone
competitor (per the doc's own correction to its original framing).

For a query drug's fingerprint, finds its K nearest TRAIN-split neighbors
by Tanimoto similarity (the standard metric for binary Morgan/ECFP bits --
intersection-over-union of the "on" bits) and predicts each label's
probability as the similarity-weighted fraction of those neighbors that
have that label. Fully vectorized (one similarity matrix, one topk, one
gather) -- no per-drug Python loop.

Usage (as a library, from scripts/run_mechanism_knn_ensemble_kfold.py):
    from scripts.mechanism_knn_transfer import knn_transfer_probs
    probs = knn_transfer_probs(x_query, x_train, y_train, k=20)
"""

import torch


def tanimoto_similarity(x_query, x_train):
    """x_query: [Nq, D] binary. x_train: [Nt, D] binary. Returns [Nq, Nt]
    Tanimoto similarity (= intersection / union for binary vectors)."""

    intersection = x_query @ x_train.T
    q_sum = x_query.sum(dim=1, keepdim=True)
    t_sum = x_train.sum(dim=1, keepdim=True).T
    union = q_sum + t_sum - intersection

    return torch.where(union > 0, intersection / union.clamp(min=1e-8), torch.zeros_like(intersection))


def knn_transfer_probs(x_query, x_train, y_train, k=20):
    """Returns [Nq, C] predicted label probabilities: for each query drug,
    the similarity-weighted vote of its k most Tanimoto-similar train
    drugs' binary labels. Query drugs with zero similarity to every train
    drug (no shared fingerprint bits at all) get a uniform-average
    fallback over their (zero-weighted) top-k rather than NaN/zero."""

    sim = tanimoto_similarity(x_query, x_train)
    k = min(k, x_train.shape[0])
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)

    neighbor_labels = y_train[topk_idx]  # [Nq, k, C]
    weights = topk_sim.unsqueeze(-1)     # [Nq, k, 1]

    weight_sum = weights.sum(dim=1)      # [Nq, 1]
    weighted = (weights * neighbor_labels).sum(dim=1)  # [Nq, C]

    fallback = neighbor_labels.float().mean(dim=1)
    has_signal = weight_sum > 1e-8

    return torch.where(has_signal, weighted / weight_sum.clamp(min=1e-8), fallback)
