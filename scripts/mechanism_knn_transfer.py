"""
Similarity-weighted k-NN mechanism-label transfer -- findings/findings.md
item 2 ("k-NN similarity transfer, as an ensemble member, not a
replacement"). This is the never-built similarity_gapfiller.py's actual
job, but framed from the start as an ensemble component to blend with
scripts/train_mechanism_predictor.py's linear model, not a standalone
competitor (per the doc's own correction to its original framing).

For a query drug, finds its K nearest TRAIN-split neighbors by similarity
and predicts each label's probability as the similarity-weighted fraction
of those neighbors that have that label. Fully vectorized (one similarity
matrix, one topk, one gather) -- no per-drug Python loop.

Two similarity metrics, both feeding the same vote logic:
  - tanimoto_similarity: for binary Morgan/ECFP fingerprints (intersection
    over union of the "on" bits) -- the original item 2 result.
  - cosine_similarity: for continuous embeddings (e.g. ChemBERTa, see
    scripts/build_chemberta_embeddings.py / scripts/run_mechanism_embedding_kfold.py)
    -- Tanimoto isn't defined for non-binary vectors, cosine is the
    standard substitute for dense embedding spaces.

Usage (as a library):
    from scripts.mechanism_knn_transfer import knn_transfer_probs, tanimoto_similarity, cosine_similarity
    probs = knn_transfer_probs(x_query, x_train, y_train, k=20, similarity_fn=tanimoto_similarity)
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


def cosine_similarity(x_query, x_train):
    """x_query: [Nq, D] continuous. x_train: [Nt, D] continuous. Returns
    [Nq, Nt] cosine similarity -- the standard metric for dense embeddings
    (e.g. ChemBERTa), where Tanimoto (defined for binary vectors) doesn't
    apply. Clamped to >=0 so it plugs into knn_transfer_probs's weighted
    vote the same way Tanimoto similarity does (always non-negative)."""

    q_norm = torch.nn.functional.normalize(x_query, dim=1)
    t_norm = torch.nn.functional.normalize(x_train, dim=1)
    return (q_norm @ t_norm.T).clamp(min=0.0)


def knn_transfer_probs(x_query, x_train, y_train, k=20, similarity_fn=tanimoto_similarity):
    """Returns [Nq, C] predicted label probabilities: for each query drug,
    the similarity-weighted vote of its k most similar train drugs' binary
    labels (similarity per similarity_fn -- tanimoto_similarity for binary
    fingerprints, cosine_similarity for continuous embeddings). Query drugs
    with zero similarity to every train drug get a uniform-average fallback
    over their (zero-weighted) top-k rather than NaN/zero."""

    sim = similarity_fn(x_query, x_train)
    k = min(k, x_train.shape[0])
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)

    neighbor_labels = y_train[topk_idx]  # [Nq, k, C]
    weights = topk_sim.unsqueeze(-1)     # [Nq, k, 1]

    weight_sum = weights.sum(dim=1)      # [Nq, 1]
    weighted = (weights * neighbor_labels).sum(dim=1)  # [Nq, C]

    fallback = neighbor_labels.float().mean(dim=1)
    has_signal = weight_sum > 1e-8

    return torch.where(has_signal, weighted / weight_sum.clamp(min=1e-8), fallback)
