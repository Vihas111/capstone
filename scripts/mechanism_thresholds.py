"""
Shared per-label threshold-tuning logic for the mechanism gap-filler.
Importable module, not a CLI -- see scripts/tune_mechanism_thresholds.py for
the command-line wrapper.

WHY a 3-tier backoff instead of naive per-label tuning: measured directly
against this project's val split (389 drugs / 422 labels), 54/422 labels
have ZERO val positives and 203/422 have only 1-2. A naive per-label
F1-argmax would produce degenerate thresholds for ~95% of labels (e.g. a
single-positive label's "best F1" point is whatever score that one example
got, precision 1.0 by chance). Fix: only trust a per-label threshold when
that label has >= min_support val positives; otherwise fall back to a
threshold pooled across every label sharing the same "<kind>:" prefix
(enzyme/target/transporter), which has enough combined support to be
meaningful; if even that pool has zero positives (shouldn't happen given
this dataset's kind-pool sizes, but handled for safety), fall back to one
global-pooled threshold.

Two objectives, used for different purposes elsewhere in this pipeline:
  - "f1": maximize F1 -- used for the final DEPLOYED thresholds
    (scripts/tune_mechanism_thresholds.py --objective f1).
  - "precision_floor": the most INCLUSIVE (highest-recall) threshold that
    still keeps precision >= --precision-floor, or 1.01 (never fires) if no
    threshold clears the floor -- used as the GATE for accepting
    self-training pseudo-labels (see scripts/generate_mechanism_pseudo_labels.py),
    where a wrong guess becomes a hard-coded fake label, so it needs a
    stricter bar than F1-optimal.
"""

import torch


def _threshold_for_pool(scores, trues, objective, precision_floor):
    """scores, trues: 1D tensors already restricted to one label or one
    pooled kind-group. Returns a single float threshold. 1.01 means
    "never fires" (no score in [0,1] clears it) -- used when there's no
    positive-class signal at all, or (precision_floor objective) no
    threshold reaches the required precision."""

    if trues.sum() == 0:
        return 1.01

    order = torch.argsort(scores, descending=True)
    trues_sorted = trues[order]
    scores_sorted = scores[order]

    tp = torch.cumsum(trues_sorted, dim=0)
    fp = torch.cumsum(1 - trues_sorted, dim=0)
    precision = tp / (tp + fp)
    recall = tp / trues_sorted.sum()

    if objective == "f1":
        denom = (precision + recall).clamp(min=1e-9)
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(precision))
        best_idx = int(torch.argmax(f1))
        return float(scores_sorted[best_idx])

    elif objective == "precision_floor":
        meets = precision >= precision_floor
        if not bool(meets.any()):
            return 1.01
        idx = int(meets.nonzero(as_tuple=True)[0].max())  # most inclusive index meeting the floor
        return float(scores_sorted[idx])

    else:
        raise ValueError(f"Unknown objective: {objective!r}")


def tune_thresholds(y_true, y_score, label_vocab, objective="f1", min_support=10, precision_floor=0.5):
    """y_true, y_score: [N, C] tensors (typically the val split).
    label_vocab: list[str] of "<kind>:<protein>:<action>" labels, length C,
    same column order as y_true/y_score.

    Returns (thresholds, tiers):
      thresholds : [C] float tensor
      tiers      : list[str] of length C, "per-label" or "per-kind" --
                   diagnostic, for reporting how many labels got a real
                   per-label threshold vs. fell back to a pooled one.
    """

    num_labels = y_true.shape[1]
    kinds = [label.split(":", 1)[0] for label in label_vocab]

    kind_threshold = {}
    for kind in set(kinds):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        kind_threshold[kind] = _threshold_for_pool(
            y_score[:, idx].flatten(), y_true[:, idx].flatten(), objective, precision_floor
        )

    thresholds = torch.empty(num_labels)
    tiers = []

    for c in range(num_labels):
        support = int(y_true[:, c].sum())
        if support >= min_support:
            thresholds[c] = _threshold_for_pool(
                y_score[:, c], y_true[:, c], objective, precision_floor
            )
            tiers.append("per-label")
        else:
            thresholds[c] = kind_threshold[kinds[c]]
            tiers.append("per-kind")

    return thresholds, tiers
