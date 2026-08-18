"""
train_full.py / evaluate_rgcn.py only ever report MICRO PR-AUC/ROC-AUC. For a
~4486-class, heavily imbalanced multi-label problem, micro-averaging is
dominated by whatever classes are most frequent -- a model that nails the
common ADE classes and is near-random on the long tail can still show a
deceptively good micro score. CLAUDE.md's own project history flags exactly
this class of problem, and the memory notes call out "per-class AUPR" as a
must-have.

CLAUDE.md also documents that sklearn's per-class average_precision_score,
looped over ~2000+ classes every epoch, was the actual training bottleneck
(>100s/epoch) -- not the GPU. This module's `vectorized_average_precision`
computes exact (matches sklearn to numerical precision) per-class AP on GPU
in one shot, so macro-AP no longer costs anything meaningful per epoch.
"""

import torch


def vectorized_average_precision(y_true, y_score):
    """Exact per-class average precision, computed for all classes at once.

    y_true, y_score : [N, C] tensors (N samples, C classes)
    returns         : [C] tensor of per-class AP (nan where a class has no
                      positive examples in this split -- caller should mask
                      those out of the macro average, same as sklearn does
                      implicitly when you skip that class).
    """

    N, C = y_true.shape
    device = y_true.device

    # Sort each column (class) by descending score.
    order = torch.argsort(y_score, dim=0, descending=True)
    y_true_sorted = torch.gather(y_true, 0, order)

    tp_cumsum = torch.cumsum(y_true_sorted, dim=0)
    rank = torch.arange(1, N + 1, device=device, dtype=torch.float32).unsqueeze(1)

    precision_at_k = tp_cumsum / rank
    total_pos = y_true.sum(dim=0)

    # AP = sum(precision_at_k * is_positive_at_k) / total_positives
    ap_numerator = (precision_at_k * y_true_sorted).sum(dim=0)

    ap = torch.where(
        total_pos > 0,
        ap_numerator / total_pos.clamp(min=1),
        torch.full_like(ap_numerator, float("nan")),
    )

    return ap


def macro_and_micro_ap(y_true, y_score, class_mask=None):
    """Returns (macro_ap, micro_ap) as Python floats.

    macro_ap : mean of per-class AP over classes that have >=1 positive
               (further restricted to `class_mask` if given)
    micro_ap : AP computed by flattening all (sample, class) pairs together
               (equivalent to sklearn's average='micro'), also restricted
               to `class_mask` if given.

    class_mask : optional bool tensor [C] -- e.g. from
        scripts/build_label_mask.py, marking which classes have enough
        positive examples across the whole dataset to be worth scoring.
        Nearly half of this project's ~4486 ADE classes have <10 total
        positive examples dataset-wide (confirmed via label-frequency scan),
        so macro-AP over the FULL label space is dominated by classes that
        are close to unlearnable from data volume alone. Restricting to a
        denser subset gives a much more honest signal of whether the model
        is actually learning, consistent with this project's own
        `build_mechanism_labels.py --min-freq` precedent on the other track.
    """

    if class_mask is not None:
        y_true = y_true[:, class_mask]
        y_score = y_score[:, class_mask]

    per_class_ap = vectorized_average_precision(y_true, y_score)
    valid = ~torch.isnan(per_class_ap)

    macro_ap = (
        per_class_ap[valid].mean().item() if valid.any() else float("nan")
    )

    micro_ap = vectorized_average_precision(
        y_true.reshape(-1, 1), y_score.reshape(-1, 1)
    )[0].item()

    return macro_ap, micro_ap
