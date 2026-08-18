"""
Loss functions for the multi-label ADE-class prediction head.

Neither train_rgcn.py nor train_full.py uses pos_weight at all, even though
scripts/compute_pos_weight.py already computes and saves one to
data/processed/pos_weight.pt -- it's just never loaded back in. Given the
~1:95-ish imbalance CLAUDE.md reports, plain BCEWithLogitsLoss() will mostly
learn to predict "no interaction" for everything and still get a low loss.

This module gives train_v2.py two options:
  - BCEWithLogitsLoss(pos_weight=...)   [wire up the already-computed weights]
  - FocalLossWithLogits                 [per memory: "focal loss and per-class
                                          AUPR recommended" for this imbalance]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Multi-label focal loss (Lin et al. 2017), applied independently per
    class/logit. Down-weights easy (well-classified) examples so rare
    positive classes aren't drowned out by the abundant easy negatives.
    """

    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma

        loss = focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(loss_type, pos_weight=None, focal_gamma=2.0):
    """loss_type: 'bce' or 'focal'."""

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        return FocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
