"""
training/losses.py  (SEGMENTATION-ONLY VARIANT)
=================================================
Loss functions for segmentation-only training.

CHANGED vs. the multi-task version
-----------------------------------
Implements §4.B step 1 from the coupling-map analysis, with one
adaptation flagged explicitly below:

  - `ClassificationLoss` is DELETED, along with `n_neg_ratio`-based
    hard-negative mining — there are no classes left to mine hard
    negatives against.
  - `WeightedF1` is DELETED — it's a classification metric with no
    meaning once there's no class output.
  - `SmoothL1Loss1D` is KEPT UNCHANGED — segmentation-only still regresses
    boundaries.

  ADAPTATION (flagged): the analysis doc's §4.B step 1 says "delete
  ClassificationLoss usage... keep only SmoothL1Loss1D," but step 3 of the
  same section adds a binary "objectness" head specifically so NMS has a
  ranking signal. Read literally, following step 1 alone would leave that
  objectness head with no training signal at all (dead weights, useless
  NMS ranking). This file resolves that by introducing a new
  `ObjectnessLoss` (binary cross-entropy against `pos_mask`) as a
  lightweight replacement for the old per-class term — NOT a
  reintroduction of `ClassificationLoss`. It carries no `n_neg_ratio` /
  top-k mining; foreground/background imbalance is instead handled via
  `pos_weight` in `BCEWithLogitsLoss`, which is a standard numerical
  technique for class imbalance, not negative mining.
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Smooth-L1 (Equations 5 & 6) — UNCHANGED, still the primary loss term
# ---------------------------------------------------------------------------

class SmoothL1Loss1D(nn.Module):
    """
    Smooth-L1 loss for the offset regression branch.

    SmoothL1(x) = 0.5 * x²   if |x| < 1
                  |x| - 0.5  otherwise

    Applied only to *positive* (foreground) windows.
    """

    def forward(self,
                pred_offsets: torch.Tensor,
                true_offsets: torch.Tensor,
                pos_mask:     torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_offsets : (B, na, 2)  predicted (f_x, f_l)
            true_offsets : (B, na, 2)  encoded ground-truth offsets
            pos_mask     : (B, na)     bool, True for positive windows

        Returns:
            scalar loss sum (un-normalized — caller divides by N)
        """
        if pos_mask.sum() == 0:
            return pred_offsets.sum() * 0.0

        pred = pred_offsets[pos_mask]
        true = true_offsets[pos_mask]

        diff = (pred - true).abs()
        loss = torch.where(diff < 1.0,
                           0.5 * diff ** 2,
                           diff - 0.5)
        return loss.sum()


# ---------------------------------------------------------------------------
# Objectness Loss  (NEW — replaces ClassificationLoss, trains obj_branch)
# ---------------------------------------------------------------------------

class ObjectnessLoss(nn.Module):
    """
    Binary cross-entropy over the foreground/background objectness score.

    Unlike the deleted `ClassificationLoss`, there's no per-class term and
    no top-k hard-negative selection — every window contributes to the
    loss, with foreground/background imbalance handled by `pos_weight`
    rather than explicit mining.

    Args:
        pos_weight : weight applied to the positive (foreground) class in
                    `BCEWithLogitsLoss`, to counteract the natural
                    imbalance where background windows vastly outnumber
                    foreground ones. Tune this per-dataset (rough starting
                    point: (#background windows / #foreground windows) in
                    a representative batch).
    """

    def __init__(self, pos_weight: float = 3.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(pos_weight))

    def forward(self,
                obj_logits: torch.Tensor,
                pos_mask:   torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_logits : (B, na)  raw objectness logits
            pos_mask   : (B, na)  bool — doubles as the binary target
                        (this is the same `pos_mask` WindowMatcher.match
                        already computes for offset supervision; no
                        separate label tensor is needed)

        Returns:
            scalar loss sum (un-normalized — caller divides by N)
        """
        logits = obj_logits.reshape(-1)
        target = pos_mask.reshape(-1).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=self.pos_weight.to(logits.device),
            reduction='sum'
        )
        return loss


# ---------------------------------------------------------------------------
# NOTE: ClassificationLoss and WeightedF1 intentionally DELETED here.
# If you re-introduce recognition later, restore them from the multi-task
# version of this file rather than re-deriving them.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Combined Segmentation-Only Loss  (replaces MTHARSLoss)
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    L = (beta * L_loc + gamma * L_obj) / N

    Segmentation-only replacement for Eq. (8). `alpha`/`L_conf` are gone —
    there is no classification term left to weight.

    Args:
        beta        : weight for the localisation (offset) loss.
        gamma       : weight for the objectness loss.
        pos_weight  : passed through to ObjectnessLoss (see above).
    """

    def __init__(self,
                 beta:       float = 1.0,
                 gamma:      float = 1.0,
                 pos_weight: float = 3.0):
        super().__init__()
        self.beta  = beta
        self.gamma = gamma
        self.loc_loss = SmoothL1Loss1D()
        self.obj_loss = ObjectnessLoss(pos_weight=pos_weight)

    def forward(self,
                pred_offsets: torch.Tensor,
                obj_logits:   torch.Tensor,
                true_offsets: torch.Tensor,
                pos_mask:     torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_offsets : (B, na, 2)
            obj_logits   : (B, na)
            true_offsets : (B, na, 2)   encoded ground-truth offsets
            pos_mask     : (B, na) bool

        Returns:
            total_loss : scalar
            stats      : dict with 'loc_loss', 'obj_loss', 'total_loss',
                         'n_pos' for logging.
        """
        L_loc_sum = self.loc_loss(pred_offsets, true_offsets, pos_mask)
        L_obj_sum = self.obj_loss(obj_logits, pos_mask)

        N = max(pos_mask.sum().item(), 1)
        N_total = pos_mask.numel()

        total = (self.beta * L_loc_sum / N) + (self.gamma * L_obj_sum / N_total)

        stats = {
            'loc_loss':   (L_loc_sum / N).item(),
            'obj_loss':   (L_obj_sum / N_total).item(),
            'total_loss': total.item(),
            'n_pos':      int(N),
        }
        return total, stats


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, na = 2, 64

    pred_offsets = torch.randn(B, na, 2)
    obj_logits   = torch.randn(B, na)
    true_offsets = torch.randn(B, na, 2)

    pos_mask = torch.rand(B, na) > 0.80   # ~20% positive

    criterion = SegmentationLoss(beta=1.0, gamma=1.0, pos_weight=4.0)
    loss, stats = criterion(pred_offsets, obj_logits, true_offsets, pos_mask)

    print(f'Total loss : {stats["total_loss"]:.4f}')
    print(f'  loc      : {stats["loc_loss"]:.4f}')
    print(f'  obj      : {stats["obj_loss"]:.4f}')
    print(f'  n_pos    : {stats["n_pos"]}')
