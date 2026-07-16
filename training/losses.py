"""
training/losses.py  (RECOGNITION-ONLY VARIANT)
================================================
Loss functions for recognition-only training.

CHANGED vs. the multi-task version
-----------------------------------
Implements §4.A step 1 from the coupling-map analysis:

  - `SmoothL1Loss1D` is DELETED. It only ever fed the localisation branch,
    which no longer exists.
  - `MTHARSLoss` (Eq. 8: L = (1/N)(α·L_conf + β·L_loc)) is replaced by
    `RecognitionLoss` (L = L_conf / N). `beta` and `L_loc` are gone;
    `alpha` is kept only as an optional scalar multiplier in case you still
    want a tunable classification-loss weight (e.g. for combining with an
    external regularizer later) — set it to 1.0 and it's a no-op.
  - `WeightedF1` is unchanged — it never touched offsets.

Kept unchanged
---------------
  - `ClassificationLoss` — identical to the multi-task version. Nothing
    about hard-negative mining or the cross-entropy term was coupled to
    segmentation; it's reused verbatim.
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NOTE: SmoothL1Loss1D intentionally DELETED here.
# It implemented Eq. (5)+(6) for the offset-regression branch, which does
# not exist in the recognition-only model. If you re-introduce segmentation
# later, restore it from the multi-task version of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Classification Loss  (Equation 7) — unchanged, robust boundary-armored version
# ---------------------------------------------------------------------------

class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss with hard-negative mining and index boundary armor.
    Identical to the multi-task version — never depended on offsets.
    """

    def __init__(self, n_neg_ratio: int = 3):
        super().__init__()
        self.n_neg_ratio = n_neg_ratio

    def forward(self,
                cls_logits:     torch.Tensor,
                matched_labels: torch.Tensor,
                pos_mask:       torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_logits     : (B, na, K+1)  raw logits
            matched_labels : (B, na)       int labels
            pos_mask       : (B, na)       bool
        """
        B, na, K_plus_1 = cls_logits.shape

        logits = cls_logits.reshape(-1, K_plus_1)
        labels = matched_labels.reshape(-1).clone()
        pos    = pos_mask.reshape(-1)

        labels[~pos] = 0
        max_valid_class_idx = K_plus_1 - 1
        labels = torch.clamp(labels, min=0, max=max_valid_class_idx)

        pos_loss = F.cross_entropy(logits[pos], labels[pos], reduction='sum')

        n_pos = pos.sum().item()
        n_neg_target = min(int(n_pos * self.n_neg_ratio),
                           int((~pos).sum().item()))

        if n_neg_target == 0:
            neg_loss = logits.sum() * 0.0
        else:
            neg_logits = logits[~pos]
            neg_targets = torch.zeros(neg_logits.shape[0],
                                      dtype=torch.long,
                                      device=logits.device)
            neg_ce = F.cross_entropy(neg_logits, neg_targets, reduction='none')

            _, topk_idx = neg_ce.topk(n_neg_target)
            hard_neg_logits = neg_logits[topk_idx]
            hard_neg_labels = torch.zeros(n_neg_target, dtype=torch.long,
                                          device=logits.device)
            neg_loss = F.cross_entropy(hard_neg_logits, hard_neg_labels,
                                       reduction='sum')

        return pos_loss + neg_loss


# ---------------------------------------------------------------------------
# Recognition-Only Loss  (replaces MTHARSLoss; drops β·L_loc entirely)
# ---------------------------------------------------------------------------

class RecognitionLoss(nn.Module):
    """
    L = alpha * L_conf / N

    Recognition-only replacement for Eq. (8). `beta`/`L_loc` are gone
    because there is no localisation branch to weight.

    Args:
        n_classes   : K (number of activity classes, excluding background).
        alpha       : optional scalar weight on the classification loss.
                      Default 1.0 (no-op) — kept for interface parity with
                      the multi-task config so existing CLI/ablation code
                      that sets `--alpha` doesn't need to be deleted, only
                      `--beta` becomes unused.
        n_neg_ratio : hard-negative mining ratio (default 3).
    """

    def __init__(self,
                 n_classes:   int,
                 alpha:       float = 1.0,
                 n_neg_ratio: int   = 3):
        super().__init__()
        self.alpha     = alpha
        self.conf_loss = ClassificationLoss(n_neg_ratio=n_neg_ratio)

    def forward(self,
                cls_logits:     torch.Tensor,
                matched_labels: torch.Tensor,
                pos_mask:       torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_logits     : (B, na, K+1)
            matched_labels : (B, na)  ground-truth class per window
            pos_mask       : (B, na)  bool

        Returns:
            total_loss : scalar
            stats      : dict with 'conf_loss', 'total_loss', 'n_pos'
        """
        L_conf_sum = self.conf_loss(cls_logits, matched_labels, pos_mask)
        N = max(pos_mask.sum().item(), 1)

        total = self.alpha * L_conf_sum / N

        stats = {
            'conf_loss':  (L_conf_sum / N).item(),
            'total_loss': total.item(),
            'n_pos':      int(N),
        }
        return total, stats


# ---------------------------------------------------------------------------
# Auxiliary: F1 metric (Equation 11) – used during evaluation, unchanged
# ---------------------------------------------------------------------------

class WeightedF1:
    """Weighted F1 score as defined in Eq. (11) of the paper. Unchanged."""

    def __init__(self, n_classes: int):
        self.n = n_classes
        self.reset()

    def reset(self):
        self.TP = torch.zeros(self.n)
        self.FP = torch.zeros(self.n)
        self.FN = torch.zeros(self.n)
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        for c in range(self.n):
            pred_c  = preds  == c
            label_c = labels == c
            self.TP[c] += (pred_c & label_c).sum().item()
            self.FP[c] += (pred_c & ~label_c).sum().item()
            self.FN[c] += (~pred_c & label_c).sum().item()
        self.total += len(labels)

    def compute(self) -> float:
        Nc     = self.TP + self.FN
        P_c    = self.TP / (self.TP + self.FP).clamp(min=1e-6)
        R_c    = self.TP / (self.TP + self.FN).clamp(min=1e-6)
        F1_c   = 2 * P_c * R_c / (P_c + R_c).clamp(min=1e-6)
        weight = Nc / max(Nc.sum().item(), 1)
        return (F1_c * weight).sum().item()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, na, K = 2, 64, 6

    cls_logits     = torch.randn(B, na, K + 1)
    matched_labels = torch.randint(0, K + 1, (B, na))
    pos_mask = torch.rand(B, na) > 0.80

    criterion = RecognitionLoss(n_classes=K, alpha=1.0)
    loss, stats = criterion(cls_logits, matched_labels, pos_mask)

    print(f'Total loss : {stats["total_loss"]:.4f}')
    print(f'  conf     : {stats["conf_loss"]:.4f}')
    print(f'  n_pos    : {stats["n_pos"]}')

    f1 = WeightedF1(n_classes=K)
    preds  = torch.randint(0, K, (100,))
    labels = torch.randint(0, K, (100,))
    f1.update(preds, labels)
    print(f'Dummy weighted F1 : {f1.compute():.4f}')
