"""
training/losses.py
==================
Loss functions for MTHARS training (Section III-E of the paper).

Components
----------
1. SmoothL1Loss1D
   – Eq. (5)+(6): element-wise Smooth-L1 for offset regression.

2. ClassificationLoss
   – Eq. (7): cross-entropy over matched positive + hard-negative windows.

3. MTHARSLoss
   – Eq. (8): combined multi-task loss
       L = (1/N) * (α * L_conf + β * L_loc)
   where N = number of matched (positive) windows in the batch.

Usage
-----
    criterion = MTHARSLoss(n_classes=6, alpha=1.0, beta=1.0)
    loss, stats = criterion(cls_logits, offsets,
                            matched_labels, matched_offsets, pos_mask)
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Smooth-L1 (Equations 5 & 6)
# ---------------------------------------------------------------------------

class SmoothL1Loss1D(nn.Module):
    """
    Smooth-L1 loss for the offset regression branch.

    SmoothL1(x) = 0.5 * x²   if |x| < 1
                  |x| - 0.5  otherwise

    Applied only to *positive* windows (those matched to a GT box).
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
            scalar loss
        """
        if pos_mask.sum() == 0:
            return pred_offsets.sum() * 0.0   # zero gradient, keep graph

        pred = pred_offsets[pos_mask]   # (P, 2)
        true = true_offsets[pos_mask]   # (P, 2)

        diff  = (pred - true).abs()
        loss  = torch.where(diff < 1.0,
                            0.5 * diff ** 2,
                            diff - 0.5)
        return loss.sum(dim=1).mean()   # mean over positives


# ---------------------------------------------------------------------------
# Classification Loss  (Equation 7)
# ---------------------------------------------------------------------------

class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss with hard-negative mining.

    Positives  : windows matched to a GT box (pos_mask == True).
    Hard-negatives : top-K background windows by their cross-entropy score,
                     with K = n_neg_ratio × n_pos.

    This implements the 3:1 negative-to-positive ratio described
    in Section III-E of the paper.

    Args:
        n_neg_ratio : negatives per positive (paper: 3).
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
            matched_labels : (B, na)       int labels (0 = background)
            pos_mask       : (B, na)       bool

        Returns:
            scalar loss
        """
        B, na, K_plus_1 = cls_logits.shape

        # Flatten over batch - FIXED: use actual B*na, not hardcoded 256
        logits = cls_logits.reshape(-1, K_plus_1)          # (B*na, K+1)
        labels = matched_labels.reshape(-1)                # (B*na,)
        pos    = pos_mask.reshape(-1)                      # (B*na,)

        # ---- Positive loss ----
        pos_loss = F.cross_entropy(logits[pos], labels[pos],
                                   reduction='sum')

        # ---- Hard-negative mining ----
    # Temporary print statements inside forward() in losses.py
        print("cls_logits shape:", cls_logits.shape)
        print("matched_labels min/max:", matched_labels.min().item(), matched_labels.max().item())
        print("pos shape:", pos.shape)
                   
        n_pos = pos.sum().item()
        n_neg_target = min(int(n_pos * self.n_neg_ratio),
                           int((~pos).sum().item()))

        if n_neg_target == 0:
            neg_loss = logits.sum() * 0.0
        else:
            # Score negatives by the CE loss w.r.t. background (class 0)
            neg_logits = logits[~pos]                  # (N_neg, K+1)
            neg_ce     = F.cross_entropy(neg_logits,
                                         torch.zeros(neg_logits.shape[0],
                                                     dtype=torch.long,
                                                     device=logits.device),
                                         reduction='none')   # (N_neg,)

            # Take the top-K hardest negatives
            _, topk_idx = neg_ce.topk(n_neg_target)
            hard_neg_logits = neg_logits[topk_idx]
            hard_neg_labels = torch.zeros(n_neg_target, dtype=torch.long,
                                          device=logits.device)
            neg_loss = F.cross_entropy(hard_neg_logits, hard_neg_labels,
                                       reduction='sum')

        N = max(int(n_pos), 1)
        return (pos_loss + neg_loss) / N


# ---------------------------------------------------------------------------
# Combined Multi-Task Loss  (Equation 8)
# ---------------------------------------------------------------------------

class MTHARSLoss(nn.Module):
    """
    L = (1/N) * (α * L_conf + β * L_loc)

    Args:
        n_classes   : K (number of activity classes, excluding background).
        alpha       : weight for the classification loss (default 1.0).
        beta        : weight for the localisation (offset) loss (default 1.0).
        n_neg_ratio : hard-negative mining ratio (default 3).

    Best settings found in ablation (Table VII of the paper):
        WISDM      → α=2, β=3  (F1=0.9881)
        OPPORTUNITY → α=1, β=1  (F1=0.9213)
    """

    def __init__(self,
                 n_classes:   int,
                 alpha:       float = 1.0,
                 beta:        float = 1.0,
                 n_neg_ratio: int   = 3):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.loc_loss  = SmoothL1Loss1D()
        self.conf_loss = ClassificationLoss(n_neg_ratio=n_neg_ratio)

    def forward(self,
                cls_logits:     torch.Tensor,
                pred_offsets:   torch.Tensor,
                matched_labels: torch.Tensor,
                true_offsets:   torch.Tensor,
                pos_mask:       torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_logits     : (B, na, K+1)
            pred_offsets   : (B, na, 2)
            matched_labels : (B, na)      ground-truth class per window
            true_offsets   : (B, na, 2)   encoded ground-truth offsets
            pos_mask       : (B, na) bool

        Returns:
            total_loss : scalar
            stats      : dict with 'conf_loss', 'loc_loss', 'total_loss',
                         'n_pos' for logging.
        """
        L_conf = self.conf_loss(cls_logits, matched_labels, pos_mask)
        L_loc  = self.loc_loss(pred_offsets, true_offsets, pos_mask)

        N     = max(pos_mask.sum().item(), 1)
        total = (self.alpha * L_conf + self.beta * L_loc) / N

        stats = {
            'conf_loss':  L_conf.item(),
            'loc_loss':   L_loc.item(),
            'total_loss': total.item(),
            'n_pos':      int(N),
        }
        return total, stats


# ---------------------------------------------------------------------------
# Auxiliary: F1 metric (Equation 11) – used during evaluation
# ---------------------------------------------------------------------------

class WeightedF1:
    """
    Weighted F1 score as defined in Eq. (11) of the paper.

    Accumulates predictions across batches and computes the final
    macro-weighted F1 from true positives, false positives, and
    false negatives per class.
    """

    def __init__(self, n_classes: int):
        self.n = n_classes
        self.reset()

    def reset(self):
        self.TP = torch.zeros(self.n)
        self.FP = torch.zeros(self.n)
        self.FN = torch.zeros(self.n)
        self.total = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            preds  : (N,) int64 – predicted class indices (0-indexed)
            labels : (N,) int64 – true class indices
        """
        for c in range(self.n):
            pred_c  = preds  == c
            label_c = labels == c
            self.TP[c] += (pred_c & label_c).sum().item()
            self.FP[c] += (pred_c & ~label_c).sum().item()
            self.FN[c] += (~pred_c & label_c).sum().item()
        self.total += len(labels)

    def compute(self) -> float:
        """Return weighted F1 (Equation 11)."""
        Nc     = self.TP + self.FN   # true count per class
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
    pred_offsets   = torch.randn(B, na, 2)
    matched_labels = torch.randint(0, K + 1, (B, na))
    true_offsets   = torch.randn(B, na, 2)

    # ~20 % positive windows
    pos_mask = torch.rand(B, na) > 0.80

    criterion = MTHARSLoss(n_classes=K, alpha=1.0, beta=1.0)
    loss, stats = criterion(cls_logits, pred_offsets,
                            matched_labels, true_offsets, pos_mask)

    print(f'Total loss : {stats["total_loss"]:.4f}')
    print(f'  conf     : {stats["conf_loss"]:.4f}')
    print(f'  loc      : {stats["loc_loss"]:.4f}')
    print(f'  n_pos    : {stats["n_pos"]}')

    # F1 test
    f1 = WeightedF1(n_classes=K)
    preds  = torch.randint(0, K, (100,))
    labels = torch.randint(0, K, (100,))
    f1.update(preds, labels)
    print(f'Dummy weighted F1 : {f1.compute():.4f}')
