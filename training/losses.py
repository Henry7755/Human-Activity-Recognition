"""
training/losses.py
==================
Loss functions for MTHARS training (Recognition-only variant).

Components
----------
1. ClassificationLoss
   – Eq. (7): cross-entropy over matched positive + hard-negative windows.

2. MTHARSLoss (Refactored)
   – Recognition-only loss: L = (1/N) * L_conf
   – Removed all localization (offset regression) components.
   – Simplified to pure classification task.
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Classification Loss  (Equation 7) - Robust Kaggle/Remote Version
# ---------------------------------------------------------------------------

class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss with hard-negative mining and index boundary armor.
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

        # Flatten over batch dimensions for vectorized processing
        logits = cls_logits.reshape(-1, K_plus_1)          # (B*na, K+1)
        labels = matched_labels.reshape(-1).clone()        # (B*na,)
        pos    = pos_mask.reshape(-1)                      # (B*na,)

        # ---- THE KAGGLE BOUNDARY ARMOR ----
        # Force all background/negative tokens to exactly 0
        labels[~pos] = 0
        
        # Guard: If any positive label falls completely outside the model's 
        # classification capacity, clip it to the maximum available valid index.
        # This prevents the CUDA device-side assertion crash immediately.
        max_valid_class_idx = K_plus_1 - 1
        labels = torch.clamp(labels, min=0, max=max_valid_class_idx)

        # ---- Positive loss ----
        pos_loss = F.cross_entropy(logits[pos], labels[pos], reduction='sum')

        # ---- Hard-negative mining ----
        n_pos = pos.sum().item()
        n_neg_target = min(int(n_pos * self.n_neg_ratio),
                           int((~pos).sum().item()))

        if n_neg_target == 0:
            neg_loss = logits.sum() * 0.0
        else:
            neg_logits = logits[~pos]                  # (N_neg, K+1)
            
            # Target for negatives is strictly background (class 0)
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
# Combined Classification-Only Loss (Refactored)
# ---------------------------------------------------------------------------

class MTHARSLoss(nn.Module):
    """
    Recognition-only loss (simplified from original multi-task).
    
    L = (1/N) * L_conf
    
    where N = number of matched (positive) windows in the batch.

    Args:
        n_classes   : K (number of activity classes, excluding background).
        n_neg_ratio : hard-negative mining ratio (default 3).
    
    Note: alpha and beta parameters removed (no longer used).
    """

    def __init__(self,
                 n_classes:   int,
                 n_neg_ratio: int = 3,
                 **kwargs):  # Accept but ignore alpha, beta for backward compatibility
        super().__init__()
        self.conf_loss = ClassificationLoss(n_neg_ratio=n_neg_ratio)

    def forward(self,
                cls_logits:     torch.Tensor,
                matched_labels: torch.Tensor,
                pos_mask:       torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_logits     : (B, na, K+1)
            matched_labels : (B, na)      ground-truth class per window
            pos_mask       : (B, na) bool

        Returns:
            total_loss : scalar
            stats      : dict with 'conf_loss', 'total_loss', 'n_pos' for logging.
        """
        # Compute classification loss (only loss component)
        L_conf_sum = self.conf_loss(cls_logits, matched_labels, pos_mask)

        N = max(pos_mask.sum().item(), 1)
        
        # Total loss is simply normalized classification loss
        total = L_conf_sum / N

        # Stats dictionary: only classification metrics
        stats = {
            'conf_loss':  (L_conf_sum / N).item(),
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
    matched_labels = torch.randint(0, K + 1, (B, na))

    # ~20 % positive windows
    pos_mask = torch.rand(B, na) > 0.80

    criterion = MTHARSLoss(n_classes=K)
    loss, stats = criterion(cls_logits, matched_labels, pos_mask)

    print(f'Total loss : {stats["total_loss"]:.4f}')
    print(f'  conf     : {stats["conf_loss"]:.4f}')
    print(f'  n_pos    : {stats["n_pos"]}')

    # F1 test
    f1 = WeightedF1(n_classes=K)
    preds  = torch.randint(0, K, (100,))
    labels = torch.randint(0, K, (100,))
    f1.update(preds, labels)
    print(f'Dummy weighted F1 : {f1.compute():.4f}')
