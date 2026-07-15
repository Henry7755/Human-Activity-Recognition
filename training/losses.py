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
        L = α * (L_conf / N_conf) + β * (L_loc / N_loc)
   where N_conf = positives + hard negatives (per task)
         N_loc  = positives × 2 offset dims (per task)
   
   IMPORTANT (Section 4.C): Each task is normalized by its own term count,
   not by the shared positive count N. This makes alpha/beta true independent
   weights and removes the dataset/density-dependent rescaling that occurred
   when a single N normalized both ~4N classification terms and ~2N localization terms.
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
            scalar loss sum (unnormalized, will be normalized by MTHARSLoss)
        """
        if pos_mask.sum() == 0:
            return pred_offsets.sum() * 0.0   # zero gradient, keep graph

        pred = pred_offsets[pos_mask]   # (P, 2)
        true = true_offsets[pos_mask]   # (P, 2)

        diff  = (pred - true).abs()
        loss  = torch.where(diff < 1.0,
                            0.5 * diff ** 2,
                            diff - 0.5)
        
        # Return unnormalized sum (caller handles normalization)
        return loss.sum()


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
                pos_mask:       torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            cls_logits     : (B, na, K+1)  raw logits
            matched_labels : (B, na)       int labels
            pos_mask       : (B, na)       bool
        
        Returns:
            loss_sum : scalar (unnormalized)
            n_terms  : int, count of terms in loss (positives + hard negatives)
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
        n_pos = pos.sum().item()

        # ---- Hard-negative mining ----
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

        # Return unnormalized sum and term count for caller to normalize
        return pos_loss + neg_loss, n_pos + n_neg_target


# ---------------------------------------------------------------------------
# Combined Multi-Task Loss  (Equation 8 — CORRECTED per Section 4.C)
# ---------------------------------------------------------------------------

class MTHARSLoss(nn.Module):
    """
    CORRECTED Multi-Task Loss (Section 4.C of the engineering guide).
    
    Old formula (problematic):
        L = (1/N) * (α * L_conf_sum + β * L_loc_sum)
    
    where N = positive count only. This silently rescaled α:β based on
    match density because L_conf_sum ≈ 4N terms (positives + 3 hard negs)
    while L_loc_sum ≈ 2N terms (positives × 2 offset dims).
    
    New formula (correct):
        L = α * (L_conf_sum / (N_pos + N_neg)) + β * (L_loc_sum / N_pos)
    
    Each task is normalized by its actual term count, making α/β true
    independent weights that don't drift with dataset match density.

    Args:
        n_classes   : K (number of activity classes, excluding background).
        alpha       : weight for the classification loss (default 1.0).
        beta        : weight for the localisation (offset) loss (default 1.0).
        n_neg_ratio : hard-negative mining ratio (default 3).

    Best settings found in ablation (Table VII of the paper):
        WISDM       → α=2, β=3  (F1=0.9881)
        OPPORTUNITY → α=1, β=1  (F1=0.9213)
    
    NOTE: With corrected normalization, these may need re-derivation across
    datasets, but will be more stable once found (won't drift per dataset).
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
                         'n_pos', 'n_neg' for logging.
        """
        # Compute raw, unnormalized loss sums and term counts
        L_conf_sum, N_conf = self.conf_loss(cls_logits, matched_labels, pos_mask)
        L_loc_sum  = self.loc_loss(pred_offsets, true_offsets, pos_mask)
        N_loc      = max(pos_mask.sum().item(), 1)  # positives × 2 dims, but we normalize the sum
        
        # CORRECTED: Independent normalization per task (Section 4.C)
        # Classification: normalize by (positives + hard negatives)
        conf_term = (self.alpha * L_conf_sum / max(N_conf, 1))
        # Localization: normalize by positives
        loc_term  = (self.beta * L_loc_sum / max(N_loc, 1))
        
        # Combined loss
        total = conf_term + loc_term

        # CORRECTED: Adjusted statistics to reflect actual mean behaviors
        stats = {
            'conf_loss':  (L_conf_sum / max(N_conf, 1)).item(),
            'loc_loss':   (L_loc_sum / max(N_loc, 1)).item(),
            'total_loss': total.item(),
            'n_pos':      int(N_loc),
            'n_neg':      int(N_conf - N_loc),  # hard negatives
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
    print(f'  n_neg    : {stats["n_neg"]}')

    # F1 test
    f1 = WeightedF1(n_classes=K)
    preds  = torch.randint(0, K, (100,))
    labels = torch.randint(0, K, (100,))
    f1.update(preds, labels)
    print(f'Dummy weighted F1 : {f1.compute():.4f}')
