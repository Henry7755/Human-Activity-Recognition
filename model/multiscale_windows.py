"""
model/multiscale_windows.py
============================
Section III-B of the paper: Multiscale Window Generation and Matching.

Components
----------
1. WindowGenerator
   – generates windows of multiple scales centered on each unit of
     the backbone feature sequence (mirrors the anchor concept from SSD).

2. iou_1d
   – Jaccard index (Intersection-over-Union) for 1-D intervals.

3. WindowMatcher
   – assigns each generated window to its closest ground-truth
     activity bounding box using the greedy Hungarian-style algorithm
     described in the paper (Section III-B, "Multiscale window labeling
     and matching").

4. offset_encode / offset_decode
   – Equations (1)-(4): convert between absolute boundaries and the
     (center-offset, log-length-offset) parameterisation used during
     training and inference.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict

import torch
import numpy as np


# ---------------------------------------------------------------------------
# 1-D Interval IOU
# ---------------------------------------------------------------------------

def iou_1d(windows: torch.Tensor,
           gt_box: torch.Tensor) -> torch.Tensor:
    """
    Compute IOU between a set of windows and a single ground-truth box.

    Args:
        windows : (N, 2)  – each row [center_x, length] in absolute coords
        gt_box  : (2,)    – [center_x, length] of the truth boundary

    Returns:
        iou : (N,) float tensor in [0, 1]
    """
    # Convert center+length to start/end
    w_start = windows[:, 0] - windows[:, 1] / 2
    w_end   = windows[:, 0] + windows[:, 1] / 2

    g_start = gt_box[0] - gt_box[1] / 2
    g_end   = gt_box[0] + gt_box[1] / 2

    inter_start = torch.max(w_start, g_start)
    inter_end   = torch.min(w_end,   g_end)
    inter       = (inter_end - inter_start).clamp(min=0)

    union = windows[:, 1] + gt_box[1] - inter
    return inter / union.clamp(min=1e-6)


def iou_matrix(windows: torch.Tensor,
               gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute full IOU matrix between all windows and all GT boxes.

    Args:
        windows  : (na, 2)  windows in (center, length) form
        gt_boxes : (nb, 2)  GT boxes in (center, length) form

    Returns:
        M : (na, nb) IOU matrix
    """
    na = windows.shape[0]
    nb = gt_boxes.shape[0]
    M = torch.zeros(na, nb, device=windows.device, dtype=windows.dtype)
    for j in range(nb):
        M[:, j] = iou_1d(windows, gt_boxes[j])
    return M


# ---------------------------------------------------------------------------
# Offset encoding / decoding  (Equations 1-4)
# ---------------------------------------------------------------------------

def offset_encode(windows: torch.Tensor,
                  gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Encode ground-truth boxes relative to matched windows.

    Implements Equations (1) and (2):
        f_x = (t_x - w_x) / w_l
        f_l = log(t_l / w_l)

    Args:
        windows  : (N, 2)  matched windows [center, length]
        gt_boxes : (N, 2)  matched GT boxes [center, length]

    Returns:
        offsets : (N, 2)  [f_x, f_l]
    """
    f_x = (gt_boxes[:, 0] - windows[:, 0]) / windows[:, 1].clamp(min=1e-6)
    f_l = torch.log(gt_boxes[:, 1] / windows[:, 1].clamp(min=1e-6))
    return torch.stack([f_x, f_l], dim=1)


def offset_decode(windows: torch.Tensor,
                  offsets: torch.Tensor) -> torch.Tensor:
    """
    Decode predicted offsets back to absolute boundaries.

    Implements Equations (3) and (4):
        t̂_x = f_x * w_l + w_x
        t̂_l = w_l * exp(f_l)

    Args:
        windows : (N, 2)  anchor windows [center, length]
        offsets : (N, 2)  predicted offsets [f_x, f_l]

    Returns:
        pred_boxes : (N, 2)  predicted boundaries [center, length]
    """
    pred_x = offsets[:, 0] * windows[:, 1] + windows[:, 0]
    pred_l = windows[:, 1] * torch.exp(offsets[:, 1])
    return torch.stack([pred_x, pred_l], dim=1)


# ---------------------------------------------------------------------------
# Window Generator
# ---------------------------------------------------------------------------

class WindowGenerator:
    """
    Generate multiscale anchor windows for a 1-D feature sequence.

    Corresponds to Section III-B: "Generation of windows."

    The feature sequence has length  n_feat  (output length of the backbone).
    For each scale  s  in  self.scales, two window lengths are created:
        l1 = n_feat * sqrt(s)
        l2 = n_feat / sqrt(s)
    Windows are centered on each unit  x  of the feature sequence.

    The absolute window centers are mapped to [0, n_feat) using relative
    positions so they can be scaled to any raw-data stream length.

    Args:
        scales     : list of scale values s ∈ (0, 1]
        feat_len   : length of the backbone feature sequence (n_feat)
        data_len   : length of the original raw activity stream (n)
    """

    def __init__(self, scales: List[float],
                 feat_len: int,
                 data_len: int):
        self.scales   = scales
        self.feat_len = feat_len
        self.data_len = data_len
        self.windows  = self._generate()   # (na, 2) in absolute data coords

    def _generate(self) -> torch.Tensor:
        """
        Returns (na, 2) tensor where each row is [center_x, length]
        expressed in raw-data coordinate space.
        """
        ratio = self.data_len / self.feat_len   # maps feature → data coords

        window_list = []
        for x in range(self.feat_len):              # center on each feature unit
            center_data = (x + 0.5) * ratio        # center in data coords

            for s in self.scales:
                sqrts = math.sqrt(s)
                for length_data in [self.feat_len * sqrts * ratio,
                                    self.feat_len / sqrts * ratio]:
                    window_list.append([center_data, length_data])

        return torch.tensor(window_list, dtype=torch.float32)   # (na, 2)

    @property
    def num_windows(self) -> int:
        return self.windows.shape[0]

    def to_start_end(self) -> torch.Tensor:
        """
        Convert (center, length) → (start, end) in data coordinates.
        Returns (na, 2).
        """
        centers = self.windows[:, 0]
        lengths = self.windows[:, 1]
        starts  = (centers - lengths / 2).clamp(min=0)
        ends    = (centers + lengths / 2).clamp(max=self.data_len - 1)
        return torch.stack([starts, ends], dim=1)


# ---------------------------------------------------------------------------
# Window Matcher
# ---------------------------------------------------------------------------

class WindowMatcher:
    """
    Match generated windows to ground-truth activity bounding boxes.

    Implements the greedy assignment algorithm described in Section III-B
    ("Multiscale window labeling and matching"):

    Step 1: Find the best window for every GT box (highest IOU).
    Step 2: For remaining unmatched windows, assign the GT box with the
            highest IOU if that IOU exceeds `pos_iou_thresh`.
    Step 3: Windows with IOU < neg_iou_thresh for ALL GT boxes become
            background (class 0).

    Args:
        pos_iou_thresh : minimum IOU to label a window as positive
        neg_iou_thresh : maximum IOU for a window to be labelled background
        n_neg_ratio    : ratio of negatives to positives for hard-negative
                         mining during training (paper uses 3:1)
    """

    def __init__(self,
                 pos_iou_thresh: float = 0.5,
                 neg_iou_thresh: float = 0.3,
                 n_neg_ratio:    int   = 3):
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.n_neg_ratio    = n_neg_ratio

    def match(self,
              windows:  torch.Tensor,
              gt_boxes: torch.Tensor,
              gt_labels: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            windows   : (na, 2)  anchor windows [center, length]
            gt_boxes  : (nb, 2)  GT bounding boxes [center, length]
            gt_labels : (nb,)    GT activity class labels (1-indexed;
                                 0 is reserved for background)

        Returns:
            matched_labels  : (na,)    class label per window (0=background)
            matched_offsets : (na, 2)  encoded offsets (valid for positives)
            pos_mask        : (na,)    bool mask of positive windows
        """
        na = windows.shape[0]
        nb = gt_boxes.shape[0]
        device = windows.device

        matched_labels  = torch.zeros(na, dtype=torch.long, device=device)
        matched_offsets = torch.zeros(na, 2, dtype=torch.float32, device=device)
        assigned_gt     = torch.full((na,), -1, dtype=torch.long, device=device)

        if nb == 0:
            return (matched_labels, matched_offsets,
                    torch.zeros(na, dtype=torch.bool, device=device))

        # Build IOU matrix  M ∈ R^{na × nb}
        M = iou_matrix(windows, gt_boxes)    # (na, nb)
        M_work = M.clone()

        # ---- Step 1: guarantee every GT box gets its best window ----
        for _ in range(nb):
            best = M_work.max()
            if best.item() == 0:
                break
            flat_idx = M_work.argmax()
            i = flat_idx // nb      # window index
            j = flat_idx %  nb      # GT box index
            assigned_gt[i] = j
            M_work[i, :] = -1       # discard row
            M_work[:, j] = -1       # discard column

        # ---- Step 2: match remaining windows by threshold (vectorized) ----
        unassigned = assigned_gt < 0
        best_iou, best_j = M.max(dim=1)

        take_pos = unassigned & (best_iou >= self.pos_iou_thresh)
        take_ignore = (unassigned
                       & (best_iou >= self.neg_iou_thresh)
                       & (best_iou < self.pos_iou_thresh))

        assigned_gt = torch.where(take_pos, best_j, assigned_gt)
        assigned_gt = torch.where(
            take_ignore, torch.full_like(assigned_gt, -2), assigned_gt
        )

        # ---- Step 3: encode labels and offsets (vectorized) ----
        pos_mask = assigned_gt >= 0
        pos_idx  = pos_mask.nonzero(as_tuple=True)[0]
        j_idx    = assigned_gt[pos_idx]
        matched_labels[pos_idx]  = gt_labels[j_idx]
        matched_offsets[pos_idx] = offset_encode(windows[pos_idx], gt_boxes[j_idx])

        return matched_labels, matched_offsets, pos_mask

    def hard_negative_mining(self,
                             class_probs: torch.Tensor,
                             pos_mask:    torch.Tensor
                             ) -> torch.Tensor:
        """
        Select hard negatives so that neg : pos ≈ n_neg_ratio : 1.

        Args:
            class_probs : (na, K+1)  softmax class probabilities
            pos_mask    : (na,)      bool mask of positive windows

        Returns:
            neg_mask : (na,) bool mask of selected hard negatives
        """
        n_pos = pos_mask.sum().item()
        n_neg_target = min(int(n_pos * self.n_neg_ratio),
                           (~pos_mask).sum().item())

        # Confidence loss for each negative (higher loss → harder)
        bg_prob  = class_probs[:, 0]                        # P(background)
        neg_loss = -torch.log(bg_prob.clamp(min=1e-7))      # cross-entropy

        # Zero out positives so they don't get selected
        neg_loss = neg_loss.masked_fill(pos_mask, -1.0)

        _, sorted_idx = neg_loss.sort(descending=True)
        neg_mask = torch.zeros_like(pos_mask)
        neg_mask[sorted_idx[:n_neg_target]] = True
        return neg_mask


# ---------------------------------------------------------------------------
# Quick functional test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Simulate a feature sequence of length 8 from a 64-sample data stream
    feat_len = 8
    data_len = 64
    scales   = [0.5, 1.0]

    gen = WindowGenerator(scales=scales, feat_len=feat_len, data_len=data_len)
    print(f'Generated {gen.num_windows} windows for feat_len={feat_len}, '
          f'data_len={data_len}, scales={scales}')
    print('First 4 windows (center, length):', gen.windows[:4])

    # Simulate 2 GT boxes
    gt_boxes  = torch.tensor([[15.0, 20.0],
                               [45.0, 10.0]])
    gt_labels = torch.tensor([1, 2])

    matcher = WindowMatcher(pos_iou_thresh=0.4)
    lbl, off, pos = matcher.match(gen.windows, gt_boxes, gt_labels)

    print(f'Positive windows  : {pos.sum().item()}')
    print(f'Background windows: {(lbl == 0).sum().item()}')

    # Test encode → decode round-trip
    pos_idx = pos.nonzero(as_tuple=True)[0]
    if len(pos_idx):
        w = gen.windows[pos_idx]
        o = off[pos_idx]
        recovered = offset_decode(w, o)
        print('Offset encode→decode round-trip error:',
              (recovered - gt_boxes[lbl[pos_idx] - 1]).abs().max().item())

    # Device-mismatch regression test
    if torch.cuda.is_available():
        gen_cuda = gen.windows.cuda()
        gt_boxes_cuda = gt_boxes.cuda()
        gt_labels_cuda = gt_labels.cuda()
        lbl_c, off_c, pos_c = matcher.match(gen_cuda, gt_boxes_cuda, gt_labels_cuda)
        print('CUDA match() succeeded, device:', lbl_c.device)
