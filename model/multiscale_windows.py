"""
model/multiscale_windows.py  (RECOGNITION-ONLY VARIANT)
=========================================================
Section III-B of the paper: Multiscale Window Generation and Matching.

CHANGED vs. the multi-task version
-----------------------------------
This file implements §4.A ("Recognition-only conversion") steps 1-2 from
the coupling-map analysis:

  - `offset_encode` / `offset_decode` (Equations 1-4) are REMOVED. They
    existed only to parameterise ground-truth boundaries relative to a
    matched window for the localisation branch; a recognition-only model
    never regresses boundaries, so these targets are never consumed.

  - `WindowMatcher.match()` no longer computes or returns
    `matched_offsets`. It still needs `gt_boxes` to know WHICH window is
    closest to WHICH activity instance (that's what makes a window
    "positive" vs "background"), but it no longer builds the
    (f_x, f_l) regression target for that assignment.

  - Return signature of `match()` changes from
        (matched_labels, matched_offsets, pos_mask)
    to
        (matched_labels, pos_mask)
    Any caller unpacking 3 values (e.g. `training/trainer.py`'s
    `prepare_targets`) must be updated to unpack 2 — see the
    recognition-only `trainer.py` in this same delivery.

Components retained (still needed for recognition)
----------------------------------------------------
1. WindowGenerator   – still generates the multiscale anchor set; a window
   still needs a candidate boundary to be scored against GT boxes via IOU
   even though the network will never learn to move that boundary.
2. iou_1d / iou_matrix – still the matching criterion.
3. WindowMatcher.match – trimmed to labels + pos_mask only.
4. WindowMatcher.hard_negative_mining – unchanged; feeds ClassificationLoss.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch


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
# NOTE: offset_encode / offset_decode intentionally DELETED here.
# They lived in the multi-task file to implement Equations (1)-(4) for the
# localisation branch. Recognition-only training never regresses a boundary,
# so there is nothing to encode a target for, and nothing to decode at
# inference. If you re-introduce segmentation later, restore them from the
# multi-task version of this file rather than re-deriving them.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Window Generator  (unchanged from multi-task version)
# ---------------------------------------------------------------------------

class WindowGenerator:
    """
    Generate multiscale anchor windows for a 1-D feature sequence.

    Corresponds to Section III-B: "Generation of windows."

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
        """Convert (center, length) → (start, end) in data coordinates."""
        centers = self.windows[:, 0]
        lengths = self.windows[:, 1]
        starts  = (centers - lengths / 2).clamp(min=0)
        ends    = (centers + lengths / 2).clamp(max=self.data_len - 1)
        return torch.stack([starts, ends], dim=1)


# ---------------------------------------------------------------------------
# Window Matcher  (RECOGNITION-ONLY: offsets removed)
# ---------------------------------------------------------------------------

class WindowMatcher:
    """
    Match generated windows to ground-truth activity bounding boxes and
    assign each a class label. Recognition-only: no offset targets.

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
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            windows   : (na, 2)  anchor windows [center, length]
            gt_boxes  : (nb, 2)  GT bounding boxes [center, length]
                       (still needed to decide window↔instance assignment,
                        even though no offset is derived from it)
            gt_labels : (nb,)    GT activity class labels (1-indexed;
                                 0 is reserved for background)

        Returns:
            matched_labels : (na,)  class label per window (0=background)
            pos_mask       : (na,)  bool mask of positive windows
        """
        na = windows.shape[0]
        nb = gt_boxes.shape[0]
        device = windows.device

        matched_labels = torch.zeros(na, dtype=torch.long, device=device)
        assigned_gt    = torch.full((na,), -1, dtype=torch.long, device=device)

        if nb == 0:
            return matched_labels, torch.zeros(na, dtype=torch.bool, device=device)

        M = iou_matrix(windows, gt_boxes)    # (na, nb)
        M_work = M.clone()

        # ---- Step 1: guarantee every GT box gets its best window ----
        for _ in range(nb):
            best = M_work.max()
            if best.item() == 0:
                break
            flat_idx = M_work.argmax()
            i = flat_idx // nb
            j = flat_idx %  nb
            assigned_gt[i] = j
            M_work[i, :] = -1
            M_work[:, j] = -1

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

        # ---- Step 3: encode labels only (no offset_encode call) ----
        pos_mask = assigned_gt >= 0
        pos_idx  = pos_mask.nonzero(as_tuple=True)[0]
        j_idx    = assigned_gt[pos_idx]
        matched_labels[pos_idx] = gt_labels[j_idx]

        return matched_labels, pos_mask

    def hard_negative_mining(self,
                             class_probs: torch.Tensor,
                             pos_mask:    torch.Tensor
                             ) -> torch.Tensor:
        """Select hard negatives so that neg : pos ≈ n_neg_ratio : 1."""
        n_pos = pos_mask.sum().item()
        n_neg_target = min(int(n_pos * self.n_neg_ratio),
                           (~pos_mask).sum().item())

        bg_prob  = class_probs[:, 0]
        neg_loss = -torch.log(bg_prob.clamp(min=1e-7))
        neg_loss = neg_loss.masked_fill(pos_mask, -1.0)

        _, sorted_idx = neg_loss.sort(descending=True)
        neg_mask = torch.zeros_like(pos_mask)
        neg_mask[sorted_idx[:n_neg_target]] = True
        return neg_mask


# ---------------------------------------------------------------------------
# Quick functional test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    feat_len = 8
    data_len = 64
    scales   = [0.5, 1.0]

    gen = WindowGenerator(scales=scales, feat_len=feat_len, data_len=data_len)
    print(f'Generated {gen.num_windows} windows for feat_len={feat_len}, '
          f'data_len={data_len}, scales={scales}')

    gt_boxes  = torch.tensor([[15.0, 20.0],
                               [45.0, 10.0]])
    gt_labels = torch.tensor([1, 2])

    matcher = WindowMatcher(pos_iou_thresh=0.4)
    lbl, pos = matcher.match(gen.windows, gt_boxes, gt_labels)

    print(f'Positive windows  : {pos.sum().item()}')
    print(f'Background windows: {(lbl == 0).sum().item()}')

    if torch.cuda.is_available():
        gen_cuda = gen.windows.cuda()
        gt_boxes_cuda = gt_boxes.cuda()
        gt_labels_cuda = gt_labels.cuda()
        lbl_c, pos_c = matcher.match(gen_cuda, gt_boxes_cuda, gt_labels_cuda)
        print('CUDA match() succeeded, device:', lbl_c.device)
