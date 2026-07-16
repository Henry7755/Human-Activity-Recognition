"""
model/multiscale_windows.py  (SEGMENTATION-ONLY VARIANT)
===========================================================
Section III-B of the paper: Multiscale Window Generation and Matching.

CHANGED vs. the multi-task version
-----------------------------------
Implements §4.B ("Segmentation-only conversion") step 2 from the
coupling-map analysis:

  - `WindowMatcher.match()` no longer returns a per-class `matched_labels`
    tensor at all. With no `cls_branch` (deleted in `model/segmentation.py`),
    there's nothing that predicts a class, so there's no per-class target
    to build. As the analysis doc puts it: "you still need pos_mask for
    offset supervision, but the label tensor itself becomes trivial" — in
    fact it's not just trivial, it's *redundant* with `pos_mask` itself
    (foreground = `pos_mask`, background = `~pos_mask`), so it is dropped
    from the return signature entirely rather than kept as a
    always-equal-to-pos_mask.long() dead value.
  - Return signature changes from
        (matched_labels, matched_offsets, pos_mask)
    to
        (matched_offsets, pos_mask)
    `pos_mask` doubles as the binary foreground/background target for the
    new objectness head (see `model/segmentation.py`).

Kept unchanged
---------------
  - `offset_encode` / `offset_decode` (Equations 1-4) — segmentation-only
    still regresses boundaries, so these are still needed, unlike in the
    recognition-only variant.
  - `WindowGenerator`, `iou_1d`, `iou_matrix` — unchanged.
  - `WindowMatcher.hard_negative_mining` — DELETED. Per the analysis doc's
    §4.B step 1, hard-negative mining existed only to feed the per-class
    `ClassificationLoss`, which no longer exists. The new binary objectness
    loss (see `training/losses.py`) handles foreground/background
    imbalance via `pos_weight` in `BCEWithLogitsLoss` instead of explicit
    negative mining.
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
# Offset encoding / decoding  (Equations 1-4) — UNCHANGED, still needed
# ---------------------------------------------------------------------------

def offset_encode(windows: torch.Tensor,
                  gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Encode ground-truth boxes relative to matched windows.

    Implements Equations (1) and (2):
        f_x = (t_x - w_x) / w_l
        f_l = log(t_l / w_l)
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
    """
    pred_x = offsets[:, 0] * windows[:, 1] + windows[:, 0]
    pred_l = windows[:, 1] * torch.exp(offsets[:, 1])
    return torch.stack([pred_x, pred_l], dim=1)


# ---------------------------------------------------------------------------
# Window Generator  (unchanged from multi-task version)
# ---------------------------------------------------------------------------

class WindowGenerator:
    """
    Generate multiscale anchor windows for a 1-D feature sequence.

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
        self.windows  = self._generate()

    def _generate(self) -> torch.Tensor:
        ratio = self.data_len / self.feat_len

        window_list = []
        for x in range(self.feat_len):
            center_data = (x + 0.5) * ratio

            for s in self.scales:
                sqrts = math.sqrt(s)
                for length_data in [self.feat_len * sqrts * ratio,
                                    self.feat_len / sqrts * ratio]:
                    window_list.append([center_data, length_data])

        return torch.tensor(window_list, dtype=torch.float32)

    @property
    def num_windows(self) -> int:
        return self.windows.shape[0]

    def to_start_end(self) -> torch.Tensor:
        centers = self.windows[:, 0]
        lengths = self.windows[:, 1]
        starts  = (centers - lengths / 2).clamp(min=0)
        ends    = (centers + lengths / 2).clamp(max=self.data_len - 1)
        return torch.stack([starts, ends], dim=1)


# ---------------------------------------------------------------------------
# Window Matcher  (SEGMENTATION-ONLY: labels collapsed away, offsets kept)
# ---------------------------------------------------------------------------

class WindowMatcher:
    """
    Match generated windows to ground-truth activity bounding boxes.

    Step 1: Find the best window for every GT box (highest IOU).
    Step 2: For remaining unmatched windows, assign the GT box with the
            highest IOU if that IOU exceeds `pos_iou_thresh`.
    Step 3: Windows with IOU < neg_iou_thresh for ALL GT boxes become
            background.

    Args:
        pos_iou_thresh : minimum IOU to label a window as positive
        neg_iou_thresh : maximum IOU for a window to be labelled background
                        (kept for interface parity / future re-use, though
                        segmentation-only doesn't do explicit "ignore"
                        handling any differently than recognition-only did)
    """

    def __init__(self,
                 pos_iou_thresh: float = 0.5,
                 neg_iou_thresh: float = 0.3):
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def match(self,
              windows:  torch.Tensor,
              gt_boxes: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            windows   : (na, 2)  anchor windows [center, length]
            gt_boxes  : (nb, 2)  GT bounding boxes [center, length]
                       (no gt_labels arg anymore — there is no class to
                        assign, only a foreground/background + boundary)

        Returns:
            matched_offsets : (na, 2)  encoded offsets (valid for positives)
            pos_mask        : (na,)    bool mask of positive (foreground)
                              windows — doubles as the objectness target
        """
        na = windows.shape[0]
        nb = gt_boxes.shape[0]
        device = windows.device

        matched_offsets = torch.zeros(na, 2, dtype=torch.float32, device=device)
        assigned_gt     = torch.full((na,), -1, dtype=torch.long, device=device)

        if nb == 0:
            return matched_offsets, torch.zeros(na, dtype=torch.bool, device=device)

        M = iou_matrix(windows, gt_boxes)
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

        # ---- Step 3: encode offsets only — no label tensor at all ----
        pos_mask = assigned_gt >= 0
        pos_idx  = pos_mask.nonzero(as_tuple=True)[0]
        j_idx    = assigned_gt[pos_idx]
        matched_offsets[pos_idx] = offset_encode(windows[pos_idx], gt_boxes[j_idx])

        return matched_offsets, pos_mask

    # NOTE: hard_negative_mining intentionally DELETED here. It only ever
    # fed the per-class ClassificationLoss, which no longer exists in the
    # segmentation-only variant. Foreground/background imbalance is now
    # handled via `pos_weight` in the new binary ObjectnessLoss (see
    # training/losses.py) instead of explicit top-k negative selection.


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

    gt_boxes = torch.tensor([[15.0, 20.0],
                              [45.0, 10.0]])

    matcher = WindowMatcher(pos_iou_thresh=0.4)
    off, pos = matcher.match(gen.windows, gt_boxes)

    print(f'Positive (foreground) windows: {pos.sum().item()}')
    print(f'Background windows           : {(~pos).sum().item()}')

    pos_idx = pos.nonzero(as_tuple=True)[0]
    if len(pos_idx):
        w = gen.windows[pos_idx]
        o = off[pos_idx]
        recovered = offset_decode(w, o)
        print('Offset encode→decode round-trip (first positive):',
              recovered[0].tolist())

    if torch.cuda.is_available():
        gen_cuda = gen.windows.cuda()
        gt_boxes_cuda = gt_boxes.cuda()
        off_c, pos_c = matcher.match(gen_cuda, gt_boxes_cuda)
        print('CUDA match() succeeded, device:', off_c.device)
