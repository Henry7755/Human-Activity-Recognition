"""
model/recognition_segmentation.py
===================================
Section III-C & III-D: Recognition and Segmentation Module + NMS.

Components
----------
1. NonMaximumSuppression
   – filters overlapping predicted windows (Section III-B, NMS sub-section).

2. RecognitionSegmentationNet
   – dual-branch Conv1D head that predicts:
       • class probabilities for each window    (k+1 values per window)
       • boundary offsets for each window       (2 values per window: f_x, f_l)

3. MTHARS
   – full multi-task network:
       backbone (SKNet1D) → Windows Generate → RecognitionSegmentationNet
   – returns class logits + offsets during training
   – returns decoded activity segments during inference

4. ConcatenateAlgorithm
   – Algorithm 1 from the paper: merges adjacent windows of the same
     class into contiguous activity segments.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.sknet import SKNet1D
from model.multiscale_windows import (
    WindowGenerator, offset_decode, iou_1d
)


# ---------------------------------------------------------------------------
# Non-Maximum Suppression  (Section III-B)
# ---------------------------------------------------------------------------

class NonMaximumSuppression:
    """
    Remove highly overlapping windows, retaining the highest-confidence one.

    Algorithm (from the paper):
        1. Sort windows by their maximum class probability c (descending).
        2. Select the top window W_1 as a base; discard windows whose
           IOU with W_1 exceeds `iou_thresh`.
        3. Repeat with the next surviving window until the list is empty.

    Args:
        iou_thresh    : IOU above which a window is suppressed (default 0.5).
        score_thresh  : discard windows whose max-class-prob < score_thresh.
        max_detections: hard cap on the number of kept windows.
    """

    def __init__(self,
                 iou_thresh:    float = 0.50,
                 score_thresh:  float = 0.01,
                 max_detections: int  = 200):
        self.iou_thresh     = iou_thresh
        self.score_thresh   = score_thresh
        self.max_detections = max_detections

    def __call__(self,
                 windows:     torch.Tensor,
                 class_probs: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            windows     : (na, 2)    decoded windows [center, length]
            class_probs : (na, K+1) softmax probabilities (col 0 = background)

        Returns:
            kept_windows : (M, 2)   retained windows
            kept_scores  : (M,)     max class probability
            kept_classes : (M,)     predicted class index (0 = background)
        """
        # Max prob and predicted class (ignoring background col 0)
        fg_probs, fg_classes = class_probs[:, 1:].max(dim=1)
        fg_classes = fg_classes + 1      # shift back to 1-indexed

        # Filter by score threshold
        keep = fg_probs >= self.score_thresh
        windows     = windows[keep]
        fg_probs    = fg_probs[keep]
        fg_classes  = fg_classes[keep]

        if windows.shape[0] == 0:
            return windows, fg_probs, fg_classes

        # Sort descending by probability
        order = fg_probs.argsort(descending=True)
        windows    = windows[order]
        fg_probs   = fg_probs[order]
        fg_classes = fg_classes[order]

        kept_idx = []
        # FIX: previously `torch.zeros(len(windows), dtype=torch.bool)` and the
        # `torch.tensor(kept_idx, dtype=torch.long)` below both defaulted to
        # CPU. When `windows` (and therefore `class_probs`) live on a CUDA
        # device — which they will the moment predict() is called on a GPU
        # model — indexing/assigning across these mismatched devices raises a
        # RuntimeError. Allocate both on windows.device instead.
        suppressed = torch.zeros(len(windows), dtype=torch.bool, device=windows.device)

        for i in range(len(windows)):
            if suppressed[i]:
                continue
            kept_idx.append(i)
            if len(kept_idx) >= self.max_detections:
                break

            # Suppress windows with high IOU with the current base
            iou = iou_1d(windows[i+1:], windows[i])
            suppress_mask = iou > self.iou_thresh
            suppressed[i+1:][suppress_mask] = True

        kept_idx = torch.tensor(kept_idx, dtype=torch.long, device=windows.device)
        return windows[kept_idx], fg_probs[kept_idx], fg_classes[kept_idx]


# ---------------------------------------------------------------------------
# Recognition and Segmentation Module  (Section III-D)
# ---------------------------------------------------------------------------

class RecognitionSegmentationNet(nn.Module):
    """
    Dual-branch Conv1D head operating on the backbone feature sequence.

    Input  : (B, feat_dim, n_feat)  – backbone feature sequence
    Output :
        class_logits : (B, n_feat * n_windows_per_unit, K+1)
        offsets      : (B, n_feat * n_windows_per_unit, 2)

    where  n_windows_per_unit = n_scales × 2  (two lengths per scale).

    Design note (Section III-D):
        We use Conv1D rather than FC layers to keep the number of
        parameters manageable and to preserve the correspondence between
        spatial positions in the feature map and window centers.
    """

    def __init__(self,
                 feat_dim:            int,
                 n_classes:           int,
                 n_windows_per_unit:  int):
        """
        Args:
            feat_dim           : channels in the backbone output.
            n_classes          : number of activity classes K
                                 (output has K+1 channels, col-0 = background).
            n_windows_per_unit : number of anchor windows per feature unit
                                 = len(scales) × 2.
        """
        super().__init__()
        self.n_classes          = n_classes
        self.n_windows_per_unit = n_windows_per_unit

        # Shared feature refinement
        self.shared = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=False),
        )

        # Class prediction branch: outputs  n_windows_per_unit × (K+1)  per position
        self.cls_branch = nn.Conv1d(
            feat_dim,
            n_windows_per_unit * (n_classes + 1),
            kernel_size=3, padding=1
        )

        # Offset prediction branch: outputs  n_windows_per_unit × 2  per position
        self.off_branch = nn.Conv1d(
            feat_dim,
            n_windows_per_unit * 2,
            kernel_size=3, padding=1
        )

    def forward(self, feat_seq: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_seq : (B, feat_dim, n_feat)

        Returns:
            class_logits : (B, n_feat * n_wpu, K+1)
            offsets      : (B, n_feat * n_wpu, 2)
        """
        B = feat_seq.shape[0]
        x = self.shared(feat_seq)              # (B, feat_dim, n_feat)

        # Class branch
        cls = self.cls_branch(x)               # (B, n_wpu*(K+1), n_feat)
        cls = cls.permute(0, 2, 1)             # (B, n_feat, n_wpu*(K+1))
        cls = cls.contiguous().view(
            B, -1, self.n_classes + 1          # (B, n_feat*n_wpu, K+1)
        )

        # Offset branch
        off = self.off_branch(x)               # (B, n_wpu*2, n_feat)
        off = off.permute(0, 2, 1)             # (B, n_feat, n_wpu*2)
        off = off.contiguous().view(B, -1, 2)  # (B, n_feat*n_wpu, 2)

        return cls, off


# ---------------------------------------------------------------------------
# Full MTHARS Network
# ---------------------------------------------------------------------------

class MTHARS(nn.Module):
    """
    Multi-Task Human Activity Recognition and Segmentation network.

    Architecture (Fig. 3 of the paper):
        Input → SKNet1D backbone → Windows Generate →
        RecognitionSegmentationNet → {class_logits, offsets}

    During *training* call forward() which returns logits and offsets.
    During *inference* call predict() which applies NMS and the
    concatenation algorithm to return activity segments.

    Args:
        in_channels   : number of sensor input channels (C).
        n_classes     : number of activity classes K.
        scales        : list of scale values for window generation.
        feat_dim      : backbone output channel width.
        data_len      : raw activity stream length fed to the network
                        per forward pass (must be fixed, e.g. 300).
        nms_iou_thresh: IOU threshold for NMS.
        nms_score_thr : minimum class score threshold for NMS.
    """

    def __init__(self,
                 in_channels:    int,
                 n_classes:      int,
                 scales:         List[float] = None,
                 feat_dim:       int = 256,
                 data_len:       int = 300,
                 nms_iou_thresh: float = 0.5,
                 nms_score_thr:  float = 0.01):
        super().__init__()

        if scales is None:
            scales = [2.0, 3.0]          # paper's best setting (Table VIII)

        self.scales   = scales
        self.data_len = data_len

        # ---- Backbone ----
        self.backbone = SKNet1D(in_channels=in_channels, feat_dim=feat_dim)

        # Infer feat_len by a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, data_len)
            feat_len = self.backbone(dummy).shape[-1]

        self.feat_len = feat_len

        # ---- Window Generator (static, not learned) ----
        self.window_gen = WindowGenerator(
            scales=scales, feat_len=feat_len, data_len=data_len
        )
        n_wpu = len(scales) * 2         # windows per feature unit

        # ---- Recognition & Segmentation Head ----
        self.head = RecognitionSegmentationNet(
            feat_dim=feat_dim,
            n_classes=n_classes,
            n_windows_per_unit=n_wpu,
        )

        # ---- NMS for inference ----
        self.nms = NonMaximumSuppression(
            iou_thresh=nms_iou_thresh,
            score_thresh=nms_score_thr,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, C, T)  sensor window (T must equal self.data_len)

        Returns:
            class_logits : (B, na, K+1)
            offsets      : (B, na, 2)
        """
        feat = self.backbone(x)                 # (B, feat_dim, n_feat)
        cls_logits, offsets = self.head(feat)   # (B, na, K+1), (B, na, 2)
        return cls_logits, offsets

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor
                ) -> List[List[Dict]]:
        """
        Full inference pipeline: forward → decode → NMS.

        Args:
            x : (B, C, T) sensor input

        Returns:
            batch_results : list of B lists, each containing dicts:
                {'center': float, 'length': float,
                 'start': int,   'end': int,
                 'class': int,   'score': float}
        """
        self.eval()
        cls_logits, raw_offsets = self.forward(x)  # (B, na, K+1), (B, na, 2)
        class_probs = F.softmax(cls_logits, dim=-1) # (B, na, K+1)

        anchors = self.window_gen.windows.to(x.device)  # (na, 2)
        batch_results = []

        for b in range(x.shape[0]):
            probs   = class_probs[b]                     # (na, K+1)
            offsets = raw_offsets[b]                     # (na, 2)

            # Decode predicted boxes
            pred_boxes = offset_decode(anchors, offsets)  # (na, 2)

            # NMS
            kept_windows, kept_scores, kept_classes = self.nms(
                pred_boxes, probs
            )

            results = []
            for i in range(len(kept_windows)):
                cx  = kept_windows[i, 0].item()
                ln  = kept_windows[i, 1].item()
                st  = max(0,   int(cx - ln / 2))
                en  = min(self.data_len - 1, int(cx + ln / 2))
                results.append({
                    'center': cx,
                    'length': ln,
                    'start':  st,
                    'end':    en,
                    'class':  kept_classes[i].item(),
                    'score':  kept_scores[i].item(),
                })
            batch_results.append(results)

        return batch_results


# ---------------------------------------------------------------------------
# Concatenation Algorithm  (Algorithm 1 in the paper)
# ---------------------------------------------------------------------------

class ConcatenateAlgorithm:
    """
    Merge adjacent predicted windows of the same activity class into
    contiguous activity segments.

    This implements Algorithm 1 exactly:
        - Sort predictions by start position.
        - Walk through sequentially; whenever the class changes, close
          the current segment and open a new one.

    Args:
        segments_per_batch : list of per-sample segment lists produced by
                             MTHARS.predict().

    Returns:
        merged : list (per sample) of merged segments
                 {'start': int, 'end': int, 'label': int, 'score': float}
    """

    @staticmethod
    def merge(segments: List[Dict]) -> List[Dict]:
        """Merge a single sample's segment list."""
        if not segments:
            return []

        # Sort by start position
        segs = sorted(segments, key=lambda s: s['start'])

        merged = []
        cur_class = segs[0]['class']
        cur_start = segs[0]['start']
        cur_end   = segs[0]['end']
        cur_score = segs[0]['score']

        for seg in segs[1:]:
            if seg['class'] == cur_class:
                # Extend current segment
                cur_end   = max(cur_end, seg['end'])
                cur_score = max(cur_score, seg['score'])
            else:
                merged.append({'start': cur_start,
                               'end':   cur_end,
                               'label': cur_class,
                               'score': cur_score})
                cur_class = seg['class']
                cur_start = seg['start']
                cur_end   = seg['end']
                cur_score = seg['score']

        merged.append({'start': cur_start,
                       'end':   cur_end,
                       'label': cur_class,
                       'score': cur_score})
        return merged

    def __call__(self,
                 batch_segs: List[List[Dict]]
                 ) -> List[List[Dict]]:
        return [self.merge(s) for s in batch_segs]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, C, T = 2, 9, 300
    K = 6

    model = MTHARS(in_channels=C, n_classes=K,
                   scales=[2.0, 3.0], feat_dim=128, data_len=T)

    print(f'Backbone feat_len : {model.feat_len}')
    print(f'Total anchors     : {model.window_gen.num_windows}')

    x = torch.randn(B, C, T)

    # Training mode
    model.train()
    cls_out, off_out = model(x)
    print(f'[Train] cls_logits : {tuple(cls_out.shape)}')   # (B, na, K+1)
    print(f'[Train] offsets    : {tuple(off_out.shape)}')   # (B, na, 2)

    # Inference mode
    results = model.predict(x)
    concat  = ConcatenateAlgorithm()(results)
    for b, segs in enumerate(concat):
        print(f'[Infer] Sample {b}: {len(segs)} activity segment(s)')
        for s in segs[:3]:
            print(f'  {s}')
