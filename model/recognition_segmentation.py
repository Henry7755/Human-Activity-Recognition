"""
model/recognition.py  (RECOGNITION-ONLY VARIANT of recognition_segmentation.py)
=================================================================================
Section III-C & III-D: Recognition Module + NMS, offset/localisation removed.

CHANGED vs. the multi-task version
-----------------------------------
Implements §4.A steps 3-5 from the coupling-map analysis:

  Step 3 (Head): `off_branch` is DELETED from `RecognitionNet`
  (renamed from `RecognitionSegmentationNet`). `forward()` now returns only
  `class_logits`.

  Step 4 (Model / predict): `MTHARSRecognition` (renamed from `MTHARS`)
  `.forward()` returns only `cls_logits`. `.predict()` no longer calls
  `offset_decode` — there is no offset to decode. `NonMaximumSuppression`
  now operates directly on the STATIC anchor windows
  (`self.window_gen.windows`), not on a decoded per-sample box, since no
  branch ever moves the boundary away from the anchor.

  Step 5 (Post-processing): `ConcatenateAlgorithm` is kept, because its
  stopping condition ("close a segment when the class changes") only reads
  `class`/`start`/`end`/`score` — it never depended on offsets, only on the
  *fact* that boundaries came from regression. With static anchors as the
  boundaries, it still does something meaningful: it coalesces adjacent
  anchor windows that were independently classified as the same activity
  into one reported span. If your use case is one-label-per-clip (matching
  what `trainer.evaluate` / `step3_evaluate.py` already report), you can
  skip `ConcatenateAlgorithm` entirely and just take the per-clip argmax —
  see the `whole_clip_argmax()` convenience method added to
  `MTHARSRecognition` below.

Deleted entirely
-----------------
  - `off_branch` (Conv1d, ~few-thousand params)
  - `offset_decode` import / call
  - the boundary-regression half of `predict()`
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.sknet import SKNet1D
from model.multiscale_windows import WindowGenerator, iou_1d


# ---------------------------------------------------------------------------
# Non-Maximum Suppression  (RECOGNITION-ONLY: ranks/suppresses on static anchors)
# ---------------------------------------------------------------------------

class NonMaximumSuppression:
    """
    Remove highly overlapping windows, retaining the highest-confidence one.

    CHANGED: previously took `pred_boxes` decoded per-sample from the offset
    branch. Recognition-only has no offset branch, so the boxes passed in
    here are always the same static anchor set
    (`MTHARSRecognition.window_gen.windows`) for every sample in the batch —
    only `class_probs` varies per sample. The ranking signal (max class
    prob, excluding background) is unchanged, since it was always a
    recognition quantity.

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
            windows     : (na, 2)    STATIC anchor windows [center, length]
                          (same for every sample — no per-sample decode step)
            class_probs : (na, K+1)  softmax probabilities (col 0 = background)

        Returns:
            kept_windows : (M, 2)   retained windows
            kept_scores  : (M,)     max class probability
            kept_classes : (M,)     predicted class index (0 = background)
        """
        fg_probs, fg_classes = class_probs[:, 1:].max(dim=1)
        fg_classes = fg_classes + 1      # shift back to 1-indexed

        keep = fg_probs >= self.score_thresh
        windows    = windows[keep]
        fg_probs   = fg_probs[keep]
        fg_classes = fg_classes[keep]

        if windows.shape[0] == 0:
            return windows, fg_probs, fg_classes

        order = fg_probs.argsort(descending=True)
        windows    = windows[order]
        fg_probs   = fg_probs[order]
        fg_classes = fg_classes[order]

        kept_idx = []
        suppressed = torch.zeros(len(windows), dtype=torch.bool, device=windows.device)

        for i in range(len(windows)):
            if suppressed[i]:
                continue
            kept_idx.append(i)
            if len(kept_idx) >= self.max_detections:
                break

            iou = iou_1d(windows[i + 1:], windows[i])
            suppress_mask = iou > self.iou_thresh
            suppressed[i + 1:][suppress_mask] = True

        kept_idx = torch.tensor(kept_idx, dtype=torch.long, device=windows.device)
        return windows[kept_idx], fg_probs[kept_idx], fg_classes[kept_idx]


# ---------------------------------------------------------------------------
# Recognition Module  (Section III-D, offset branch removed)
# ---------------------------------------------------------------------------

class RecognitionNet(nn.Module):
    """
    Single-branch Conv1D head operating on the backbone feature sequence.

    CHANGED: `off_branch` deleted. Output is class logits only.

    Input  : (B, feat_dim, n_feat)  – backbone feature sequence
    Output : class_logits : (B, n_feat * n_windows_per_unit, K+1)
    """

    def __init__(self,
                 feat_dim:            int,
                 n_classes:           int,
                 n_windows_per_unit:  int):
        super().__init__()
        self.n_classes          = n_classes
        self.n_windows_per_unit = n_windows_per_unit

        self.shared = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=False),
        )

        self.cls_branch = nn.Conv1d(
            feat_dim,
            n_windows_per_unit * (n_classes + 1),
            kernel_size=3, padding=1
        )
        # off_branch: DELETED (was nn.Conv1d(feat_dim, n_windows_per_unit*2, ...))

    def forward(self, feat_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_seq : (B, feat_dim, n_feat)

        Returns:
            class_logits : (B, n_feat * n_wpu, K+1)
        """
        B = feat_seq.shape[0]
        x = self.shared(feat_seq)

        cls = self.cls_branch(x)               # (B, n_wpu*(K+1), n_feat)
        cls = cls.permute(0, 2, 1)             # (B, n_feat, n_wpu*(K+1))
        cls = cls.contiguous().view(
            B, -1, self.n_classes + 1
        )
        return cls


# ---------------------------------------------------------------------------
# Full Recognition-Only Network
# ---------------------------------------------------------------------------

class MTHARSRecognition(nn.Module):
    """
    Recognition-only Human Activity Recognition network.

    Architecture:
        Input → SKNet1D backbone → Windows Generate (static anchors) →
        RecognitionNet → class_logits

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
            scales = [2.0, 3.0]

        self.scales   = scales
        self.data_len = data_len

        self.backbone = SKNet1D(in_channels=in_channels, feat_dim=feat_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, data_len)
            feat_len = self.backbone(dummy).shape[-1]

        self.feat_len = feat_len

        self.window_gen = WindowGenerator(
            scales=scales, feat_len=feat_len, data_len=data_len
        )
        n_wpu = len(scales) * 2

        self.head = RecognitionNet(
            feat_dim=feat_dim,
            n_classes=n_classes,
            n_windows_per_unit=n_wpu,
        )

        self.nms = NonMaximumSuppression(
            iou_thresh=nms_iou_thresh,
            score_thresh=nms_score_thr,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, T)  sensor window (T must equal self.data_len)

        Returns:
            class_logits : (B, na, K+1)
        """
        feat = self.backbone(x)
        cls_logits = self.head(feat)
        return cls_logits

    # ------------------------------------------------------------------
    # Inference — per-window detections (windowed / multi-instance mode)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[List[Dict]]:
        """
        Full inference pipeline: forward → NMS on static anchors.
        No decode step — the anchors ARE the boundaries.

        Args:
            x : (B, C, T) sensor input

        Returns:
            batch_results : list of B lists, each containing dicts:
                {'center': float, 'length': float,
                 'start': int,   'end': int,
                 'class': int,   'score': float}
        """
        self.eval()
        cls_logits = self.forward(x)                    # (B, na, K+1)
        class_probs = F.softmax(cls_logits, dim=-1)      # (B, na, K+1)

        anchors = self.window_gen.windows.to(x.device)   # (na, 2), static

        batch_results = []
        for b in range(x.shape[0]):
            probs = class_probs[b]                       # (na, K+1)

            kept_windows, kept_scores, kept_classes = self.nms(
                anchors, probs
            )

            results = []
            for i in range(len(kept_windows)):
                cx = kept_windows[i, 0].item()
                ln = kept_windows[i, 1].item()
                st = max(0,   int(cx - ln / 2))
                en = min(self.data_len - 1, int(cx + ln / 2))
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

    # ------------------------------------------------------------------
    # Inference — single label per clip (matches what trainer.evaluate /
    # step3_evaluate.py already compute; use this if you don't need
    # multiple detections per sample and want to drop ConcatenateAlgorithm)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def whole_clip_argmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, T)

        Returns:
            preds : (B,) predicted class index per clip (0-indexed,
                    background column excluded — mirrors
                    `trainer.evaluate`'s `cls_logits[:, :, 1:].mean(dim=1)`)
        """
        self.eval()
        cls_logits = self.forward(x)                       # (B, na, K+1)
        agg_logits = cls_logits[:, :, 1:].mean(dim=1)       # (B, K)
        return agg_logits.argmax(dim=1)


# ---------------------------------------------------------------------------
# Concatenation Algorithm  (kept — see module docstring for why)
# ---------------------------------------------------------------------------

class ConcatenateAlgorithm:
    """
    Merge adjacent predicted windows of the same activity class into
    contiguous activity segments. Unchanged from the multi-task version:
    it never read offsets directly, only start/end/class/score, which
    static anchors still provide.
    """

    @staticmethod
    def merge(segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []

        segs = sorted(segments, key=lambda s: s['start'])

        merged = []
        cur_class = segs[0]['class']
        cur_start = segs[0]['start']
        cur_end   = segs[0]['end']
        cur_score = segs[0]['score']

        for seg in segs[1:]:
            if seg['class'] == cur_class:
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

    def __call__(self, batch_segs: List[List[Dict]]) -> List[List[Dict]]:
        return [self.merge(s) for s in batch_segs]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, C, T = 2, 9, 300
    K = 6

    model = MTHARSRecognition(in_channels=C, n_classes=K,
                              scales=[2.0, 3.0], feat_dim=128, data_len=T)

    print(f'Backbone feat_len : {model.feat_len}')
    print(f'Total anchors     : {model.window_gen.num_windows}')

    x = torch.randn(B, C, T)

    model.train()
    cls_out = model(x)
    print(f'[Train] cls_logits : {tuple(cls_out.shape)}')   # (B, na, K+1)

    results = model.predict(x)
    concat  = ConcatenateAlgorithm()(results)
    for b, segs in enumerate(concat):
        print(f'[Infer/windowed] Sample {b}: {len(segs)} activity segment(s)')

    preds = model.whole_clip_argmax(x)
    print(f'[Infer/whole-clip] preds: {preds.tolist()}')
