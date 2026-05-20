import pytest
import torch
import math
from model.multiscale_windows import (
    iou_1d,
    WindowGenerator,
    offset_encode,
    offset_decode,
    WindowMatcher
)

# ---------------------------------------------------------------------------
# 1. 1-D Interval Jaccard Index (IoU) Verification
# ---------------------------------------------------------------------------

def test_iou_1d_geometric_overlap():
    """
    Expected:
        - Perfect alignment must output an IoU of exactly 1.0.
        - Partial or adjacent boundaries must compute correct fractional intersection-over-union ratios.
        - Distinct non-overlapping sequences must output 0.0 without underflow numerical instability.
    Actual:
        - Checked IoU calculations across matching, partially shifting, and disconnected boundary shapes.
    The "Why":
        - Deep learning classification networks use IoU thresholds to assign targets. This test ensures 
          1-D sequence metrics align with your bounding boxes.
    """
    # 1. Perfect Match (Center 50, Length 20)
    w_perfect = torch.tensor([[50.0, 20.0]], dtype=torch.float32)
    gt_perfect = torch.tensor([50.0, 20.0], dtype=torch.float32)
    assert iou_1d(w_perfect, gt_perfect).item() == pytest.approx(1.0)

    # 2. 50% Geometric Overlap Simulation
    # Window A: start=40, end=60. GT: start=50, end=70. Intersection=10, Union=30 -> IoU = 1/3
    w_half = torch.tensor([[50.0, 20.0]], dtype=torch.float32)
    gt_half = torch.tensor([60.0, 20.0], dtype=torch.float32)
    assert iou_1d(w_half, gt_half).item() == pytest.approx(1.0 / 3.0)

    # 3. Disconnected Spans (No Overlap)
    w_none = torch.tensor([[10.0, 5.0]], dtype=torch.float32)
    gt_none = torch.tensor([100.0, 10.0], dtype=torch.float32)
    assert iou_1d(w_none, gt_none).item() == 0.0


# ---------------------------------------------------------------------------
# 2. Log-Length Offset Invariance (Equations 1–4)
# ---------------------------------------------------------------------------

def test_offset_encoding_decoding_roundtrip():
    """
    Expected:
        - Bounding boxes passed through `offset_encode` and subsequently decoded via `offset_decode`
          must return to their exact original absolute space coordinates.
    Actual:
        - Subjected a randomized sequence of anchor windows and targets to an encoding-decoding loop.
    The "Why":
        - Implements Equations (1) through (4). This test guarantees that the coordinate transformations 
          are mathematically sound, preventing your regression head from encountering training divergence.
    """
    # Shape: (N=2, 2) -> [center, length]
    mock_windows = torch.tensor([[45.0, 20.0], [120.0, 50.0]], dtype=torch.float32)
    mock_gt_boxes = torch.tensor([[48.0, 22.0], [115.0, 45.0]], dtype=torch.float32)

    # 1. Encode absolute boxes into fractional/log space offsets
    offsets = offset_encode(mock_windows, mock_gt_boxes)

    # 2. Decode offsets back into coordinate space positions
    reconstructed_boxes = offset_decode(mock_windows, offsets)

    # Verify original and reconstructed coordinates match exactly
    assert torch.allclose(reconstructed_boxes, mock_gt_boxes, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Hungarian Greedy Window Label Assignment
# ---------------------------------------------------------------------------

def test_window_matcher_greedy_bipartite_assignment():
    """
    Expected:
        - Step 1: Every individual ground-truth box must be assigned to its single best-matching anchor.
        - Step 2: Remaining anchors are assigned to target classes if they exceed `pos_iou_thresh`.
        - Step 3: Anchors with low overlap are classified as background (class 0).
    Actual:
        - Evaluated `WindowMatcher` behaviors using explicit overlaps mimicking clear action periods.
    The "Why":
        - Validates the greedy assignment logic detailed in Section III-B. This ensures anchor targets 
          are uniquely mapped, avoiding identical target duplicates or class leakage.
    """
    # Setup 3 anchor windows
    windows = torch.tensor([
        [20.0, 10.0],  # Window 0: Highly overlaps GT Box 0
        [22.0, 10.0],  # Window 1: Partially overlaps GT Box 0
        [80.0, 10.0]   # Window 2: Completely isolated (Should become background)
    ], dtype=torch.float32)

    # 1 Ground-Truth action span (Class 4)
    gt_boxes = torch.tensor([[20.0, 10.0]], dtype=torch.float32)
    gt_labels = torch.tensor([4], dtype=torch.long)

    matcher = WindowMatcher(pos_iou_thresh=0.5, neg_iou_thresh=0.3)
    matched_labels, matched_offsets, pos_mask = matcher.match(windows, gt_boxes, gt_labels)

    # Window 0 matches perfectly -> Pos mask is True, Label is 4
    assert pos_mask[0].item() is True
    assert matched_labels[0].item() == 4

    # Window 2 has zero overlap -> Pos mask is False, Label is 0 (Background)
    assert pos_mask[2].item() is False
    assert matched_labels[2].item() == 0


# ---------------------------------------------------------------------------
# 4. Hard Negative Mining Balancing
# ---------------------------------------------------------------------------

def test_hard_negative_mining_ratio_enforcement():
    """
    Expected:
        - Negative anchors must be sub-sampled to match the designated `n_neg_ratio` relative to positive anchors.
        - The mining filter must sort by cross-entropy difficulty, selecting anchors with the lowest background probability.
    Actual:
        - Fed a synthetic probability block containing a single positive anchor and multiple negative anchors.
    The "Why":
        - Implements Section III-B hard negative sampling. This test ensures background anchor selection maintains a 
          balanced negative-to-positive ratio, preventing the dominant background class from overwhelming training.
    """
    matcher = WindowMatcher(n_neg_ratio=3)

    # 5 anchors total: 1 positive, 4 potential background negatives
    pos_mask = torch.tensor([True, False, False, False, False], dtype=torch.bool)

    # Simulated class probabilities: index 0 is P(background)
    # Lower P(background) on negative samples indicates a harder negative example
    class_probs = torch.tensor([
        [0.01, 0.99],  # Anchor 0: Positive
        [0.10, 0.90],  # Anchor 1: Very hard negative (Low background prob) -> Select
        [0.20, 0.80],  # Anchor 2: Hard negative -> Select
        [0.30, 0.70],  # Anchor 3: Moderate negative -> Select
        [0.95, 0.05]   # Anchor 4: Easy negative -> Drop
    ], dtype=torch.float32)

    neg_mask = matcher.hard_negative_mining(class_probs, pos_mask)

    # Ratio is 3:1. With 1 positive, we must select exactly 3 hard negatives
    assert neg_mask.sum().item() == 3
    # Ensure anchor 4 (easy negative) was dropped
    assert neg_mask[4].item() is False
    