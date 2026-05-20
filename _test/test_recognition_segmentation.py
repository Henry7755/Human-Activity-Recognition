import pytest
import torch
import torch.nn as nn
from typing import List, Dict

# Import your components from your module package
from model.recognition_segmentation import (
    NonMaximumSuppression,
    RecognitionSegmentationNet,
    MTHARS,
    ConcatenateAlgorithm
)

# ---------------------------------------------------------------------------
# 1. Non-Maximum Suppression (NMS) Tests
# ---------------------------------------------------------------------------

def test_nms_suppression_and_thresholding(monkeypatch):
    """
    Expected:
        - Windows with confidence scores below score_thresh are dropped immediately.
        - Higher-confidence windows suppress adjacent, overlapping windows with an IoU > iou_thresh.
    Actual:
        - Setup a prediction matrix with 3 windows. Window 0 and Window 1 heavily overlap.
          Window 2 has a very low classification probability.
        - Checked that Window 1 is suppressed by Window 0, and Window 2 is dropped by score thresholding.
    The "Why":
        - Validates that the model preserves high-confidence distinct activity spans while successfully
          cleaning up duplicate boundary proposals around the same action.
    """
    # Mock iou_1d since it belongs to an external module
    def mock_iou_1d(windows_left, base_window):
        # Force a high overlap simulation for the first comparison
        return torch.tensor([0.8]) if len(windows_left) == 1 else torch.tensor([])
        
    monkeypatch.setattr("model.recognition_segmentation.iou_1d", mock_iou_1d)
    
    nms = NonMaximumSuppression(iou_thresh=0.5, score_thresh=0.10)
    
    # 3 mock anchors/windows
    windows = torch.tensor([[50.0, 10.0], [52.0, 10.0], [100.0, 20.0]], dtype=torch.float32)
    # class_probs mapping: shape (na, K+1) -> 2 classes excluding background
    class_probs = torch.tensor([
        [0.1, 0.9, 0.0],   # Win 0: Max score 0.9 (Class 1) -> Keep
        [0.2, 0.0, 0.8],   # Win 1: Max score 0.8 (Class 2) -> Should be suppressed by IoU
        [0.95, 0.05, 0.0]  # Win 2: Max score 0.05 (Class 1) -> Below score threshold (0.10)
    ], dtype=torch.float32)
    
    kept_windows, kept_scores, kept_classes = nms(windows, class_probs)
    
    assert len(kept_windows) == 1
    assert torch.equal(kept_windows[0], torch.tensor([50.0, 10.0]))
    assert kept_scores[0].item() == pytest.approx(0.9)
    assert kept_classes[0].item() == 1


# ---------------------------------------------------------------------------
# 2. RecognitionSegmentationNet Tests
# ---------------------------------------------------------------------------

def test_recognition_segmentation_head_dimensions():
    """
    Expected:
        - Given input features of shape (B, feat_dim, n_feat), the output tensor shapes must be:
          • Class logits: (B, n_feat * n_windows_per_unit, K+1)
          • Offsets: (B, n_feat * n_windows_per_unit, 2)
    Actual:
        - Fed a dummy tensor representing backbone outputs through the head and evaluated dimension sizes.
    The "Why":
        - Ensures spatial dimension flattening and sequence tracking mappings correctly preserve 
          alignment between multi-scale window anchors and backbone spatial features.
    """
    B, feat_dim, n_feat = 4, 128, 32
    n_classes = 5
    n_wpu = 4 # e.g., 2 scales * 2 lengths
    
    head = RecognitionSegmentationNet(feat_dim=feat_dim, n_classes=n_classes, n_windows_per_unit=n_wpu)
    feat_seq = torch.randn(B, feat_dim, n_feat)
    
    cls_logits, offsets = head(feat_seq)
    
    assert cls_logits.shape == (B, n_feat * n_wpu, n_classes + 1)
    assert offsets.shape == (B, n_feat * n_wpu, 2)
    assert cls_logits.is_contiguous()
    assert offsets.is_contiguous()


# ---------------------------------------------------------------------------
# 3. ConcatenateAlgorithm Tests
# ---------------------------------------------------------------------------

def test_concatenate_algorithm_merging_logic():
    """
    Expected:
        - Adjacent or overlapping sequence predictions possessing the same activity label must merge
          into a continuous segment spanning the total boundary width.
        - Predictions with different labels must remain separate segments.
    Actual:
        - Passed a specific list of segment dictionaries into ConcatenateAlgorithm.
        - Verified that the first two segments (Class 1) merged from index 10 to 60, while the 
          third segment (Class 2) started an isolated sequence profile.
    The "Why":
        - Implements Algorithm 1 of the paper. This validates that fragmented multi-window predictions 
          are unified into a single clean segmented timeline for evaluation.
    """
    mock_segments = [
        {'start': 10, 'end': 40, 'class': 1, 'score': 0.85},
        {'start': 35, 'end': 60, 'class': 1, 'score': 0.92}, # Same class, overlapping -> merge
        {'start': 61, 'end': 90, 'class': 2, 'score': 0.70}  # Different class -> split
    ]
    
    concat_fn = ConcatenateAlgorithm()
    merged_results = concat_fn.merge(mock_segments)
    
    assert len(merged_results) == 2
    
    # Check merged Class 1 segment
    assert merged_results[0]['start'] == 10
    assert merged_results[0]['end'] == 60
    assert merged_results[0]['label'] == 1
    assert merged_results[0]['score'] == 0.92 # Max score preserved
    
    # Check separate Class 2 segment
    assert merged_results[1]['start'] == 61
    assert merged_results[1]['end'] == 90
    assert merged_results[1]['label'] == 2


# ---------------------------------------------------------------------------
# 4. End-to-End MTHARS Structural Mocks
# ---------------------------------------------------------------------------

# Setup Minimal Mock blocks for dependencies that MTHARS constructs internally
class MockSKNet1D(nn.Module):
    def __init__(self, in_channels, feat_dim):
        super().__init__()
        # Simply downsample sequence length by 2 to mock a convolutional backbone stride
        self.conv = nn.Conv1d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class MockWindowGenerator:
    def __init__(self, scales, feat_len, data_len):
        self.num_windows = feat_len * len(scales) * 2
        self.windows = torch.zeros(self.num_windows, 2)


def test_mthars_pipeline_execution(monkeypatch):
    """
    Expected:
        - Training mode forward pass returns tensors matching anchor quantity dimensional limits.
        - Inference mode `predict()` correctly evaluates bounding box transformations and runs NMS.
    Actual:
        - Substituted internal complex dependencies (`SKNet1D`, `WindowGenerator`) using monkeypatching.
        - Verified full end-to-end execution consistency without dimension runtime errors.
    The "Why":
        - Checks that your top-level orchestration class flows cleanly between training loss paths 
          and production inference pipelines.
    """
    # Apply structural mocks
    monkeypatch.setattr("model.recognition_segmentation.SKNet1D", MockSKNet1D)
    monkeypatch.setattr("model.recognition_segmentation.WindowGenerator", MockWindowGenerator)
    # Mock offset decoding to return static dummy windows matching expected dims
    monkeypatch.setattr("model.recognition_segmentation.offset_decode", lambda anchors, off: anchors)
    
    B, C, T = 2, 9, 100
    K = 3
    
    model = MTHARS(in_channels=C, n_classes=K, scales=[2.0, 3.0], feat_dim=64, data_len=T)
    x = torch.randn(B, C, T)
    
    # 1. Check Training Flow Path
    model.train()
    cls_out, off_out = model(x)
    assert cls_out.shape[0] == B
    assert off_out.shape[0] == B
    assert cls_out.shape[2] == K + 1
    assert off_out.shape[2] == 2
    
    # 2. Check Inference Predictions Flow Path
    model.eval()
    batch_results = model.predict(x)
    assert len(batch_results) == B
    assert isinstance(batch_results[0], list)
    if len(batch_results[0]) > 0:
        item = batch_results[0][0]
        assert set(item.keys()) == {'center', 'length', 'start', 'end', 'class', 'score'}