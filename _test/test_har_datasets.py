import pytest
import numpy as np
import torch
from datasets.har_datasets import sliding_window, build_segments, normalise, HARDataset

# ---------------------------------------------------------------------------
# 1. Sliding Window Preprocessing Block
# ---------------------------------------------------------------------------

def test_sliding_window_matrix_generation():
    """
    Expected:
        - The window slice must reshape a continuous stream of shape (T, C) into (N, C, window).
        - The window assignment label must pick the statistical mode (majority vote) inside the window boundaries.
    Actual:
        - Fed a mock continuous array of length 10 with 2 features, where a label swap occurs at index 5.
        - Set a window size of 4 and a stride of 2.
    The "Why":
        - Deep learning networks expect uniformly sized matrix blocks. This test verifies your sliding data tokenization
          slices the timeline accurately and resolves overlapping frame label shifts without off-by-one index bugs.
    """
    # 10 timestamp frames, 2 sensors
    mock_data = np.arange(20, dtype=np.float32).reshape(10, 2)
    # 0 0 0 0 0 | 1 1 1 1 1
    mock_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    
    window = 4
    stride = 2
    
    X, y = sliding_window(mock_data, mock_labels, window=window, stride=stride)
    
    # Expected number of windows: (10 - 4) // 2 + 1 = 4 windows
    assert X.shape == (4, 2, 4)
    assert y.shape == (4,)
    
    # Verify majority voting logic on the 3rd window (starts at frame index 4: ends at 8)
    # Labels inside window 3: [0, 1, 1, 1] -> Majority vote must equal 1
    assert y[2] == 1


# ---------------------------------------------------------------------------
# 2. Contiguous Bounding Frame Builder
# ---------------------------------------------------------------------------

def test_build_segments_run_length_encoding():
    """
    Expected:
        - Group long linear sequences of matching IDs into distinct bounding box profiles.
        - Output dictionaries containing the precise 'start', 'end', and 'label' boundary tags.
    Actual:
        - Passed an array containing a distinct run of class 0 followed immediately by class 3.
        - Verified the index limits match exactly (inclusive boundaries).
    The "Why":
        - Section III-E requires extracting ground-truth segmentation boundaries. This test ensures the run-length
          encoding logic doesn't drop edge frames or overlap index coordinates.
    """
    mock_frame_labels = np.array([0, 0, 0, 3, 3, 3, 3], dtype=np.int64)
    
    segments = build_segments(mock_frame_labels)
    
    assert len(segments) == 2
    
    # Verify Class 0 bounds
    assert segments[0] == {'start': 0, 'end': 2, 'label': 0}
    # Verify Class 3 bounds
    assert segments[1] == {'start': 3, 'end': 6, 'label': 3}


# ---------------------------------------------------------------------------
# 3. Z-Score Normalization Scale Variance
# ---------------------------------------------------------------------------

def test_normalise_z_score_properties():
    """
    Expected:
        - Apply channel-wise normalization across the batch and timeline axes.
        - The resulting tensor channel arrays must display a mean of 0.0 and a standard deviation of 1.0.
    Actual:
        - Hand-crafted a high-variance array block across two unique window distributions.
        - Evaluated the mean and standard deviation profiles post-normalization.
    The "Why":
        - Sensor signals vary heavily across subjects and accelerometers. Failing to normalize correctly will corrupt
          your SKNet feature selection mappings. This test ensures normalization stays restricted per-channel.
    """
    # Shape: (N=2, C=2, T=3)
    mock_windows = np.array([
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        [[2.0, 3.0, 4.0], [20.0, 30.0, 40.0]]
    ], dtype=np.float32)
    
    norm_X = normalise(mock_windows)
    
    # Evaluate properties along the combined (Batch, Time) dimensions per channel
    # Channel 0 checking
    assert np.isclose(norm_X[:, 0, :].mean(), 0.0, atol=1e-5)
    assert np.isclose(norm_X[:, 0, :].std(), 1.0, atol=1e-5)
    
    # Channel 1 checking
    assert np.isclose(norm_X[:, 1, :].mean(), 0.0, atol=1e-5)
    assert np.isclose(norm_X[:, 1, :].std(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset Wrapper & Augmentation Injection
# ---------------------------------------------------------------------------

def test_har_dataset_tensor_conversion_and_augmentation():
    """
    Expected:
        - Convert standard NumPy structures cleanly into active torch.Tensor classes.
        - If `augment=True`, add deterministic random noise to the raw signal during extraction loops.
    Actual:
        - Evaluated `HARDataset` behavior with augmentation toggled off vs toggled on.
    The "Why":
        - Validates data plumbing safety. Ensuring augmentation modifies data dynamically on demand checks
          that your training code does not overwrite original evaluation validation metrics.
    """
    X_np = np.ones((5, 3, 50), dtype=np.float32)
    y_np = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    
    # Test Baseline Conversion
    dataset_clean = HARDataset(X_np, y_np, augment=False)
    x_tensor, y_tensor = dataset_clean[0]
    
    assert isinstance(x_tensor, torch.Tensor)
    assert isinstance(y_tensor, torch.Tensor)
    assert torch.equal(x_tensor, torch.ones(3, 50))
    
    # Test Gaussian Noise Augmentation Path
    dataset_augmented = HARDataset(X_np, y_np, augment=True)
    x_aug, _ = dataset_augmented[0]
    
    # Augmented tensor should now vary slightly from a vector of flat ones
    assert not torch.equal(x_aug, torch.ones(3, 50))
    assert x_aug.shape == (3, 50)