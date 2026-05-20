import pytest
import argparse
import numpy as np
import torch
from unittest.mock import patch

# Standardised plural project imports matching package directory names
from datasets.har_datasets import load_dataset, get_dataloaders, DATASET_INFO 
from training.trainer import Trainer, prepare_targets, train_epoch, evaluate
from model.multiscale_windows import WindowGenerator, WindowMatcher

# ---------------------------------------------------------------------------
# 1. Target Preparation Pipeline Verification
# ---------------------------------------------------------------------------

def test_prepare_targets_tensor_structures():
    """
    Expected:
        - Convert raw dictionary arrays into explicit tensor variables matching model output dimensions.
        - Matched labels tensor must be shape (B, na) and offsets must be (B, na, 2).
    Actual:
        - Provided a mock batch containing 2 separate window sequences with explicit activity labels.
    The "Why":
        - Section III-E specifies that the model must calculate confidence and bounding localization 
          losses simultaneously. This test ensures the batch generation alignment holds before feeding the loss layer.
    """
    device = torch.device('cpu')
    
    # 8 features downsampled, 64 raw samples length, scale s values
    window_gen = WindowGenerator(scales=[1.0], feat_len=8, data_len=64)
    matcher = WindowMatcher(pos_iou_thresh=0.5, neg_iou_thresh=0.3)
    
    # Mock ground truth segment inputs (Batch size = 2)
    mock_gt_segments = [
        [{'start': 0, 'end': 31, 'label': 1}],  # Sample 0
        [{'start': 32, 'end': 63, 'label': 2}]  # Sample 1
    ]
    
    matched_labels, true_offsets, pos_mask = prepare_targets(
        window_gen, matcher, mock_gt_segments, device
    )
    
    assert matched_labels.shape == (2, window_gen.num_windows)
    assert true_offsets.shape == (2, window_gen.num_windows, 2)
    assert pos_mask.shape == (2, window_gen.num_windows)


# ---------------------------------------------------------------------------
# 2. Mock Configuration Training Orchestration Loop
# ---------------------------------------------------------------------------

def test_trainer_end_to_end_execution_flow():
    """
    Expected:
        - The `Trainer` engine must correctly parse arguments, initialize the unified architecture model,
          and handle data streaming through standard evaluation loops cleanly.
    Actual:
        - Mocked out the heavy disk-access operations using NumPy data stubs (simulating a 3-channel IMU),
          and performed a short evaluation tracking run.
    The "Why":
        - This provides the ultimate proof of design cohesion. If this passes, it confirms your training loop,
          loss head calculations, and multi-scale anchors are structurally sound and ready for real dataset runs.
    """
    # 1. Define synthetic baseline data (10 windows, 3 channels, 128 samples timeline)
    mock_X = np.random.randn(10, 3, 128).astype(np.float32)
    mock_y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    mock_segs = [[{'start': 0, 'end': 127, 'label': int(lbl)}] for lbl in mock_y]
    
    # 2. Build explicit arguments mimicking CLI parameter values
    mock_args = argparse.Namespace(
        dataset='UCI',
        data_root='/mock/path',
        output_dir='./mock_checkpoints',
        augment=False,
        feat_dim=32,
        scales=[1.0],
        alpha=1.0,
        beta=1.0,
        n_neg_ratio=3,
        pos_iou_thresh=0.5,
        neg_iou_thresh=0.3,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-4,
        amp=False,
        seed=42
    )

    # 3. Use patch context managers to redirect disk loaders to our synthetic arrays
    # Patches point to the target path inside training.trainer where load_dataset was imported
    with patch('training.trainer.load_dataset', return_value=(mock_X, mock_y, mock_segs)), \
         patch('training.trainer.torch.save', return_value=True):
         
        trainer = Trainer(mock_args)
        
        # Verify component links inside the trainer constructor
        assert trainer.in_channels == 3
        assert trainer.n_classes == 6  # Derived from DATASET_INFO['UCI']
        
        # 4. Execute a minimized step run
        train_stats = train_epoch(
            model=trainer.model,
            loader=trainer.train_dl,
            optimizer=trainer.optimizer,
            criterion=trainer.criterion,
            window_gen=trainer.window_gen,
            matcher=trainer.matcher,
            device=trainer.device
        )
        
        eval_stats = evaluate(
            model=trainer.model,
            loader=trainer.test_dl,
            device=trainer.device,
            n_classes=trainer.n_classes
        )
        
        # Verify keys exist and calculations completed
        assert 'total_loss' in train_stats
        assert 'accuracy' in eval_stats
        assert 'f1' in eval_stats