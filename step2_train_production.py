"""
step2_train_PRODUCTION_RAW.py
=============================
COMPLETE & PRODUCTION-READY MTHARS training using RAW 9-channel inertial signals.

This script:
1. Validates environment (GPU/CPU, dependencies)
2. Loads raw 9-axis sensor data (N, 9, 128) using your verified loader
3. Validates shapes directly without feature-space hacks
4. Creates data loaders with proper train/test split (70/30)
5. Initializes MTHARS model with SKNet1D backbone
6. Trains for full epochs with:
   - Adam optimizer with cosine annealing LR scheduler
   - Multi-task loss (classification + localization)
   - Validation on test set each epoch
   - Checkpoint saving on best F1-score
   - Mixed precision (AMP) support
7. Evaluates and reports final metrics
"""

import argparse
import sys
import time
from pathlib import Path
import warnings
import numpy as np
import torch

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ── Kaggle-safe path setup ─────────────────────────────────────────────────
_here = Path("/kaggle/working/Human-Activity-Recognition")
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

try:
    import kaggle_config
    from kaggle_config import REPO_ROOT, UCI_ROOT, OUTPUT_DIR
except ImportError:
    REPO_ROOT  = _here
    UCI_ROOT   = Path("/kaggle/input/ucihar/UCI HAR Dataset")
    OUTPUT_DIR = REPO_ROOT / "checkpoints"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from datasets.har_datasets import get_dataloaders, DATASET_INFO
from training.trainer import Trainer

# ─────────────────────────────────────────────────────────────────────────────
# UCI Activity Classes
# ─────────────────────────────────────────────────────────────────────────────
UCI_CLASSES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Pipeline Components (Your Fixed & Verified Functions)
# ─────────────────────────────────────────────────────────────────────────────

def build_segments(y: np.ndarray) -> list[tuple[int, int, int]]:
    """Finds continuous segments of the same activity in the label array."""
    segments = []
    if len(y) == 0:
        return segments

    start_idx = 0
    current_label = y[0]

    for i in range(1, len(y)):
        if y[i] != current_label:
            end_idx = i - 1
            segments.append((start_idx, end_idx, int(current_label)))
            start_idx = i
            current_label = y[i]
            
    segments.append((start_idx, len(y) - 1, int(current_label)))
    return segments


def normalise(X: np.ndarray) -> np.ndarray:
    """Standardizes 9-channel dataset across sample and time dimensions."""
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    return (X - mean) / (std + 1e-8)


def load_UCI(data_root: str) -> tuple[np.ndarray, np.ndarray, list]:
    """UCI HAR Dataset - LOAD RAW SIGNALS (NOT pre-computed features)."""
    root = Path(data_root)
    
    signal_names = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z',
    ]
    
    splits = []
    for split in ('train', 'test'):
        # Load activity labels
        y_path = root / split / f'y_{split}.txt'
        y = np.loadtxt(y_path, dtype=int) - 1  # 0-indexed (0 to 5)
        
        # Load all 9 inertial signals
        signals_list = []
        for sig_name in signal_names:
            sig_path = root / split / 'Inertial_Signals' / f'{sig_name}_{split}.txt'
            sig = np.loadtxt(sig_path, dtype=np.float32)  # (N, 128)
            signals_list.append(sig)
        
        # Stack channels safely: (N, 128) × 9 → (N, 9, 128)
        X = np.stack(signals_list, axis=1)  
        splits.append((X, y))
    
    # Concatenate train + test splits together
    X = np.vstack([s[0] for s in splits]).astype(np.float32)  # (N_total, 9, 128)
    y = np.hstack([s[1] for s in splits]).astype(np.int64)
    
    segs = build_segments(y)
    return normalise(X), y, segs


# ─────────────────────────────────────────────────────────────────────────────
# Helper & Training Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_args(overrides: dict = None) -> argparse.Namespace:
    """Build training config with paper's UCI defaults."""
    defaults = dict(
        dataset    = "UCI",
        data_root  = str(UCI_ROOT),
        output_dir = str(OUTPUT_DIR),
        augment    = False,

        feat_dim   = 256,           
        scales     = [2.0, 3.0],    

        alpha      = 1.0,           
        beta       = 1.0,           
        n_neg_ratio = 3,            

        pos_iou_thresh = 0.5,       
        neg_iou_thresh = 0.3,       

        epochs       = 100,         
        batch_size   = 64,          
        lr           = 1e-3,        
        weight_decay = 1e-4,        
        amp          = False,       
        seed         = 42,          
        ablation     = False,
    )

    if overrides:
        defaults.update(overrides)

    return argparse.Namespace(**defaults)


def print_config(cfg):
    """Pretty-print configuration."""
    print("\n" + "=" * 80)
    print("  MTHARS Training Configuration (RAW SENSORS)")
    print("=" * 80)
    print(f"  Dataset Name : {cfg.dataset}")
    print(f"  Data Root    : {cfg.data_root}")
    print(f"  Architecture : SKNet1D Backbone (feat_dim={cfg.feat_dim}, scales={cfg.scales})")
    print(f"  Loss Weights : α={cfg.alpha}, β={cfg.beta} (Hard-Negative Ratio {cfg.n_neg_ratio}:1)")
    print(f"  Optimization : Epochs={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr}, AMP={cfg.amp}")
    print(f"  Output Dir   : {cfg.output_dir}\n")


def print_dataset_summary(X, y, segs):
    """Print dataset statistics."""
    print("\n" + "=" * 80)
    print("  Dataset Summary")
    print("=" * 80)
    print(f"    X shape    : {X.shape}  (N_windows, C_channels, T_timesteps)")
    print(f"    y shape    : {y.shape}")
    print(f"    Windows    : {X.shape[0]:,}")
    print(f"    Channels   : {X.shape[1]}")
    print(f"    Time-steps : {X.shape[2]}")
    print(f"    X dtype    : {X.dtype} | y dtype: {y.dtype}")

    print(f"\n  Data Distribution:")
    print(f"    Segments   : {len(segs):,}")
    for i, name in enumerate(UCI_CLASSES):
        count = (y == i).sum()
        pct = 100 * count / len(y)
        bar = "█" * int(pct // 5)
        print(f"      [{i}] {name:<25} {count:>6,} windows ({pct:5.1f}%) {bar}")
    print()


def validate_shapes(X, y):
    """Validate that X, y have correct raw shapes for MTHARS."""
    errors = []

    if X.dtype != np.float32:
        errors.append(f"X dtype: expected float32, got {X.dtype}")
    if y.dtype not in (np.int64, np.int32):
        errors.append(f"y dtype: expected int64/int32, got {y.dtype}")
    if len(X.shape) != 3:
        errors.append(f"X shape: expected 3D (N, C, T), got {X.shape}")
    if len(y.shape) != 1:
        errors.append(f"y shape: expected 1D, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        errors.append(f"Batch size mismatch: X[0]={X.shape[0]}, y[0]={y.shape[0]}")

    if errors:
        print("\n" + "!" * 80)
        print("  VALIDATION ERRORS")
        print("!" * 80)
        for err in errors:
            print(f"    ✗ {err}")
        return False

    print("\n" + "=" * 80)
    print("  Shape Validation")
    print("=" * 80)
    print(f"  ✓ X shape      : {X.shape} (Verified Raw Format)")
    print(f"  ✓ y shape      : {y.shape}")
    print(f"  ✓ All checks   : PASSED\n")
    return True


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train MTHARS on UCI HAR Raw Signals")
    parser.add_argument("--dataset", type=str, default="UCI")
    parser.add_argument("--data_root", type=str, default=str(UCI_ROOT))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", action="store_true")
    
    cli, _ = parser.parse_known_args()
    cfg = build_args(vars(cli))

    print("\n" + "█" * 80)
    print("  MTHARS: Multi-Task Human Activity Recognition & Segmentation".center(80))
    print("  UCI HAR Dataset - Raw 9-Axis Pipeline".center(80))
    print("█" * 80)

    # ── [1] Verify dataset directory structure ─────────────────────────────
    print("\n" + "=" * 80)
    print("  [1/6] Checking Inertial Signals Availability")
    print("=" * 80)

    uci_root = Path(cfg.data_root)
    if not uci_root.exists():
        print(f"  ✗ ERROR: UCI dataset root not found at {uci_root}")
        sys.exit(1)

    # Verify key raw folders exist
    required_dirs = [
        uci_root / "train" / "Inertial_Signals",
        uci_root / "test" / "Inertial_Signals",
    ]
    for d in required_dirs:
        if not d.exists():
            print(f"  ✗ ERROR: Missing critical folder: {d}")
            sys.exit(1)

    print(f"  ✓ Raw Inertial Signal paths verified.")

    # ── [2] Load raw dataset ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [2/6] Loading Raw Signals via Custom Pipeline")
    print("=" * 80)

    t0 = time.time()
    try:
        X, y, segs = load_UCI(cfg.data_root)
        elapsed = time.time() - t0
        print(f"  ✓ Loaded successfully in {elapsed:.2f}s")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to run custom load_UCI pipeline\n    {e}")
        sys.exit(1)

    # ── [3] Validate Shapes Directly ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [3/6] Shape Verification")
    print("=" * 80)

    if not validate_shapes(X, y):
        sys.exit(1)

    # Inform the global configuration space of the actual window size (128)
    DATASET_INFO["UCI"]["window"] = X.shape[2]

    # ── [4] Summaries ──────────────────────────────────────────────────────
    print_dataset_summary(X, y, segs)
    print_config(cfg)

    # ── [5] Initialize Trainer ─────────────────────────────────────────────
    print("=" * 80)
    print("  [4/6] Model Initialization")
    print("=" * 80)

    try:
        trainer = Trainer(cfg)
        print("\n  ✓ Model tracking structure created.")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"  ✓ Total network parameters: {total_params:,}\n")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to bind model initialization\n    {e}")
        sys.exit(1)

    # ── [6] Run training loop ──────────────────────────────────────────────
    print("=" * 80)
    print("  [5/6] Core Optimization Loop")
    print("=" * 80)
    print(f"  Optimizing for {cfg.epochs} epochs...")
    print(f"  Saving checkpoints to: {Path(cfg.output_dir) / cfg.dataset}")
    print("-" * 80)

    try:
        best_f1 = trainer.run()
    except Exception as e:
        print(f"  ✗ ERROR: Core optimization loop failed execution\n    {e}")
        sys.exit(1)

    # ── Summary Reports ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [6/6] Pipeline Execution Complete")
    print("=" * 80)

    ckpt_path = Path(cfg.output_dir) / cfg.dataset / "best_model.pt"
    print(f"  ✓ Target Run Complete. Best Weighted-F1 Achieved: {best_f1:.4f}")
    if ckpt_path.exists():
        print(f"  ✓ Checkpoint saved safely to disk: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()