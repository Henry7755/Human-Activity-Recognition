"""
step2_train_PRODUCTION.py
=========================
COMPLETE & PRODUCTION-READY MTHARS training for Kaggle + UCI HAR dataset.

This script:
1. Validates environment (GPU/CPU, dependencies)
2. Loads and preprocesses UCI HAR dataset
3. Fixes shape mismatches (N, 561, 1) → (N, 1, 561)
4. Creates data loaders with proper train/test split (70/30)
5. Initializes MTHARS model with SKNet1D backbone
6. Trains for full epochs with:
   - Adam optimizer with cosine annealing LR scheduler
   - Multi-task loss (classification + localization)
   - Validation on test set each epoch
   - Checkpoint saving on best F1-score
   - Mixed precision (AMP) support
7. Evaluates and reports final metrics

Paper Reference:
  "Multi-Task Learning for Human Activity Recognition and Segmentation"
  Section IV-E: Recognition Results (Table V)

Usage:
    # Full training (paper defaults)
    python step2_train_production.py --epochs 100 --batch_size 64

    # Quick test
    python step2_train_production.py --epochs 10 --batch_size 32

    # With GPU optimization
    python step2_train_production.py --epochs 100 --batch_size 64 --amp

    # Custom learning rate
    python step2_train_production.py --epochs 100 --lr 5e-4
"""

import argparse
import sys
import time
from pathlib import Path
import warnings

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

import numpy as np
import torch

from datasets.har_datasets import load_dataset, get_dataloaders, DATASET_INFO
from training.trainer import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# UCI Activity Classes
# ─────────────────────────────────────────────────────────────────────────────
UCI_CLASSES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def build_args(overrides: dict = None) -> argparse.Namespace:
    """Build training config with paper's UCI defaults (Table V)."""
    defaults = dict(
        # ── Data ──────────────────────────────────────────────
        dataset    = "UCI",
        data_root  = str(UCI_ROOT),
        output_dir = str(OUTPUT_DIR),
        augment    = False,

        # ── Model ─────────────────────────────────────────────
        feat_dim   = 256,           # backbone feature dimension
        scales     = [2.0, 3.0],    # Table VIII: best scales

        # ── Loss (paper defaults, Table VII) ──────────────────
        alpha      = 1.0,           # classification loss weight
        beta       = 1.0,           # localisation loss weight
        n_neg_ratio = 3,            # hard-negative mining ratio

        # ── IOU thresholds (Section III-B) ───────────────────
        pos_iou_thresh = 0.5,       # positive anchor threshold
        neg_iou_thresh = 0.3,       # negative anchor threshold

        # ── Training ──────────────────────────────────────────
        epochs       = 100,         # full training epochs
        batch_size   = 64,          # batch size
        lr           = 1e-3,        # initial learning rate
        weight_decay = 1e-4,        # L2 regularization
        amp          = False,       # mixed precision (set True for speed)
        seed         = 42,          # random seed

        # ── Ablation flag ─────────────────────────────────────
        ablation = False,
    )

    if overrides:
        defaults.update(overrides)

    return argparse.Namespace(**defaults)


def print_config(cfg):
    """Pretty-print configuration."""
    print("\n" + "=" * 80)
    print("  MTHARS Training Configuration")
    print("=" * 80)
    print(f"\n  Dataset:")
    print(f"    Name       : {cfg.dataset}")
    print(f"    Root       : {cfg.data_root}")
    print(f"    Split      : 70% train, 30% test")

    print(f"\n  Model (SKNet1D Backbone):")
    print(f"    feat_dim   : {cfg.feat_dim}")
    print(f"    scales     : {cfg.scales}")

    print(f"\n  Loss:")
    print(f"    α (class)  : {cfg.alpha}")
    print(f"    β (offset) : {cfg.beta}")
    print(f"    neg_ratio  : {cfg.n_neg_ratio}:1")

    print(f"\n  Training:")
    print(f"    Epochs     : {cfg.epochs}")
    print(f"    Batch size : {cfg.batch_size}")
    print(f"    LR         : {cfg.lr}")
    print(f"    Weight dec : {cfg.weight_decay}")
    print(f"    AMP        : {cfg.amp}")
    print(f"    Seed       : {cfg.seed}")

    print(f"\n  Output:")
    print(f"    Directory  : {cfg.output_dir}")
    print()


def print_dataset_summary(X, y, segs):
    """Print dataset statistics."""
    print("\n" + "=" * 80)
    print("  Dataset Summary")
    print("=" * 80)
    print(f"\n  Shape & Size:")
    print(f"    X shape    : {X.shape}  (N_windows, C_channels, T_timesteps)")
    print(f"    y shape    : {y.shape}")
    print(f"    Windows    : {X.shape[0]:,}")
    print(f"    Channels   : {X.shape[1]}")
    print(f"    Time-steps : {X.shape[2]}")

    print(f"\n  Data Type:")
    print(f"    X dtype    : {X.dtype}")
    print(f"    y dtype    : {y.dtype}")

    print(f"\n  Data Distribution:")
    print(f"    Segments   : {len(segs):,}")
    print(f"    Classes    : {len(UCI_CLASSES)} activities")
    for i, name in enumerate(UCI_CLASSES):
        count = (y == i).sum()
        pct = 100 * count / len(y)
        bar = "█" * int(pct // 5)
        print(f"      [{i}] {name:<25} {count:>6,} windows ({pct:5.1f}%) {bar}")

    print(f"\n  Normalization:")
    print(f"    X min/max  : [{X.min():7.3f}, {X.max():7.3f}]")
    print()


def validate_shapes(X, y):
    """Validate that X, y have correct shapes for MTHARS."""
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
        print()
        return False

    print("\n" + "=" * 80)
    print("  Shape Validation")
    print("=" * 80)
    print(f"  ✓ X shape      : {X.shape} (float32)")
    print(f"  ✓ y shape      : {y.shape} (int64)")
    print(f"  ✓ All checks   : PASSED")
    print()

    return True


def fix_uci_shape(X, y):
    """
    Fix UCI's shape inconsistency.

    UCI provides pre-computed features in shape (N, 561, 1) but MTHARS
    expects sensor data in shape (N, C, T). We transpose to (N, 1, 561)
    so the model treats 561 features as time-steps and 1 as channel count.

    This is a pragmatic workaround since UCI only provides pre-computed features.
    Ideally, we'd use raw 9-axis sensor data for better model performance.

    Args:
        X: (N, 561, 1) pre-computed features
        y: (N,) activity labels

    Returns:
        X: (N, 1, 561) transposed features
        y: (N,) labels (unchanged, but as int64)
    """
    print("\n" + "=" * 80)
    print("  Shape Transformation")
    print("=" * 80)
    print(f"  Original X     : {X.shape}  (pre-computed features)")
    print(f"  Transposing    : (N, 561, 1) → (N, 1, 561)")

    if X.shape[2] == 1 and X.shape[1] > 1:
        X = X.transpose(0, 2, 1)
        print(f"  After transpose: {X.shape}")
    else:
        print(f"  [WARNING] Shape unexpected: {X.shape}")

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    print(f"  Final X dtype  : {X.dtype}")
    print(f"  Final y dtype  : {y.dtype}")
    print()

    return X, y


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train MTHARS on UCI HAR (Production Ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full training:      python step2_train_production.py --epochs 100
  Quick test:         python step2_train_production.py --epochs 10 --batch_size 32
  With GPU speedup:   python step2_train_production.py --epochs 100 --amp
  Custom LR:          python step2_train_production.py --lr 5e-4
        """
    )

    # Data arguments
    parser.add_argument("--dataset", type=str, default="UCI", help="Dataset name")
    parser.add_argument("--data_root", type=str, default=str(UCI_ROOT), help="Dataset root path")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for checkpoints")
    parser.add_argument("--augment", action="store_true", help="Enable Gaussian noise augmentation")

    # Model arguments
    parser.add_argument("--feat_dim", type=int, default=256, help="Backbone feature dimension")
    parser.add_argument("--scales", type=float, nargs="+", default=[2.0, 3.0], help="Multi-scale window sizes")

    # Loss arguments
    parser.add_argument("--alpha", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Localization loss weight")
    parser.add_argument("--n_neg_ratio", type=int, default=3, help="Hard-negative mining ratio")

    # IOU thresholds
    parser.add_argument("--pos_iou_thresh", type=float, default=0.5, help="Positive anchor IOU threshold")
    parser.add_argument("--neg_iou_thresh", type=float, default=0.3, help="Negative anchor IOU threshold")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    cli, _ = parser.parse_known_args()
    cfg = build_args(vars(cli))

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  MTHARS: Multi-Task Human Activity Recognition & Segmentation".center(78) + "█")
    print("█" + "  UCI HAR Dataset - Full Training".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # ── [1] Verify dataset exists ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [1/7] Checking Dataset Availability")
    print("=" * 80)

    uci_root = Path(cfg.data_root)
    if not uci_root.exists():
        print(f"\n  ✗ ERROR: UCI dataset not found at {uci_root}")
        print(f"  Please upload 'ucihar' dataset to Kaggle input first!")
        sys.exit(1)

    required_files = [
        uci_root / "train" / "X_train.txt",
        uci_root / "train" / "y_train.txt",
        uci_root / "test" / "X_test.txt",
        uci_root / "test" / "y_test.txt",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"\n  ✗ ERROR: Missing files:")
        for f in missing:
            print(f"    - {f}")
        sys.exit(1)

    print(f"  ✓ Dataset found at : {uci_root}")
    print(f"  ✓ All required files present")

    # ── [2] Load dataset ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [2/7] Loading Dataset")
    print("=" * 80)

    t0 = time.time()
    try:
        X, y, segs = load_dataset("UCI", cfg.data_root)
        elapsed = time.time() - t0
        print(f"\n  ✓ Loaded in {elapsed:.2f}s")
        print(f"    X shape: {X.shape}")
        print(f"    y shape: {y.shape}")
        print(f"    Segments: {len(segs)}")
    except Exception as e:
        print(f"\n  ✗ ERROR: Failed to load dataset")
        print(f"    {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── [3] Fix shape ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  [3/7] Preprocessing")
    print("=" * 80)

    X, y = fix_uci_shape(X, y)

    if not validate_shapes(X, y):
        sys.exit(1)

    # Patch DATASET_INFO so Trainer uses correct window size
    DATASET_INFO["UCI"]["window"] = X.shape[2]

    # ── [4] Print dataset summary ──────────────────────────────────────────
    print_dataset_summary(X, y, segs)

    # ── [5] Print configuration ────────────────────────────────────────────
    print_config(cfg)

    # ── [6] Initialize trainer ─────────────────────────────────────────────
    print("=" * 80)
    print("  [6/7] Model Initialization")
    print("=" * 80)

    try:
        trainer = Trainer(cfg)
        print("\n  ✓ Model created successfully")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"  ✓ Total parameters: {total_params:,}")
        print()
    except Exception as e:
        print(f"\n  ✗ ERROR: Failed to initialize model")
        print(f"    {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── [7] Run training ───────────────────────────────────────────────────
    print("=" * 80)
    print("  [7/7] Training")
    print("=" * 80)
    print(f"\n  Starting {cfg.epochs} epochs of training...")
    print(f"  Checkpoint directory: {Path(cfg.output_dir) / cfg.dataset}")
    print(f"  {'Epoch':<8} {'Loss':<12} {'Conf':<12} {'Loc':<12} {'Acc':<10} {'F1':<10} {'Time':<8}")
    print("-" * 80)

    try:
        best_f1 = trainer.run()
    except Exception as e:
        print(f"\n  ✗ ERROR: Training failed")
        print(f"    {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Training Complete")
    print("=" * 80)

    ckpt_path = Path(cfg.output_dir) / cfg.dataset / "best_model.pt"

    print(f"\n  Results:")
    print(f"    Best Weighted-F1  : {best_f1:.4f}")
    print(f"    Paper target      : 0.9723  (Table V)")
    print(f"    Performance gap   : {100*(best_f1-0.9723):+.2f}%")

    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / 1e6
        print(f"\n  Checkpoint:")
        print(f"    Path  : {ckpt_path}")
        print(f"    Size  : {size_mb:.1f} MB")
    else:
        print(f"\n  ✗ WARNING: Checkpoint file not found")

    print(f"\n  Next steps:")
    print(f"    1. Evaluate: python step3_evaluate.py")
    print(f"    2. Infer:    python step4_inference.py")

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  Training completed successfully! 🎉".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
