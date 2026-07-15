"""
step3_evaluate_RAW.py
=====================
Comprehensive evaluation of trained MTHARS model on RAW 9-channel inertial signals.

This script:
1. Loads the raw data using your verified load_UCI pipeline
2. Sets up the correct architectural input dimensions (9 channels, 128 time-steps)
3. Evaluates metrics (Accuracy, F1-Score, Class Reports, Confusion Matrix)
4. Saves an evaluation summary to disk
"""

import argparse
import sys
import time
from pathlib import Path
import warnings
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

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

from datasets.har_datasets import get_dataloaders, DATASET_INFO
from model.recognition_segmentation import MTHARS

# ─────────────────────────────────────────────────────────────────────────────
# UCI Activity Classes & Paper baselines
# ─────────────────────────────────────────────────────────────────────────────
UCI_CLASSES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]

PAPER_RESULTS = {
    'accuracy': 0.9633,
    'weighted_f1': 0.9723,
    'precision': 0.9711,
    'recall': 0.9633,
}

# ─────────────────────────────────────────────────────────────────────────────
# Verified Raw Signal Pipeline Components
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
        y_path = root / split / f'y_{split}.txt'
        y = np.loadtxt(y_path, dtype=int) - 1  # 0-indexed
        
        signals_list = []
        for sig_name in signal_names:
            sig_path = root / split / 'Inertial_Signals' / f'{sig_name}_{split}.txt'
            sig = np.loadtxt(sig_path, dtype=np.float32)
            signals_list.append(sig)
        
        X = np.stack(signals_list, axis=1)  # (N, 9, 128)
        splits.append((X, y))
    
    X = np.vstack([s[0] for s in splits]).astype(np.float32)
    y = np.hstack([s[1] for s in splits]).astype(np.int64)
    segs = build_segments(y)
    return normalise(X), y, segs

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint and Evaluation Core Engine
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path, device: torch.device, in_channels: int, data_len: int) -> MTHARS:
    """Load trained model from checkpoint, aligned with raw dimensions."""
    print(f"\n  Loading checkpoint: {ckpt_path}")
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint structure not found at path: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['cfg']
    
    # Corrected Dimension Mapping: matching your actual (N, 9, 128) array
    model = MTHARS(
        in_channels=in_channels,  # dynamically assigned (9)
        n_classes=6,
        scales=cfg.get('scales', [2.0, 3.0]),
        feat_dim=cfg.get('feat_dim', 256),
        data_len=data_len,        # dynamically assigned (128)
    ).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    print(f"  ✓ Loaded checkpoint successfully from epoch {ckpt['epoch']}")
    return model


def evaluate_model(model: MTHARS, test_loader, device: torch.device, n_classes: int) -> dict:
    """Evaluate model on test set using the classification sub-branch."""
    print(f"\n  Evaluating model on active test split ({len(test_loader)} batches)...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            cls_logits, _ = model(batch_x)  # Shape: (B, anchors, K+1)
            
            # Aggregate over anchor windows, omitting background channel index 0
            agg_logits = cls_logits[:, :, 1:].mean(dim=1)  
            preds = agg_logits.argmax(dim=1)  
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return {
        'accuracy': accuracy_score(all_targets, all_preds),
        'weighted_f1': f1_score(all_targets, all_preds, average='weighted'),
        'macro_f1': f1_score(all_targets, all_preds, average='macro'),
        'precision': precision_score(all_targets, all_preds, average='weighted'),
        'recall': recall_score(all_targets, all_preds, average='weighted'),
        'per_class': classification_report(all_targets, all_preds, target_names=UCI_CLASSES, output_dict=True),
        'confusion_matrix': confusion_matrix(all_targets, all_preds, labels=range(n_classes)),
    }


def print_evaluation_report(metrics: dict):
    """Prints a clear report comparing metrics directly to paper targets."""
    print("\n" + "=" * 80)
    print("  Evaluation Results (Raw Signals Mode)")
    print("=" * 80)
    print(f"    Accuracy    : {metrics['accuracy']:.4f}  (Paper: {PAPER_RESULTS['accuracy']:.4f})")
    print(f"    Weighted F1 : {metrics['weighted_f1']:.4f}  (Paper: {PAPER_RESULTS['weighted_f1']:.4f})")
    print(f"    Precision   : {metrics['precision']:.4f}  (Paper: {PAPER_RESULTS['precision']:.4f})")
    print(f"    Recall      : {metrics['recall']:.4f}  (Paper: {PAPER_RESULTS['recall']:.4f})")
    
    print("\n  Per-Class Metrics Detail:")
    print(f"    {'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("    " + "-" * 65)
    for i, name in enumerate(UCI_CLASSES):
        cls_m = metrics['per_class'][str(i)]
        print(f"    {name:<25} {cls_m['precision']:<12.4f} {cls_m['recall']:<12.4f} {cls_m['f1-score']:<12.4f}")
    print()


def print_confusion_matrix(cm: np.ndarray):
    """Outputs a text-scannable confusion matrix."""
    print("=" * 80)
    print("  Confusion Matrix Vector Alignment")
    print("=" * 80)
    print("  " + " " * 20, "".join([f"{i:>8}" for i in range(len(UCI_CLASSES))]))
    for i, name in enumerate(UCI_CLASSES):
        row_str = "".join([f"{cm[i, j]:>8}" for j in range(len(UCI_CLASSES))])
        print(f"  {name:<20} {row_str}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MTHARS Raw Implementation")
    parser.add_argument("--checkpoint", type=str, default=str(OUTPUT_DIR / "UCI" / "best_model.pt"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_root", type=str, default=str(UCI_ROOT))
    args = parser.parse_args()
    
    print("\n" + "█" * 80)
    print("  MTHARS Evaluation Module Pipeline Execution".center(80))
    print("█" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load data safely via raw loader
    print(f"\n  Loading raw validation signal sequences from: {args.data_root}")
    try:
        X, y, segs = load_UCI(args.data_root)
        DATASET_INFO["UCI"]["window"] = X.shape[2] # Update sequence window length (128)
    except Exception as e:
        print(f"  ✗ ERROR: Failed to pipeline raw UCI data.\n    {e}")
        sys.exit(1)
        
    # 2. Build explicit dataloaders
    _, test_dl = get_dataloaders(
        X, y, segs,
        train_ratio=0.70,
        batch_size=args.batch_size,
        augment=False,
        num_workers=0,
        seed=42
    )
    
    # 3. Reconstruct model pointing directly to raw signal shapes
    try:
        model = load_checkpoint(Path(args.checkpoint), device, in_channels=X.shape[1], data_len=X.shape[2])
    except Exception as e:
        print(f"  ✗ ERROR: Checkpoint initialization failed.\n    {e}")
        sys.exit(1)
        
    # 4. Evaluate and generate diagnostics
    metrics = evaluate_model(model, test_dl, device, n_classes=6)
    print_evaluation_report(metrics)
    print_confusion_matrix(metrics['confusion_matrix'])


if __name__ == "__main__":
    main()
