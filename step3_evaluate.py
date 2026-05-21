"""
step3_evaluate.py
=================
Comprehensive evaluation of trained MTHARS model on UCI HAR test set.

This script:
1. Loads the best trained checkpoint
2. Evaluates on test set with multiple metrics:
   - Accuracy
   - Weighted F1-Score
   - Per-class metrics (precision, recall, F1)
   - Confusion matrix
3. Compares against paper baseline
4. Generates evaluation report
5. Creates confusion matrix visualization

Usage:
    python step3_evaluate.py
    python step3_evaluate.py --checkpoint checkpoints/UCI/best_model.pt
"""

import argparse
import sys
from pathlib import Path
import warnings

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

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from datasets.har_datasets import load_dataset, get_dataloaders, DATASET_INFO
from model.recognition_segmentation import MTHARS


# ─────────────────────────────────────────────────────────────────────────────
# UCI Activity Classes
# ─────────────────────────────────────────────────────────────────────────────
UCI_CLASSES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]

# Paper's reported results (Table V)
PAPER_RESULTS = {
    'accuracy': 0.9633,
    'weighted_f1': 0.9723,
    'precision': 0.9711,
    'recall': 0.9633,
}


def fix_uci_shape(X, y):
    """Fix UCI's shape (N, 561, 1) → (N, 1, 561)."""
    if X.shape[2] == 1 and X.shape[1] > 1:
        X = X.transpose(0, 2, 1)
    return X.astype(np.float32), y.astype(np.int64)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> MTHARS:
    """Load trained model from checkpoint."""
    print(f"\n  Loading checkpoint: {ckpt_path}")
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Reconstruct model from config
    cfg = ckpt['cfg']
    model = MTHARS(
        in_channels=1,  # UCI has 1 channel after transposition
        n_classes=6,
        scales=cfg.get('scales', [2.0, 3.0]),
        feat_dim=cfg.get('feat_dim', 256),
        data_len=561,
    ).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    print(f"  ✓ Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  ✓ Best F1 during training: {ckpt['f1']:.4f}")
    
    return model


def evaluate_model(model: MTHARS, test_loader, device: torch.device, n_classes: int) -> dict:
    """
    Evaluate model on test set.
    
    Uses the recognition branch only: aggregates class probabilities
    across all anchor windows and predicts the dominant class.
    """
    print(f"\n  Evaluating on test set ({len(test_loader)} batches)...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            cls_logits, _ = model(batch_x)  # (B, na, K+1)
            
            # Aggregate: mean logits over all anchor windows, skip background (col 0)
            agg_logits = cls_logits[:, :, 1:].mean(dim=1)  # (B, K)
            preds = agg_logits.argmax(dim=1)  # (B,)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    
    # Per-class metrics
    per_class_report = classification_report(
        all_targets, all_preds,
        target_names=UCI_CLASSES,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(n_classes))
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'per_class': per_class_report,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_targets': all_targets,
    }


def print_evaluation_report(metrics: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 80)
    print("  Evaluation Results")
    print("=" * 80)
    
    print(f"\n  Overall Metrics:")
    print(f"    Accuracy       : {metrics['accuracy']:.4f}  (paper: {PAPER_RESULTS['accuracy']:.4f})")
    print(f"    Weighted F1    : {metrics['weighted_f1']:.4f}  (paper: {PAPER_RESULTS['weighted_f1']:.4f})")
    print(f"    Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"    Precision      : {metrics['precision']:.4f}  (paper: {PAPER_RESULTS['precision']:.4f})")
    print(f"    Recall         : {metrics['recall']:.4f}  (paper: {PAPER_RESULTS['recall']:.4f})")
    
    # Performance gaps
    acc_gap = 100 * (metrics['accuracy'] - PAPER_RESULTS['accuracy'])
    f1_gap = 100 * (metrics['weighted_f1'] - PAPER_RESULTS['weighted_f1'])
    
    print(f"\n  Performance vs Paper:")
    print(f"    Accuracy gap   : {acc_gap:+.2f}%")
    print(f"    F1 gap         : {f1_gap:+.2f}%")
    
    print(f"\n  Per-Class Metrics:")
    print(f"    {'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("    " + "-" * 76)
    
    for i, class_name in enumerate(UCI_CLASSES):
        class_metrics = metrics['per_class'][str(i)]
        print(
            f"    {class_name:<25} "
            f"{class_metrics['precision']:<12.4f} "
            f"{class_metrics['recall']:<12.4f} "
            f"{class_metrics['f1-score']:<12.4f} "
            f"{int(class_metrics['support']):<10}"
        )
    
    # Weighted average
    weighted = metrics['per_class']['weighted avg']
    print("    " + "-" * 76)
    print(
        f"    {'Weighted Avg':<25} "
        f"{weighted['precision']:<12.4f} "
        f"{weighted['recall']:<12.4f} "
        f"{weighted['f1-score']:<12.4f} "
        f"{int(weighted['support']):<10}"
    )
    
    print()


def print_confusion_matrix(cm: np.ndarray):
    """Pretty-print confusion matrix."""
    print("=" * 80)
    print("  Confusion Matrix")
    print("=" * 80)
    
    print(f"\n  Rows: True labels | Columns: Predicted labels\n")
    
    # Header
    print("  " + " " * 20, end="")
    for i in range(len(UCI_CLASSES)):
        print(f"{i:>8}", end="")
    print()
    
    # Matrix
    for i, class_name in enumerate(UCI_CLASSES):
        print(f"  {class_name:<20}", end="")
        for j in range(len(UCI_CLASSES)):
            count = cm[i, j]
            print(f"{count:>8}", end="")
        print()
    
    print()


def save_evaluation_report(metrics: dict, output_path: Path):
    """Save evaluation report to text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  MTHARS Evaluation Report - UCI HAR\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy       : {metrics['accuracy']:.4f}\n")
        f.write(f"  Weighted F1    : {metrics['weighted_f1']:.4f}\n")
        f.write(f"  Macro F1       : {metrics['macro_f1']:.4f}\n")
        f.write(f"  Precision      : {metrics['precision']:.4f}\n")
        f.write(f"  Recall         : {metrics['recall']:.4f}\n\n")
        
        f.write("Performance vs Paper:\n")
        acc_gap = 100 * (metrics['accuracy'] - PAPER_RESULTS['accuracy'])
        f1_gap = 100 * (metrics['weighted_f1'] - PAPER_RESULTS['weighted_f1'])
        f.write(f"  Accuracy gap   : {acc_gap:+.2f}%\n")
        f.write(f"  F1 gap         : {f1_gap:+.2f}%\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write(f"  {'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("  " + "-" * 76 + "\n")
        
        for i, class_name in enumerate(UCI_CLASSES):
            class_metrics = metrics['per_class'][str(i)]
            f.write(
                f"  {class_name:<25} "
                f"{class_metrics['precision']:<12.4f} "
                f"{class_metrics['recall']:<12.4f} "
                f"{class_metrics['f1-score']:<12.4f} "
                f"{int(class_metrics['support']):<10}\n"
            )
        
        f.write("  " + "-" * 76 + "\n")
        weighted = metrics['per_class']['weighted avg']
        f.write(
            f"  {'Weighted Avg':<25} "
            f"{weighted['precision']:<12.4f} "
            f"{weighted['recall']:<12.4f} "
            f"{weighted['f1-score']:<12.4f} "
            f"{int(weighted['support']):<10}\n"
        )
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Rows: True labels | Columns: Predicted labels\n\n")
        f.write("  " + " " * 20)
        for i in range(len(UCI_CLASSES)):
            f.write(f"{i:>8}")
        f.write("\n")
        
        cm = metrics['confusion_matrix']
        for i, class_name in enumerate(UCI_CLASSES):
            f.write(f"  {class_name:<20}")
            for j in range(len(UCI_CLASSES)):
                count = cm[i, j]
                f.write(f"{count:>8}")
            f.write("\n")
    
    print(f"  ✓ Report saved to {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate MTHARS on UCI HAR test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(OUTPUT_DIR / "UCI" / "best_model.pt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(UCI_ROOT),
        help="Dataset root directory"
    )
    
    args = parser.parse_args()
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  MTHARS Model Evaluation".center(78) + "█")
    print("█" + "  UCI HAR Dataset".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Load dataset
    print(f"\n  Loading dataset from {args.data_root}...")
    try:
        X, y, segs = load_dataset("UCI", args.data_root)
        X, y = fix_uci_shape(X, y)
        DATASET_INFO["UCI"]["window"] = X.shape[2]
        print(f"  ✓ Loaded {X.shape[0]} samples")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        sys.exit(1)
    
    # Create test loader (using 30% of data as test)
    try:
        _, test_dl = get_dataloaders(
            X, y, segs,
            train_ratio=0.70,
            batch_size=args.batch_size,
            augment=False,
            num_workers=0,
            seed=42
        )
        print(f"  ✓ Created test loader with {len(test_dl)} batches")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        sys.exit(1)
    
    # Load model
    print()
    try:
        model = load_checkpoint(Path(args.checkpoint), device)
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: {e}")
        print(f"  Please run training first: python step2_train_production.py")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate
    print()
    try:
        metrics = evaluate_model(model, test_dl, device, n_classes=6)
    except Exception as e:
        print(f"  ✗ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print_evaluation_report(metrics)
    print_confusion_matrix(metrics['confusion_matrix'])
    
    # Save report
    report_path = Path(args.checkpoint).parent / "evaluation_report.txt"
    try:
        save_evaluation_report(metrics, report_path)
    except Exception as e:
        print(f"  ✗ ERROR saving report: {e}")
    
    # Summary
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"\n  Model Performance:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1-Score: {metrics['weighted_f1']:.4f}")
    
    if metrics['weighted_f1'] >= PAPER_RESULTS['weighted_f1']:
        print(f"\n  ✓ EXCEEDS paper baseline! 🎉")
    else:
        gap = 100 * (metrics['weighted_f1'] - PAPER_RESULTS['weighted_f1'])
        print(f"\n  ℹ Within {-gap:.1f}% of paper baseline")
    
    print(f"\n  Report saved to: {report_path}")
    print("\n  Next step: python step4_inference.py\n")


if __name__ == "__main__":
    main()
