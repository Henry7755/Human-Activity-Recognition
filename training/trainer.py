"""
training/trainer.py  (RECOGNITION-ONLY VARIANT)
=================================================
Training, evaluation, and empirical monitoring loop for the
recognition-only model (MTHARSRecognition).

CHANGED vs. the multi-task version
-----------------------------------
This implements the remaining §4.A steps against `trainer.py`:

  - `prepare_targets`: `WindowMatcher.match()` now returns 2 values
    (`matched_labels`, `pos_mask`) instead of 3 — the `all_offsets` buffer
    and everything that built/returned it is deleted.
  - `train_epoch`: `model(batch_x)` now returns `cls_logits` only (not a
    `(cls_logits, offsets)` tuple). `criterion(...)` is called with
    `(cls_logits, matched_labels, pos_mask)` — no `pred_offsets` /
    `true_offsets` args, matching `RecognitionLoss.forward`'s new
    signature. `total_stats` drops the `loc_loss` key.
  - `evaluate`: UNCHANGED. This function was already recognition-only in
    the multi-task codebase (`cls_logits, _ = model(batch_x)` — it always
    discarded the offset output). Recognition-only `model(batch_x)` now
    simply returns `cls_logits` directly instead of a tuple, so the single
    line `cls_logits = model(batch_x)` replaces the old unpacking.
  - `Trainer.__init__`: instantiates `MTHARSRecognition` (not `MTHARS`) and
    `RecognitionLoss` (not `MTHARSLoss`); drops `cfg.beta` from the
    criterion call.
  - CLI (`parse_args`): `--beta` removed. `--alpha` kept (now a no-op
    unless you want a scalar multiplier on the classification loss).
  - `run_ablation_study`: the α/β sweep (Table VII) doesn't apply anymore
    since β no longer exists — replaced with a trivial single-run smoke
    test at α=1.0 plus the original scale sweep (Table VIII), which is
    still meaningful for recognition-only (larger anchor context can still
    help classification accuracy).
  - `generate_report_image`: drops the `loc_loss` curve from the loss
    decomposition chart.

GPU SUPPORT: unchanged (multi-GPU DataParallel, AMP, non-blocking transfer).
"""

from __future__ import annotations

import argparse
import os
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backbone.sknet import SKNet1D
from datasets.har_datasets import load_dataset, get_dataloaders, DATASET_INFO
from model.recognition import MTHARSRecognition
from model.multiscale_windows import WindowGenerator, WindowMatcher
from training.losses import RecognitionLoss, WeightedF1


# ---------------------------------------------------------------------------
# GPU Utilities  (unchanged)
# ---------------------------------------------------------------------------

def get_multi_gpu_device() -> Tuple[torch.device, int]:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*80}")
        print(f"  🎯 GPU ACCELERATION ENABLED")
        print(f"{'='*80}")
        print(f"  ✓ Found {num_gpus} GPU(s) available")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

        device = torch.device('cuda:0')
        print(f"  ✓ Primary device: {device}\n")
        return device, num_gpus
    else:
        print("\n⚠ No GPU found, falling back to CPU")
        return torch.device('cpu'), 0


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Ground-truth preparation  (window → matched labels + positive mask)
# ---------------------------------------------------------------------------

def prepare_targets(window_gen: WindowGenerator,
                    matcher:    WindowMatcher,
                    gt_segments: List[List[Dict]],
                    device:      torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of GT segment lists into per-window labels and positive
    masks, ready for the loss computation.

    CHANGED: no longer builds/returns `all_offsets` — `matcher.match()`
    has a 2-value return signature in the recognition-only variant.
    """
    B  = len(gt_segments)
    na = window_gen.num_windows

    all_labels = torch.zeros(B, na, dtype=torch.long)
    all_pos    = torch.zeros(B, na, dtype=torch.bool)

    anchors = window_gen.windows.to(device)   # (na, 2) [center, length]

    for b, segs in enumerate(gt_segments):
        if not segs:
            continue

        gt_boxes = []
        gt_lbl   = []
        for s in segs:
            cx = (s['start'] + s['end']) / 2.0
            ln = float(s['end'] - s['start'] + 1)
            gt_boxes.append([cx, ln])
            gt_lbl.append(s['label'] + 1)   # 1-indexed (0=background)

        gt_boxes_t  = torch.tensor(gt_boxes,  dtype=torch.float32, device=device)
        gt_labels_t = torch.tensor(gt_lbl,    dtype=torch.long, device=device)

        lbl, pos = matcher.match(anchors, gt_boxes_t, gt_labels_t)
        all_labels[b] = lbl.cpu()
        all_pos[b]    = pos.cpu()

    return all_labels.to(device), all_pos.to(device)

# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model:       MTHARSRecognition,
                loader:      DataLoader,
                optimizer:   optim.Optimizer,
                criterion:   RecognitionLoss,
                window_gen:  WindowGenerator,
                matcher:     WindowMatcher,
                device:      torch.device,
                scaler:      Optional[GradScaler] = None,
                max_norm:    float = 1.0
                ) -> Dict[str, float]:
    """
    Run one training epoch. Supports both single-GPU and multi-GPU training.
    """
    model.train()
    total_stats: Dict[str, float] = {
        'conf_loss': 0.0, 'total_loss': 0.0, 'n_pos': 0.0
    }
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)   # (B, C, T)
        batch_y = batch_y.to(device, non_blocking=True)   # (B, T) or (B,)
        B       = batch_x.shape[0]
        T       = batch_x.shape[2]

        # ---- DYNAMIC TARGET ENGINEERING (unchanged from multi-task) ----
        # Still needed: recognition still relies on WindowMatcher assigning
        # windows to GT segments/instances so hard-negative mining and the
        # per-class cross-entropy have well-defined positives/negatives.
        gt_segs = []
        for i in range(B):
            if batch_y.dim() > 1:
                seq = batch_y[i].cpu().numpy()
                diffs = np.diff(seq, prepend=-1)
                starts = np.where(diffs != 0)[0]
                ends = np.append(starts[1:] - 1, seq.shape[0] - 1)

                sample_segs = []
                for s, e in zip(starts, ends):
                    if seq[s] != -1:
                        sample_segs.append({
                            'start': int(s),
                            'end': int(e),
                            'label': int(seq[s])
                        })

                if len(sample_segs) == 0:
                    sample_segs.append({'start': 0, 'end': T - 1, 'label': 0})
                gt_segs.append(sample_segs)
            else:
                lbl = batch_y[i].item()
                gt_segs.append([
                    {'start': 0, 'end': T - 1, 'label': lbl},
                    {'start': T // 4, 'end': (3 * T) // 4, 'label': lbl}
                ])

        matched_labels, pos_mask = prepare_targets(
            window_gen, matcher, gt_segs, device
        )

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                cls_logits = model(batch_x)
                loss, stats = criterion(cls_logits, matched_labels, pos_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_logits = model(batch_x)
            loss, stats = criterion(cls_logits, matched_labels, pos_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        for k in total_stats:
            total_stats[k] += stats.get(k, 0.0)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_stats.items()}


# ---------------------------------------------------------------------------
# Evaluation epoch — UNCHANGED logic, just no tuple to unpack anymore
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model:      MTHARSRecognition,
             loader:     DataLoader,
             device:     torch.device,
             n_classes:  int
             ) -> Dict[str, float]:
    """
    Evaluate recognition accuracy and F1. This was already recognition-only
    in the multi-task codebase; the only change is that `model(batch_x)`
    now returns `cls_logits` directly instead of `(cls_logits, offsets)`.
    """
    model.eval()
    f1_meter = WeightedF1(n_classes=n_classes)
    correct = total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        cls_logits = model(batch_x)        # (B, na, K+1) — no longer a tuple

        agg_logits = cls_logits[:, :, 1:].mean(dim=1)   # (B, K)
        preds = agg_logits.argmax(dim=1)                 # (B,)

        if batch_y.dim() > 1:
            eval_targets = torch.mode(batch_y, dim=1)[0]
        else:
            eval_targets = batch_y

        correct += (preds == eval_targets).sum().item()
        total   += eval_targets.shape[0]
        f1_meter.update(preds.cpu(), eval_targets.cpu())

    return {
        'accuracy': correct / max(total, 1),
        'f1':        f1_meter.compute(),
    }

# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates data pipeline, recognition-only backbone, telemetry, and
    graphical metric reporting execution. Multi-GPU support unchanged.
    """

    def __init__(self, cfg: argparse.Namespace, device: torch.device = None, num_gpus: int = None):
        self.cfg = cfg

        if device is None:
            self.device, self.num_gpus = get_multi_gpu_device()
        else:
            self.device = device
            self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()

        set_seed(cfg.seed)

        info = DATASET_INFO[cfg.dataset.upper().replace('-', '_')]
        self.n_classes = info['n_classes']
        self.window_t  = info['window']
        self.freq      = info['freq']

        print(f'Device     : {self.device}')
        print(f'GPUs       : {self.num_gpus}')
        print(f'Dataset    : {cfg.dataset}  ({self.n_classes} classes)')
        print(f'Window     : {self.window_t} samples @ {self.freq} Hz')

        run_id = f"opt_{cfg.optimizer}_lr_{cfg.lr}_clip_{cfg.max_norm}_warm_{cfg.warmup_epochs}"
        self.exp_dir = Path(cfg.output_dir) / cfg.dataset / run_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        with open(self.exp_dir / "hparams.json", "w") as f:
            json.dump(vars(cfg), f, indent=4)

        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "telemetry"))

        # --- Data ---
        X, y, segs = load_dataset(cfg.dataset, cfg.data_root)
        self.in_channels = X.shape[1]

        train_ratio = 0.80 if cfg.dataset.upper() == 'PAMAP2' else 0.70
        self.train_dl, self.test_dl = get_dataloaders(
            X, y, segs,
            train_ratio=train_ratio,
            batch_size=cfg.batch_size,
            augment=cfg.augment,
            num_workers=getattr(cfg, 'num_workers', 4),
            pin_memory=(self.num_gpus > 0),
        )
        print(f'Train batches: {len(self.train_dl)} | Test batches: {len(self.test_dl)}')

        # --- Model (recognition-only) ---
        self.model = MTHARSRecognition(
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            scales=cfg.scales,
            feat_dim=cfg.feat_dim,
            data_len=self.window_t,
        ).to(self.device)

        if getattr(cfg, 'use_multi_gpu', True) and self.num_gpus > 1:
            print(f"\n📦 Wrapping model for {self.num_gpus}-GPU training via DataParallel")
            self.model = nn.DataParallel(self.model)
            print(f"✓ Model wrapped and distributed across GPUs\n")
        else:
            self.model = self.model.to(self.device)

        self.window_gen = self.model.module.window_gen if isinstance(self.model, nn.DataParallel) else self.model.window_gen
        self.matcher    = WindowMatcher(
            pos_iou_thresh=cfg.pos_iou_thresh,
            neg_iou_thresh=cfg.neg_iou_thresh,
            n_neg_ratio=cfg.n_neg_ratio,
        )

        # CHANGED: RecognitionLoss instead of MTHARSLoss — no `beta` arg.
        self.criterion = RecognitionLoss(
            n_classes=self.n_classes,
            alpha=cfg.alpha,
            n_neg_ratio=cfg.n_neg_ratio,
        )

        if cfg.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=1e-6
        )
        if cfg.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[cfg.warmup_epochs]
            )
        else:
            self.scheduler = base_scheduler

        self.scaler = GradScaler() if (self.device.type == 'cuda' and cfg.amp) else None
        self.best_f1 = 0.0

    def run(self) -> float:
        cfg = self.cfg
        print(f'\nInitialization Verification. Outputs Routing to: {self.exp_dir}\n')

        # CHANGED: dropped "loc_loss" key — no localisation term to track.
        history = {"train_loss": [], "val_f1": [], "conf_loss": []}

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            train_stats = train_epoch(
                self.model, self.train_dl,
                self.optimizer, self.criterion,
                self.window_gen, self.matcher,
                self.device, self.scaler,
                max_norm=cfg.max_norm
            )
            eval_stats = evaluate(
                self.model, self.test_dl,
                self.device, self.n_classes,
            )
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Engine/Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Losses/Total_Loss', train_stats["total_loss"], epoch)
            self.writer.add_scalar('Losses/Confidence_Component', train_stats["conf_loss"], epoch)
            self.writer.add_scalar('Evaluation/Accuracy', eval_stats["accuracy"], epoch)
            self.writer.add_scalar('Evaluation/Macro_F1', eval_stats["f1"], epoch)

            history["train_loss"].append(train_stats["total_loss"])
            history["conf_loss"].append(train_stats["conf_loss"])
            history["val_f1"].append(eval_stats["f1"])

            elapsed = time.time() - t0
            print(
                f'Epoch {epoch:03d}/{cfg.epochs} '
                f'| loss {train_stats["total_loss"]:.4f} '
                f'(conf {train_stats["conf_loss"]:.4f}) '
                f'| acc {eval_stats["accuracy"]:.4f} '
                f'| F1 {eval_stats["f1"]:.4f} '
                f'| {elapsed:.1f}s'
            )

            if eval_stats['f1'] > self.best_f1:
                self.best_f1 = eval_stats['f1']
                ckpt = self.exp_dir / 'best_model.pt'

                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

                torch.save({
                    'epoch':     epoch,
                    'state_dict': model_state,
                    'f1':        self.best_f1,
                    'cfg':        vars(cfg),
                }, ckpt)
                print(f'  ✓ New best F1 {self.best_f1:.4f} saved → {ckpt}')

        self.writer.close()
        self.generate_report_image(history)
        return self.best_f1

    def generate_report_image(self, history: Dict[str, List[float]]):
        """
        Compiles training performance metadata into a PNG graphic.
        CHANGED: loss-decomposition chart no longer plots a loc_loss curve
        (there is nothing to decompose the total loss into besides conf).
        """
        epochs_range = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(epochs_range, history["train_loss"], color='black', linewidth=2, label='Total Loss')
        ax1.plot(epochs_range, history["conf_loss"], color='crimson', linestyle=':', label='Conf Loss (α)')
        ax1.set_xlabel('Training Epochs')
        ax1.set_ylabel('Loss Magnitudes')
        ax1.set_title('Recognition-Only Objective Component Decomposition')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax3 = ax2.twinx()
        p1 = ax2.plot(epochs_range, history["train_loss"], color='tab:red', alpha=0.7, label='Train Loss')
        p2 = ax3.plot(epochs_range, history["val_f1"], color='tab:blue', linewidth=2, label='Validation F1')

        ax2.set_xlabel('Training Epochs')
        ax2.set_ylabel('Loss', color='tab:red')
        ax3.set_ylabel('Macro F1 Score', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax3.tick_params(axis='y', labelcolor='tab:blue')

        plots = p1 + p2
        labels = [l.get_label() for l in plots]
        ax2.legend(plots, labels, loc='center right')
        ax2.set_title('Convergence Telemetry Profile')
        ax2.grid(True, alpha=0.3)

        gpu_info = f"GPU Mode ({self.num_gpus} GPUs)" if self.num_gpus > 0 else "CPU Mode"
        plt.suptitle(f"Execution Analysis Matrix (Recognition-Only)\nDataset: {self.cfg.dataset} | Optimizer: {self.cfg.optimizer} | {gpu_info}", fontsize=12, fontweight='bold')
        fig.tight_layout()

        report_path = self.exp_dir / 'evaluation_report.png'
        plt.savefig(report_path, dpi=150)
        plt.close()
        print(f"--> Visual Report Graph Successfully Exported to: {report_path}")


# ---------------------------------------------------------------------------
# Ablation Study Runner
# ---------------------------------------------------------------------------

def run_ablation_study(base_cfg: argparse.Namespace) -> None:
    """
    CHANGED: the original α/β sweep (Table VII) doesn't apply anymore —
    there's no β/L_loc term to weight against α/L_conf. Recognition-only
    training has one loss term, so α just rescales the whole objective and
    has no effect on optimization trajectory beyond the learning rate
    (removed as a sweep axis entirely). The scale sweep (Table VIII) is
    kept: anchor scale still affects how much temporal context each window
    sees before classification, which can still matter for accuracy.
    """
    print('\n' + '='*60)
    print('ABLATION: scale s combinations (Table VIII) — recognition-only')
    print('='*60)

    scale_combos = [
        [2.0],
        [0.5, 0.3],
        [2.0, 3.0],
        [2.0, 3.0, 4.0],
    ]
    for scales in scale_combos:
        cfg = argparse.Namespace(**vars(base_cfg))
        cfg.alpha  = 1.0
        cfg.scales = scales
        cfg.epochs = min(cfg.epochs, 30)
        print(f'\ns = {scales}')
        t = Trainer(cfg)
        f1 = t.run()
        print(f'  → F1 = {f1:.4f}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train MTHARS (recognition-only)')

    # Data
    p.add_argument('--dataset',   type=str, default='UCI',
                   choices=['SKODA','HCI','PS','WISDM','UCI',
                            'OPPORTUNITY','PAMAP2','UNIMIB_SHAR'])
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--output_dir',type=str, default='./checkpoints')
    p.add_argument('--augment',   action='store_true')

    # Model
    p.add_argument('--feat_dim', type=int,   default=256)
    p.add_argument('--scales',   type=float, nargs='+', default=[2.0, 3.0])

    # Loss
    # CHANGED: --beta removed (no localisation term). --alpha kept as an
    # optional no-op scalar for interface parity with existing scripts.
    p.add_argument('--alpha',   type=float, default=1.0)
    p.add_argument('--n_neg_ratio', type=int, default=3)

    # IOU thresholds (still used for window↔instance assignment)
    p.add_argument('--pos_iou_thresh', type=float, default=0.5)
    p.add_argument('--neg_iou_thresh', type=float, default=0.3)

    # Training
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--amp',          action='store_true', default=True, help='Enable automatic mixed precision')
    p.add_argument('--seed',         type=int,   default=42)

    p.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'],
                   help='Target optimization engine implementation type')
    p.add_argument('--max_norm', type=float, default=1.0,
                   help='Hard bound value clip maximum for backward gradient norms')
    p.add_argument('--warmup_epochs', type=int, default=5,
                   help='Linear training update introduction phase epoch duration')

    p.add_argument('--num_workers', type=int, default=4,
                   help='Number of workers for parallel data loading (optimal for Kaggle GPUs)')
    p.add_argument('--use_multi_gpu', action='store_true', default=True,
                   help='Enable multi-GPU training via DataParallel if available')

    p.add_argument('--ablation', action='store_true',
                   help='Run scale-only ablation study (recognition-only)')

    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    if args.ablation:
        run_ablation_study(args)
    else:
        trainer = Trainer(args)
        trainer.run()
