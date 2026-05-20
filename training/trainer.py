"""
training/trainer.py
====================
Training and evaluation loop for MTHARS.

Covers:
    - Section III-E  : Training Model
    - Section IV-C   : Static Sliding-Window Segmentation experiments
    - Section IV-D   : Dynamic Segmentation experiments
    - Section IV-E   : Activity Recognition experiments
    - Section IV-F   : Ablation experiments (α/β weights, scale s)

Usage
-----
    python training/trainer.py --dataset UCI --data_root /data/UCI \
           --epochs 100 --alpha 1.0 --beta 1.0 --scales 2 3
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backbone.sknet import SKNet1D
from datasets.har_datasets import load_dataset, get_dataloaders, DATASET_INFO
from model.recognition_segmentation import MTHARS, ConcatenateAlgorithm
from model.multiscale_windows import WindowGenerator, WindowMatcher, offset_encode
from training.losses import MTHARSLoss, WeightedF1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Ground-truth preparation  (window → matched labels + offsets)
# ---------------------------------------------------------------------------

def prepare_targets(window_gen: WindowGenerator,
                    matcher:    WindowMatcher,
                    gt_segments: List[List[Dict]],
                    device:     torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of GT segment lists into per-window labels, offsets,
    and positive masks, ready for the loss computation.

    Args:
        window_gen   : WindowGenerator instance (anchors in data coords)
        matcher      : WindowMatcher instance
        gt_segments  : list (B) of lists of {'start','end','label'} dicts
        device       : torch device

    Returns:
        matched_labels  : (B, na)
        true_offsets    : (B, na, 2)
        pos_mask        : (B, na) bool
    """
    B  = len(gt_segments)
    na = window_gen.num_windows

    all_labels  = torch.zeros(B, na, dtype=torch.long)
    all_offsets = torch.zeros(B, na, 2, dtype=torch.float32)
    all_pos     = torch.zeros(B, na, dtype=torch.bool)

    anchors = window_gen.windows   # (na, 2) [center, length]

    for b, segs in enumerate(gt_segments):
        if not segs:
            continue

        # Convert segments to (center, length) tensor
        gt_boxes = []
        gt_lbl   = []
        for s in segs:
            cx = (s['start'] + s['end']) / 2.0
            ln = float(s['end'] - s['start'] + 1)
            gt_boxes.append([cx, ln])
            gt_lbl.append(s['label'] + 1)   # 1-indexed (0=background)

        gt_boxes_t  = torch.tensor(gt_boxes,  dtype=torch.float32)
        gt_labels_t = torch.tensor(gt_lbl,    dtype=torch.long)

        lbl, off, pos = matcher.match(anchors, gt_boxes_t, gt_labels_t)
        all_labels[b]  = lbl
        all_offsets[b] = off
        all_pos[b]     = pos

    return (all_labels.to(device),
            all_offsets.to(device),
            all_pos.to(device))


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model:       MTHARS,
                loader:      DataLoader,
                optimizer:   optim.Optimizer,
                criterion:   MTHARSLoss,
                window_gen:  WindowGenerator,
                matcher:     WindowMatcher,
                device:      torch.device,
                scaler:      Optional[GradScaler] = None,
                gt_segs_all: Optional[List] = None
                ) -> Dict[str, float]:
    """
    Run one training epoch.

    Since HARDataset returns (x, label) pairs from *pre-segmented* windows,
    we synthesise single-activity GT segments for each window to drive the
    segmentation head.  When gt_segs_all is provided (full stream), it is
    used directly.

    Returns dict with averaged loss statistics.
    """
    model.train()
    total_stats: Dict[str, float] = {
        'conf_loss': 0.0, 'loc_loss': 0.0,
        'total_loss': 0.0, 'n_pos': 0.0
    }
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)   # (B, C, T)
        batch_y = batch_y.to(device, non_blocking=True)   # (B,)
        B       = batch_x.shape[0]
        T       = batch_x.shape[2]

        # Build pseudo GT segments: each window is one activity
        gt_segs = []
        for i in range(B):
            gt_segs.append([{
                'start': 0,
                'end':   T - 1,
                'label': batch_y[i].item()
            }])

        matched_labels, true_offsets, pos_mask = prepare_targets(
            window_gen, matcher, gt_segs, device
        )

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                cls_logits, pred_offsets = model(batch_x)
                loss, stats = criterion(cls_logits, pred_offsets,
                                        matched_labels, true_offsets, pos_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_logits, pred_offsets = model(batch_x)
            loss, stats = criterion(cls_logits, pred_offsets,
                                    matched_labels, true_offsets, pos_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        for k in total_stats:
            total_stats[k] += stats.get(k, 0.0)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_stats.items()}


# ---------------------------------------------------------------------------
# Evaluation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model:      MTHARS,
             loader:     DataLoader,
             device:     torch.device,
             n_classes:  int
             ) -> Dict[str, float]:
    """
    Evaluate recognition accuracy and F1.

    Uses the *recognition* branch only: takes argmax of the aggregated
    class probabilities across all windows in each sample.
    """
    model.eval()
    f1_meter = WeightedF1(n_classes=n_classes)
    correct = total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        cls_logits, _ = model(batch_x)        # (B, na, K+1)
        # Aggregate: sum logits over all anchor windows, then argmax
        agg_logits = cls_logits[:, :, 1:].mean(dim=1)   # (B, K)  – skip BG
        preds = agg_logits.argmax(dim=1)                 # (B,)

        correct += (preds == batch_y).sum().item()
        total   += batch_y.shape[0]
        f1_meter.update(preds.cpu(), batch_y.cpu())

    return {
        'accuracy': correct / max(total, 1),
        'f1':       f1_meter.compute(),
    }


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates dataset loading, model creation, training, and evaluation.

    Supports the ablation study parameters from Section IV-F:
        - alpha, beta   : loss weights  (Table VII)
        - scales        : window scales (Table VIII)
    """

    def __init__(self, cfg: argparse.Namespace):
        self.cfg    = cfg
        self.device = get_device()
        set_seed(cfg.seed)

        info = DATASET_INFO[cfg.dataset.upper().replace('-', '_')]
        self.n_classes = info['n_classes']
        self.window_t  = info['window']       # window size in samples
        self.freq      = info['freq']

        print(f'Device     : {self.device}')
        print(f'Dataset    : {cfg.dataset}  ({self.n_classes} classes)')
        print(f'Window     : {self.window_t} samples @ {self.freq} Hz')

        # --- Data ---
        X, y, segs = load_dataset(cfg.dataset, cfg.data_root)
        self.in_channels = X.shape[1]

        train_ratio = 0.80 if cfg.dataset.upper() == 'PAMAP2' else 0.70
        self.train_dl, self.test_dl = get_dataloaders(
            X, y, segs,
            train_ratio=train_ratio,
            batch_size=cfg.batch_size,
            augment=cfg.augment,
        )
        print(f'Train batches: {len(self.train_dl)} | '
              f'Test batches: {len(self.test_dl)}')

        # --- Model ---
        self.model = MTHARS(
            in_channels=self.in_channels,
            n_classes=self.n_classes,
            scales=cfg.scales,
            feat_dim=cfg.feat_dim,
            data_len=self.window_t,
        ).to(self.device)

        # --- Window infrastructure (mirrors the model's internal generator) ---
        self.window_gen = self.model.window_gen
        self.matcher    = WindowMatcher(
            pos_iou_thresh=cfg.pos_iou_thresh,
            neg_iou_thresh=cfg.neg_iou_thresh,
            n_neg_ratio=cfg.n_neg_ratio,
        )

        # --- Loss ---
        self.criterion = MTHARSLoss(
            n_classes=self.n_classes,
            alpha=cfg.alpha,
            beta=cfg.beta,
            n_neg_ratio=cfg.n_neg_ratio,
        )

        # --- Optimiser & scheduler ---
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=1e-6
        )

        # Mixed precision
        self.scaler = GradScaler() if (self.device.type == 'cuda'
                                        and cfg.amp) else None

        self.best_f1   = 0.0
        self.save_dir  = Path(cfg.output_dir) / cfg.dataset
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def run(self):
        cfg = self.cfg
        print(f'\nStarting training for {cfg.epochs} epochs …\n')

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            train_stats = train_epoch(
                self.model, self.train_dl,
                self.optimizer, self.criterion,
                self.window_gen, self.matcher,
                self.device, self.scaler,
            )
            eval_stats = evaluate(
                self.model, self.test_dl,
                self.device, self.n_classes,
            )
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f'Epoch {epoch:03d}/{cfg.epochs} '
                f'| loss {train_stats["total_loss"]:.4f} '
                f'(conf {train_stats["conf_loss"]:.4f} '
                f'loc {train_stats["loc_loss"]:.4f}) '
                f'| acc {eval_stats["accuracy"]:.4f} '
                f'| F1 {eval_stats["f1"]:.4f} '
                f'| {elapsed:.1f}s'
            )

            if eval_stats['f1'] > self.best_f1:
                self.best_f1 = eval_stats['f1']
                ckpt = self.save_dir / 'best_model.pt'
                torch.save({
                    'epoch':     epoch,
                    'state_dict': self.model.state_dict(),
                    'f1':        self.best_f1,
                    'cfg':       vars(cfg),
                }, ckpt)
                print(f'  ✓ New best F1 {self.best_f1:.4f} saved → {ckpt}')

        print(f'\nTraining complete. Best F1: {self.best_f1:.4f}')
        return self.best_f1


# ---------------------------------------------------------------------------
# Ablation Study Runner  (Section IV-F)
# ---------------------------------------------------------------------------

def run_ablation_study(base_cfg: argparse.Namespace) -> None:
    """
    Systematically sweep α/β weights and scale s values
    as in Tables VII and VIII of the paper.
    """
    print('\n' + '='*60)
    print('ABLATION: α / β weight combinations (Table VII)')
    print('='*60)

    weight_combos = [
        (1, 1), (1, 2), (1, 3), (2, 1), (2, 3)
    ]
    for alpha, beta in weight_combos:
        cfg = argparse.Namespace(**vars(base_cfg))
        cfg.alpha  = float(alpha)
        cfg.beta   = float(beta)
        cfg.scales = [2.0, 3.0]
        cfg.epochs = min(cfg.epochs, 30)   # quick ablation
        print(f'\nα={alpha}, β={beta}')
        t = Trainer(cfg)
        f1 = t.run()
        print(f'  → F1 = {f1:.4f}')

    print('\n' + '='*60)
    print('ABLATION: scale s combinations (Table VIII)')
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
        cfg.beta   = 1.0
        cfg.scales = scales
        cfg.epochs = min(cfg.epochs, 30)
        print(f'\ns = {scales}')
        t = Trainer(cfg)
        f1 = t.run()
        print(f'  → F1 = {f1:.4f}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train MTHARS')

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

    # Loss (ablation)
    p.add_argument('--alpha',   type=float, default=1.0)
    p.add_argument('--beta',    type=float, default=1.0)
    p.add_argument('--n_neg_ratio', type=int, default=3)

    # IOU thresholds
    p.add_argument('--pos_iou_thresh', type=float, default=0.5)
    p.add_argument('--neg_iou_thresh', type=float, default=0.3)

    # Training
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--amp',          action='store_true')
    p.add_argument('--seed',         type=int,   default=42)

    # Ablation
    p.add_argument('--ablation', action='store_true',
                   help='Run full ablation study (Sec IV-F)')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.ablation:
        run_ablation_study(args)
    else:
        trainer = Trainer(args)
        trainer.run()
