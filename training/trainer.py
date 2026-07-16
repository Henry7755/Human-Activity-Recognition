"""
training/trainer.py  (SEGMENTATION-ONLY VARIANT)
==================================================
Training, evaluation, and empirical monitoring loop for the
segmentation-only model (MTHARSSegmentation).

CHANGED vs. the multi-task version
-----------------------------------
Implements the remaining §4.B steps against `trainer.py`:

  - `prepare_targets`: `WindowMatcher.match()` no longer takes `gt_labels`
    and returns 2 values (`matched_offsets`, `pos_mask`) instead of 3 — no
    `all_labels` buffer.
  - `train_epoch`: `model(batch_x)` now returns `(pred_offsets, obj_logits)`
    instead of `(cls_logits, offsets)`. `criterion(...)` is called with
    `(pred_offsets, obj_logits, true_offsets, pos_mask)`, matching
    `SegmentationLoss.forward`'s new signature. `total_stats` drops
    `conf_loss`, adds `obj_loss`.
  - `evaluate` — REWRITTEN, and flagged explicitly:

      The coupling-map analysis (§4.B step 6) says to "wire
      SegmentationEvaluator (NED) into trainer.py's eval loop." Taken
      literally, that doesn't transfer cleanly: NED (Eq. 9/10) is defined
      as an edit distance between two *class-label* sequences — but a
      segmentation-only model has no classification branch left to
      produce a label sequence from, only a foreground/background span.
      Comparing a "no class" prediction sequence against a multi-class
      ground truth via edit distance isn't well-defined without inventing
      semantics the paper doesn't specify.

      This file instead evaluates segmentation quality the way it's
      actually measurable given what the model outputs: frame-level
      foreground/background IoU (Jaccard index) between the predicted
      foreground mask (rendered from `MTHARSSegmentation.predict()` +
      `ConcatenateAlgorithm`) and the true foreground mask. ASSUMPTION,
      flagged: this treats raw label `0` in the dataset as a
      "null/no-activity" background class (the same convention already
      implicit in `prepare_targets`' `s['label'] + 1` 1-indexing scheme,
      and consistent with common HAR segmentation benchmarks like
      OPPORTUNITY/SKODA's "null" class). If your dataset doesn't reserve
      label 0 for background, adjust `_true_foreground_mask` below.

      If you have the project's real `evaluation/metrics.py`, wiring its
      actual `SegmentationEvaluator`/NED implementation in place of the
      frame-IoU evaluator below is straightforward — swap the import and
      the two calls inside `evaluate()`. I don't have that file's content,
      so this is a self-contained substitute rather than a guess at its
      internals.

  - `Trainer.__init__`: instantiates `MTHARSSegmentation` (not `MTHARS`,
    no `n_classes` arg) and `SegmentationLoss` (not `MTHARSLoss`); drops
    `cfg.alpha`, adds `cfg.gamma`/`cfg.pos_weight`.
  - CLI (`parse_args`): `--alpha` removed, `--gamma` and `--pos_weight`
    added. `--beta` kept (still weights the localisation term).
  - `run_ablation_study`: the α/β sweep (Table VII) is replaced with a
    β/γ/pos_weight-aware analogue, since α/L_conf no longer exists.
  - `generate_report_image`: plots `loc_loss`/`obj_loss` instead of
    `conf_loss`/`loc_loss`; the accuracy/F1 panel is replaced with mean
    foreground IoU.

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
from model.segmentation import MTHARSSegmentation, ConcatenateAlgorithm
from model.multiscale_windows import WindowGenerator, WindowMatcher
from training.losses import SegmentationLoss


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
# Ground-truth preparation  (window → matched offsets + positive mask)
# ---------------------------------------------------------------------------

def prepare_targets(window_gen: WindowGenerator,
                    matcher:    WindowMatcher,
                    gt_segments: List[List[Dict]],
                    device:      torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of GT segment lists into per-window offsets and
    positive masks. CHANGED: no `gt_labels` passed to `matcher.match()`
    anymore, and no `all_labels` buffer returned — segmentation-only
    doesn't classify.
    """
    B  = len(gt_segments)
    na = window_gen.num_windows

    all_offsets = torch.zeros(B, na, 2, dtype=torch.float32)
    all_pos     = torch.zeros(B, na, dtype=torch.bool)

    anchors = window_gen.windows.to(device)   # (na, 2) [center, length]

    for b, segs in enumerate(gt_segments):
        if not segs:
            continue

        gt_boxes = []
        for s in segs:
            cx = (s['start'] + s['end']) / 2.0
            ln = float(s['end'] - s['start'] + 1)
            gt_boxes.append([cx, ln])
        # NOTE: no gt_lbl list built here — segmentation-only never needs
        # a class id, only the box itself.

        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)

        off, pos = matcher.match(anchors, gt_boxes_t)
        all_offsets[b] = off.cpu()
        all_pos[b]     = pos.cpu()

    return all_offsets.to(device), all_pos.to(device)


# ---------------------------------------------------------------------------
# GT segment extraction  (shared helper — same transition-tracing logic as
# the multi-task/recognition-only trainer, but 'label' is only used
# internally to find segment boundaries, never propagated to the matcher)
# ---------------------------------------------------------------------------

def _extract_gt_segments(batch_y: torch.Tensor, T: int) -> List[List[Dict]]:
    """
    Args:
        batch_y : (B, T) frame-level labels, or (B,) whole-clip labels
        T       : window length

    Returns:
        list of per-sample GT segment lists: [{'start', 'end', 'label'}]
        ('label' is kept in the dict for foreground-mask evaluation later,
        but `prepare_targets` above ignores it when building matcher input)
    """
    B = batch_y.shape[0]
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

    return gt_segs


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model:       MTHARSSegmentation,
                loader:      DataLoader,
                optimizer:   optim.Optimizer,
                criterion:   SegmentationLoss,
                window_gen:  WindowGenerator,
                matcher:     WindowMatcher,
                device:      torch.device,
                scaler:      Optional[GradScaler] = None,
                max_norm:    float = 1.0
                ) -> Dict[str, float]:
    """Run one training epoch. Supports both single-GPU and multi-GPU."""
    model.train()
    total_stats: Dict[str, float] = {
        'loc_loss': 0.0, 'obj_loss': 0.0, 'total_loss': 0.0, 'n_pos': 0.0
    }
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)   # (B, C, T)
        batch_y = batch_y.to(device, non_blocking=True)   # (B, T) or (B,)
        T       = batch_x.shape[2]

        gt_segs = _extract_gt_segments(batch_y, T)

        true_offsets, pos_mask = prepare_targets(
            window_gen, matcher, gt_segs, device
        )

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                pred_offsets, obj_logits = model(batch_x)
                loss, stats = criterion(pred_offsets, obj_logits,
                                        true_offsets, pos_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_offsets, obj_logits = model(batch_x)
            loss, stats = criterion(pred_offsets, obj_logits,
                                    true_offsets, pos_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        for k in total_stats:
            total_stats[k] += stats.get(k, 0.0)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_stats.items()}


# ---------------------------------------------------------------------------
# Frame-level foreground IoU  (substitute for the class-based NED metric —
# see the module docstring for why NED itself doesn't transfer here)
# ---------------------------------------------------------------------------

def _true_foreground_mask(batch_y: torch.Tensor, T: int) -> torch.Tensor:
    """
    ASSUMPTION: raw label 0 means "no activity / null / background" in the
    dataset's own label space (same convention `_extract_gt_segments` and
    `prepare_targets` already lean on). Adjust here if your dataset uses a
    different background convention or has no null class at all.

    Args:
        batch_y : (B, T) or (B,)

    Returns:
        mask : (B, T) bool, True where a frame is foreground
    """
    if batch_y.dim() > 1:
        return batch_y != 0
    # Whole-clip label with no frame annotation: assume the entire clip
    # is foreground (mirrors the dual-scale synthetic GT box injection
    # used elsewhere for this label mode).
    B = batch_y.shape[0]
    return torch.ones(B, T, dtype=torch.bool, device=batch_y.device)


def _pred_foreground_mask(segments: List[Dict], T: int, device) -> torch.Tensor:
    """Render a single sample's merged segments into a (T,) bool mask."""
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    for seg in segments:
        s = max(0, seg['start'])
        e = min(T - 1, seg['end'])
        if e >= s:
            mask[s:e + 1] = True
    return mask


class SegmentationEvaluator:
    """
    Accumulates frame-level foreground IoU across batches.

    This plays the role `evaluation/metrics.py::SegmentationEvaluator`
    (NED-based) played in the multi-task pipeline, adapted for a model
    that has no class output — see module docstring for the reasoning.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0
        self.union = 0
        self.n_samples = 0

    def update(self, pred_mask: torch.Tensor, true_mask: torch.Tensor):
        """
        Args:
            pred_mask, true_mask : (T,) bool tensors for one sample
        """
        inter = (pred_mask & true_mask).sum().item()
        uni   = (pred_mask | true_mask).sum().item()
        self.intersection += inter
        self.union += uni
        self.n_samples += 1

    def compute(self) -> float:
        if self.union == 0:
            return 1.0 if self.intersection == 0 else 0.0
        return self.intersection / self.union


# ---------------------------------------------------------------------------
# Evaluation epoch  (REWRITTEN: frame-IoU instead of accuracy/F1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model:      MTHARSSegmentation,
             loader:     DataLoader,
             device:     torch.device,
             concat:     Optional[ConcatenateAlgorithm] = None,
             ) -> Dict[str, float]:
    """
    Evaluate segmentation quality via frame-level foreground IoU.

    CHANGED: this used to report classification accuracy/F1 by discarding
    the offset output (`cls_logits, _ = model(batch_x)`). There is no
    classification output left to report on. Instead this runs the full
    `predict()` → `ConcatenateAlgorithm` pipeline (the same inference path
    a deployment would use) and scores the resulting foreground spans
    against ground truth.
    """
    model.eval()
    if concat is None:
        concat = ConcatenateAlgorithm()

    evaluator = SegmentationEvaluator()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        T = batch_x.shape[2]

        raw_segments = model.predict(batch_x)          # list of B lists
        merged = concat(raw_segments)                  # list of B lists

        true_masks = _true_foreground_mask(batch_y, T)  # (B, T)

        for b in range(batch_x.shape[0]):
            pred_mask = _pred_foreground_mask(merged[b], T, device)
            evaluator.update(pred_mask, true_masks[b])

    return {
        'foreground_iou': evaluator.compute(),
    }

# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates data pipeline, segmentation-only backbone, telemetry, and
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
        # NOTE: n_classes is no longer stored/used for model construction —
        # kept here only if you want it for logging/reporting.
        self.window_t = info['window']
        self.freq     = info['freq']

        print(f'Device     : {self.device}')
        print(f'GPUs       : {self.num_gpus}')
        print(f'Dataset    : {cfg.dataset}  (segmentation-only)')
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

        # --- Model (segmentation-only, no n_classes) ---
        self.model = MTHARSSegmentation(
            in_channels=self.in_channels,
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
        )
        self.concat = ConcatenateAlgorithm(merge_gap_thresh=cfg.merge_gap_thresh)

        # CHANGED: SegmentationLoss instead of MTHARSLoss — no `alpha`,
        # adds `gamma`/`pos_weight` for the new objectness term.
        self.criterion = SegmentationLoss(
            beta=cfg.beta,
            gamma=cfg.gamma,
            pos_weight=cfg.pos_weight,
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
        self.best_iou = 0.0

    def run(self) -> float:
        cfg = self.cfg
        print(f'\nInitialization Verification. Outputs Routing to: {self.exp_dir}\n')

        history = {"train_loss": [], "loc_loss": [], "obj_loss": [], "val_iou": []}

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
                self.device, self.concat,
            )
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Engine/Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Losses/Total_Loss', train_stats["total_loss"], epoch)
            self.writer.add_scalar('Losses/Localization_Component', train_stats["loc_loss"], epoch)
            self.writer.add_scalar('Losses/Objectness_Component', train_stats["obj_loss"], epoch)
            self.writer.add_scalar('Evaluation/Foreground_IoU', eval_stats["foreground_iou"], epoch)

            history["train_loss"].append(train_stats["total_loss"])
            history["loc_loss"].append(train_stats["loc_loss"])
            history["obj_loss"].append(train_stats["obj_loss"])
            history["val_iou"].append(eval_stats["foreground_iou"])

            elapsed = time.time() - t0
            print(
                f'Epoch {epoch:03d}/{cfg.epochs} '
                f'| loss {train_stats["total_loss"]:.4f} '
                f'(loc {train_stats["loc_loss"]:.4f} '
                f'obj {train_stats["obj_loss"]:.4f}) '
                f'| fg-IoU {eval_stats["foreground_iou"]:.4f} '
                f'| {elapsed:.1f}s'
            )

            if eval_stats['foreground_iou'] > self.best_iou:
                self.best_iou = eval_stats['foreground_iou']
                ckpt = self.exp_dir / 'best_model.pt'

                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

                torch.save({
                    'epoch':     epoch,
                    'state_dict': model_state,
                    'foreground_iou': self.best_iou,
                    'cfg':        vars(cfg),
                }, ckpt)
                print(f'  ✓ New best fg-IoU {self.best_iou:.4f} saved → {ckpt}')

        self.writer.close()
        self.generate_report_image(history)
        return self.best_iou

    def generate_report_image(self, history: Dict[str, List[float]]):
        """
        Compiles training performance metadata into a PNG graphic.
        CHANGED: loss-decomposition chart plots loc/obj instead of
        conf/loc; the metric panel plots foreground IoU instead of F1.
        """
        epochs_range = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(epochs_range, history["train_loss"], color='black', linewidth=2, label='Total Loss')
        ax1.plot(epochs_range, history["loc_loss"], color='darkorange', linestyle='--', label='Loc Loss (β)')
        ax1.plot(epochs_range, history["obj_loss"], color='teal', linestyle=':', label='Objectness Loss (γ)')
        ax1.set_xlabel('Training Epochs')
        ax1.set_ylabel('Loss Magnitudes')
        ax1.set_title('Segmentation-Only Objective Component Decomposition')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax3 = ax2.twinx()
        p1 = ax2.plot(epochs_range, history["train_loss"], color='tab:red', alpha=0.7, label='Train Loss')
        p2 = ax3.plot(epochs_range, history["val_iou"], color='tab:blue', linewidth=2, label='Foreground IoU')

        ax2.set_xlabel('Training Epochs')
        ax2.set_ylabel('Loss', color='tab:red')
        ax3.set_ylabel('Foreground IoU', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax3.tick_params(axis='y', labelcolor='tab:blue')

        plots = p1 + p2
        labels = [l.get_label() for l in plots]
        ax2.legend(plots, labels, loc='center right')
        ax2.set_title('Convergence Telemetry Profile')
        ax2.grid(True, alpha=0.3)

        gpu_info = f"GPU Mode ({self.num_gpus} GPUs)" if self.num_gpus > 0 else "CPU Mode"
        plt.suptitle(f"Execution Analysis Matrix (Segmentation-Only)\nDataset: {self.cfg.dataset} | Optimizer: {self.cfg.optimizer} | {gpu_info}", fontsize=12, fontweight='bold')
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
    CHANGED: the original α/β sweep (Table VII) doesn't apply — α/L_conf
    no longer exists. Replaced with a β/γ sweep (localization weight vs.
    objectness weight) plus the original scale sweep (Table VIII), which
    still matters for boundary regression quality.
    """
    print('\n' + '='*60)
    print('ABLATION: β / γ weight combinations (segmentation-only analogue of Table VII)')
    print('='*60)

    weight_combos = [
        (1, 1), (1, 2), (2, 1), (3, 1)
    ]
    for beta, gamma in weight_combos:
        cfg = argparse.Namespace(**vars(base_cfg))
        cfg.beta   = float(beta)
        cfg.gamma  = float(gamma)
        cfg.scales = [2.0, 3.0]
        cfg.epochs = min(cfg.epochs, 30)
        print(f'\nβ={beta}, γ={gamma}')
        t = Trainer(cfg)
        iou = t.run()
        print(f'  → fg-IoU = {iou:.4f}')

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
        cfg.beta   = 1.0
        cfg.gamma  = 1.0
        cfg.scales = scales
        cfg.epochs = min(cfg.epochs, 30)
        print(f'\ns = {scales}')
        t = Trainer(cfg)
        iou = t.run()
        print(f'  → fg-IoU = {iou:.4f}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train MTHARS (segmentation-only)')

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
    # CHANGED: --alpha removed (no classification term). --gamma and
    # --pos_weight added for the new objectness term.
    p.add_argument('--beta',       type=float, default=1.0)
    p.add_argument('--gamma',      type=float, default=1.0)
    p.add_argument('--pos_weight', type=float, default=3.0,
                   help='BCEWithLogitsLoss pos_weight for the objectness head')

    # IOU thresholds (window↔instance assignment)
    p.add_argument('--pos_iou_thresh', type=float, default=0.5)
    p.add_argument('--neg_iou_thresh', type=float, default=0.3)

    # NEW: ConcatenateAlgorithm gap threshold (replaces the "class changed"
    # stopping condition, which no longer applies with a single class)
    p.add_argument('--merge_gap_thresh', type=int, default=5,
                   help='Max sample gap between detections before closing a segment')

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
                   help='Run β/γ + scale ablation study (segmentation-only)')

    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    if args.ablation:
        run_ablation_study(args)
    else:
        trainer = Trainer(args)
        trainer.run()
