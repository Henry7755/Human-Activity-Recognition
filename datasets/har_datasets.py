"""
dataset/har_datasets.py
=======================
Dataset loaders for all eight benchmark HAR datasets used in MTHARS.

Datasets (Table II of the paper):
    1. SKODA       – gesture, 1 subject,  96 Hz, 10 classes
    2. HCI         – gesture, 1 subject,  96 Hz,  8 classes
    3. PS          – ADL,     4 subjects, 50 Hz,  6 classes
    4. WISDM       – ADL,    29 subjects, 20 Hz,  6 classes
    5. UCI         – ADL,    30 subjects, 50 Hz,  6 classes  [UPDATED: Uses raw signals]
    6. OPPORTUNITY – ADL,     4 subjects, 30 Hz, 18 classes
    7. PAMAP2      – ADL,     9 subjects, 33 Hz, 18 classes
    8. UNIMIB SHAR – ADL,    30 subjects, 30 Hz, 17 classes

Each loader returns:
    X  : np.ndarray  (N, C, T)      – normalised sensor windows
    y  : np.ndarray  (N,)           – integer activity labels
    seg: list[dict]                  – ground-truth activity segments
         each dict: {'start': int, 'end': int, 'label': int}

Usage:
    from datasets.har_datasets import load_dataset
    X, y, segments = load_dataset('UCI', data_root='/data/UCI')
"""

import os
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import math
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_INFO: Dict[str, Dict[str, Any]] = {
    'SKODA':       {'freq': 96,   'window': 96,   'n_classes': 10, 'type': 'gesture'},
    'HCI':         {'freq': 96,   'window': 96,   'n_classes':  8, 'type': 'gesture'},
    'PS':          {'freq': 50,   'window': 100,  'n_classes':  6, 'type': 'ADL'},
    'WISDM':       {'freq': 20,   'window': 200,  'n_classes':  6, 'type': 'ADL'},
    'UCI':         {'freq': 50,   'window': 128,  'n_classes':  6, 'type': 'ADL'},  # UPDATED: 128 for raw signals
    'OPPORTUNITY': {'freq': 30,   'window': 30,   'n_classes': 18, 'type': 'ADL'},
    'PAMAP2':      {'freq': 33,   'window': 33,   'n_classes': 18, 'type': 'ADL'},
    'UNIMIB_SHAR': {'freq': 50,   'window': 151,  'n_classes': 17, 'type': 'ADL'},
}


def sliding_window(data: np.ndarray, labels: np.ndarray,
                   window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a continuous sensor stream using a sliding window.

    Args:
        data   : (T, C) array of sensor readings.
        labels : (T,)   array of per-frame activity labels.
        window : window length in samples.
        stride : hop between consecutive windows.

    Returns:
        X : (N, C, window)  windowed sensor data
        y : (N,)            majority-vote label per window
    """
    X_list, y_list = [], []
    for start in range(0, len(data) - window + 1, stride):
        end = start + window
        segment = data[start:end]          # (window, C)
        seg_label = labels[start:end]

        X_list.append(segment.T)          # → (C, window)
        # majority vote for label
        vals, counts = np.unique(seg_label, return_counts=True)
        y_list.append(int(vals[counts.argmax()]))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def build_segments(labels: np.ndarray) -> List[Dict[str, int]]:
    """
    Convert a per-frame label array to a list of contiguous activity segments.

    Returns:
        list of {'start': int, 'end': int, 'label': int}
    """
    segs, i = [], 0
    while i < len(labels):
        lbl = labels[i]
        j = i
        while j < len(labels) and labels[j] == lbl:
            j += 1
        segs.append({'start': i, 'end': j - 1, 'label': int(lbl)})
        i = j
    return segs


def normalise(X: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation across all windows."""
    mean = X.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std  = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def load_UCI(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    UCI HAR Dataset - LOAD RAW SIGNALS (NOT pre-computed features).
    
    Expected structure:
        data_root/
            train/
                Inertial_Signals/body_acc_x_train.txt
                Inertial_Signals/body_acc_y_train.txt
                Inertial_Signals/body_acc_z_train.txt
                Inertial_Signals/body_gyro_x_train.txt
                Inertial_Signals/body_gyro_y_train.txt
                Inertial_Signals/body_gyro_z_train.txt
                Inertial_Signals/total_acc_x_train.txt
                Inertial_Signals/total_acc_y_train.txt
                Inertial_Signals/total_acc_z_train.txt
                y_train.txt
            test/
                Inertial_Signals/[same signal files with _test]
                y_test.txt
    
    Signals: 9 channels (3 acc + 3 gyro + 3 total_acc) × 128 samples @ 50 Hz
    """
    root = Path(data_root)
    
    # Signal file names (9 sensors)
    signal_names = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z',
    ]
    
    splits = []
    for split in ('train', 'test'):
        # Load activity labels
        y_path = root / split / f'y_{split}.txt'
        y = np.loadtxt(y_path, dtype=int) - 1  # 0-indexed
        
        # Load all 9 inertial signals
        signals_list = []
        for sig_name in signal_names:
            sig_path = root / split / 'Inertial Signals' / f'{sig_name}_{split}.txt'
            sig = np.loadtxt(sig_path, dtype=np.float32)  # (N, 128)
            signals_list.append(sig)
        
        # Stack: (N, 128) × 9 → (N, 9, 128)
        X = np.stack(signals_list, axis=1)  # (N, 9, 128)
        
        splits.append((X, y))
    
    # Concatenate train + test
    X = np.vstack([s[0] for s in splits]).astype(np.float32)  # (N_total, 9, 128)
    y = np.hstack([s[1] for s in splits]).astype(np.int64)
    
    segs = build_segments(y)
    return normalise(X), y, segs


def load_WISDM(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    WISDM v1.1 Activity Recognition Dataset.
    Expected: data_root/WISDM_ar_v1.1_raw.txt
    """
    fp = Path(data_root) / 'WISDM_ar_v1.1_raw.txt'
    rows = []
    activity_map = {'Walking': 0, 'Jogging': 1, 'Upstairs': 2,
                    'Downstairs': 3, 'Sitting': 4, 'Standing': 5}

    with open(fp) as f:
        for line in f:
            line = line.strip().rstrip(';')
            parts = line.split(',')
            if len(parts) != 6:
                continue
            try:
                label = activity_map.get(parts[1].strip(), -1)
                if label == -1:
                    continue
                x, y_ax, z = float(parts[3]), float(parts[4]), float(parts[5])
                rows.append([x, y_ax, z, label])
            except ValueError:
                continue

    arr = np.array(rows, dtype=np.float32)
    data, labels = arr[:, :3], arr[:, 3].astype(np.int64)

    # Fill NaN
    data = np.nan_to_num(data)

    window = DATASET_INFO['WISDM']['window']
    stride = window // 2
    X, y = sliding_window(data, labels, window, stride)
    segs  = build_segments(labels)
    return normalise(X), y, segs


def load_PAMAP2(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    PAMAP2 Physical Activity Monitoring Dataset.
    Expected: data_root/Protocol/subject10{1..9}.dat
    Activity IDs 1-24 (12 mandatory + optional); nulls (id=0) dropped.
    """
    root    = Path(data_root) / 'Protocol'
    keep_cols = list(range(1, 54))    # timestamp + 3 IMU × 3 sensors × 6 axes
    label_col = 1                     # column index in raw file

    all_data, all_labels = [], []

    for subj in range(1, 10):
        fp = root / f'subject10{subj}.dat'
        if not fp.exists():
            warnings.warn(f'PAMAP2: {fp} not found, skipping.')
            continue
        df = pd.read_csv(fp, sep=' ', header=None)
        df.replace(0, np.nan, inplace=True)
        df.interpolate(inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        lbl = df.iloc[:, 1].values.astype(int)
        # Drop null (0) and optional activities (> 12 → keep 1-12)
        valid = (lbl > 0) & (lbl <= 12)
        lbl   = lbl[valid] - 1      # 0-indexed
        data  = df.iloc[valid, 2:].values.astype(np.float32)

        all_data.append(data)
        all_labels.append(lbl)

    data   = np.vstack(all_data)
    labels = np.concatenate(all_labels).astype(np.int64)

    window = DATASET_INFO['PAMAP2']['window'] * 10   # 33 Hz × ~10 = ~330 samples
    stride = window // 2
    X, y  = sliding_window(data, labels, window, stride)
    segs  = build_segments(labels)
    return normalise(X), y, segs


def load_OPPORTUNITY(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    OPPORTUNITY Activity Recognition Dataset.
    Expected: data_root/dataset/S{1..4}-ADL{1..5}.dat
    Uses column 244 (Locomotion label).
    """
    root = Path(data_root) / 'dataset'
    label_col = 243    # 0-indexed, Locomotion
    sensor_cols = list(range(1, 37)) + list(range(38, 46))  # subset

    loco_map = {0: -1, 1: 0, 2: 1, 4: 2, 5: 3}  # map to 0..3

    all_data, all_labels = [], []

    for subj in range(1, 5):
        for run in range(1, 6):
            fp = root / f'S{subj}-ADL{run}.dat'
            if not fp.exists():
                continue
            df = pd.read_csv(fp, sep=' ', header=None)
            df.interpolate(inplace=True)
            df.fillna(0, inplace=True)

            lbl  = df.iloc[:, label_col].map(loco_map).fillna(-1).values.astype(int)
            data = df.iloc[:, sensor_cols].values.astype(np.float32)

            valid   = lbl >= 0
            all_data.append(data[valid])
            all_labels.append(lbl[valid])

    data   = np.vstack(all_data)
    labels = np.concatenate(all_labels).astype(np.int64)

    window = DATASET_INFO['OPPORTUNITY']['window'] * 30   # 1 s
    stride = window // 2
    X, y  = sliding_window(data, labels, window, stride)
    segs  = build_segments(labels)
    return normalise(X), y, segs


def load_SKODA(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    SKODA Mini Checkpoint Dataset.
    Expected: data_root/right_classall_clean.mat  (or .txt)
    """
    root = Path(data_root)
    fp_mat = root / 'right_classall_clean.mat'
    fp_txt = root / 'right_classall_clean.txt'

    if fp_mat.exists():
        mat  = loadmat(str(fp_mat))
        data = mat['right_classall_clean'].astype(np.float32)
        lbl  = data[:, 0].astype(np.int64)
        data = data[:, 1:]
    elif fp_txt.exists():
        arr  = np.loadtxt(fp_txt)
        lbl  = arr[:, 0].astype(np.int64)
        data = arr[:, 1:].astype(np.float32)
    else:
        raise FileNotFoundError(f'SKODA raw file not found in {root}')

    le   = LabelEncoder()
    lbl  = le.fit_transform(lbl).astype(np.int64)

    window = DATASET_INFO['SKODA']['window'] * 1   # 1 s × 96 Hz
    stride = window // 2
    X, y  = sliding_window(data, lbl, window, stride)
    segs  = build_segments(lbl)
    return normalise(X), y, segs


def load_HCI(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    HCI Gesture Dataset.
    Expected layout mirrors SKODA (single .dat or .mat).
    """
    root = Path(data_root)
    candidates = list(root.glob('*.dat')) + list(root.glob('*.mat'))
    if not candidates:
        raise FileNotFoundError(f'No HCI data files found in {root}')

    fp = candidates[0]
    if fp.suffix == '.mat':
        mat  = loadmat(str(fp))
        key  = [k for k in mat if not k.startswith('_')][0]
        arr  = mat[key].astype(np.float32)
    else:
        arr  = np.loadtxt(fp).astype(np.float32)

    lbl  = arr[:, 0].astype(np.int64)
    data = arr[:, 1:]
    le   = LabelEncoder()
    lbl  = le.fit_transform(lbl).astype(np.int64)

    window = DATASET_INFO['HCI']['window']
    stride = window // 2
    X, y  = sliding_window(data, lbl, window, stride)
    segs  = build_segments(lbl)
    return normalise(X), y, segs


def load_PS(data_root: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Phone Sensors (PS) Dataset.
    Expected: data_root/ contains per-user CSV files with columns:
        timestamp, activity, ax, ay, az, gx, gy, gz, mx, my, mz
    """
    root  = Path(data_root)
    files = sorted(root.glob('*.csv'))
    activity_map = {'walking': 0, 'standing': 1, 'sitting': 2,
                    'upstairs': 3, 'downstairs': 4, 'jogging': 5}

    all_data, all_labels = [], []
    for fp in files:
        df  = pd.read_csv(fp)
        lbl = df['activity'].str.lower().map(activity_map).values
        dat = df.iloc[:, 2:].values.astype(np.float32)
        valid = ~np.isnan(lbl)
        all_data.append(dat[valid])
        all_labels.append(lbl[valid].astype(np.int64))

    data   = np.vstack(all_data)
    labels = np.concatenate(all_labels)

    window = DATASET_INFO['PS']['window']
    stride = window // 2
    X, y  = sliding_window(data, labels, window, stride)
    segs  = build_segments(labels)
    return normalise(X), y, segs


def load_UNIMIB_SHAR(data_root: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    UniMiB SHAR Dataset - Updated to load .npy files.
    
    Expected structure:
        data_root/
            full_data.npy    -> shape (N, T) or (N, 1, T)
            full_labels.npy  -> shape (N,) or (N, 1)
    """
    root = Path(data_root)
    
    # 1. Define paths for the .npy files
    data_path = root / 'full_data.npy'
    labels_path = root / 'full_labels.npy'

    # 2. Load using np.load instead of loadmat
    if data_path.exists() and labels_path.exists():
        data = np.load(data_path).astype(np.float32)
        labels = np.load(labels_path)
        
        # Ensure labels are a flat, 1D array of integers
        labels = labels.ravel().astype(np.int64)
        
        # Check if labels are 1-indexed (like the original .mat files) 
        # and normalize to 0-indexed if necessary
        if labels.min() > 0:
            labels = labels - 1
    else:
        # Fallback: keep your original sub-folder CSV parsing logic just in case
        X_list, y_list = [], []
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(f'.npy files or class directories not found in {root}')
            
        for idx, cdir in enumerate(class_dirs):
            for fp2 in cdir.glob('*.csv'):
                seg = np.loadtxt(fp2, delimiter=',').astype(np.float32)
                X_list.append(seg)
                y_list.append(idx)
        data   = np.stack(X_list)
        labels = np.array(y_list, dtype=np.int64)

    # 3. Shape verification: Ensure it matches the (N, C, T) expected format
    if data.ndim == 2:
        # If it's (N, T), add the channel axis -> (N, 1, T)
        data = data[:, np.newaxis, :]  
        
    segs = build_segments(labels)
    return normalise(data.astype(np.float32)), labels, segs


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_LOADERS = {
    'SKODA':       load_SKODA,
    'HCI':         load_HCI,
    'PS':          load_PS,
    'WISDM':       load_WISDM,
    'UCI':         load_UCI,
    'OPPORTUNITY': load_OPPORTUNITY,
    'PAMAP2':      load_PAMAP2,
    'UNIMIB_SHAR': load_UNIMIB_SHAR,
}


def load_dataset(name: str, data_root: str
                 ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load any of the 8 HAR benchmark datasets.

    Args:
        name      : one of 'SKODA','HCI','PS','WISDM','UCI',
                    'OPPORTUNITY','PAMAP2','UNIMIB_SHAR'
        data_root : path to the dataset's root directory.

    Returns:
        X    : (N, C, T)   float32 normalised windows
        y    : (N,)        int64  activity labels
        segs : list of {'start','end','label'} dicts
    """
    name = name.upper().replace('-', '_')
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from {list(_LOADERS.keys())}")
    return _LOADERS[name](data_root)


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------
class HARDataset(Dataset):
    """
    Wraps (X, y) arrays into a torch Dataset.

    Args:
        X        : (N, C, T) float32
        y        : (N,)      int64
        segments : optional list of segment dicts (for segmentation tasks)
        augment  : if True, apply simple Gaussian-noise augmentation
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 segments: List[Dict] = None, augment: bool = False):
        self.X        = torch.from_numpy(X)
        self.y        = torch.from_numpy(y)
        self.segments = segments or []
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]          # (C, T)
        if self.augment:
            x = x + 0.01 * torch.randn_like(x)
        return x, self.y[idx]

def get_dataloaders(X: np.ndarray, y: np.ndarray,
                    segments: List[Dict],
                    train_ratio: float = 0.70,
                    batch_size: int = 64,
                    augment: bool = True,
                    num_workers: int = 4,
                    seed: int = 42
                    ) -> Tuple[DataLoader, DataLoader]:
    """
    Split into train/test and return DataLoaders.

    Train/test ratio follows the paper: 70/30 for most datasets,
    80/20 for PAMAP2 (pass train_ratio=0.8 for that).
    """
    rng = np.random.default_rng(seed)
    N   = len(y)
    idx = rng.permutation(N)
    n_train = int(N * train_ratio)

    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_ds = HARDataset(X[train_idx], y[train_idx],
                          augment=augment)
    test_ds  = HARDataset(X[test_idx],  y[test_idx],
                          augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers,
                          pin_memory=pin_memory)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory)
    return train_dl, test_dl


# ---------------------------------------------------------------------------
# CLI demo
# --------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'UCI'
    root = sys.argv[2] if len(sys.argv) > 2 else f'/data/{name}'
    try:
        X, y, segs = load_dataset(name, root)
        print(f'[{name}]  X:{X.shape}  y:{y.shape}  n_segments:{len(segs)}')
        print(f'  classes : {np.unique(y).tolist()}')
        print(f'  first 3 segments : {segs[:3]}')
    except FileNotFoundError as e:
        print(f'ERROR: {e}')
        print('Please download the dataset and specify the correct data_root.')
