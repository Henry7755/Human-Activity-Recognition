"""
utils/baselines_and_inference.py
================================
Utility module covering:

1. RNNBaseline   – LSTM / GRU classifiers used in Tables III & IV.
2. train_rnn_baseline – full train/eval loop for the RNN baselines.
3. plot_confusion_matrix – reproduces Fig. 6 (PAMAP2 confusion matrices).
4. plot_ned_bar   – reproduces Fig. 5 (NED bar chart).
5. inference_demo – end-to-end MTHARS prediction on a raw sensor stream.
6. SKMSWBaseline  – the SKMSW (SK + multi-scale window) recognition-only
                    ablation referenced in Fig. 5 legend.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, confusion_matrix


# ---------------------------------------------------------------------------
# 1. RNN Baselines  (LSTM / GRU)
# ---------------------------------------------------------------------------

class RNNBaseline(nn.Module):
    """
    Simple stacked LSTM or GRU for HAR classification.

    Used as the LSTM / GRU entries in Table III (static window experiments).

    Input  : (B, T, C)   – time-first sensor window
    Output : (B, K)      – class logits
    """

    def __init__(self,
                 input_size:  int,
                 hidden_size: int,
                 n_classes:   int,
                 n_layers:    int = 2,
                 rnn_type:    str = 'LSTM',
                 dropout:     float = 0.3):
        super().__init__()

        rnn_cls = nn.LSTM if rnn_type.upper() == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)
        self.rnn_type   = rnn_type.upper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C) sensor input

        Returns:
            logits : (B, K)
        """
        out, _ = self.rnn(x)             # (B, T, H)
        last    = out[:, -1, :]           # take last time-step
        last    = self.dropout(last)
        return self.classifier(last)      # (B, K)


def train_rnn_baseline(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        rnn_type:    str   = 'LSTM',
        hidden_size: int   = 128,
        n_layers:    int   = 2,
        epochs:      int   = 50,
        batch_size:  int   = 64,
        lr:          float = 1e-3,
        device:      Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Train an LSTM or GRU classifier and return weighted F1 on the test set.

    Args:
        X_train / X_test : (N, C, T) float32
        y_train / y_test : (N,)      int64

    Returns:
        dict with keys 'f1', 'accuracy'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reshape (N, C, T) → (N, T, C)  for RNN (time-first)
    Xt  = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    yt  = torch.from_numpy(y_train).long()
    Xv  = torch.from_numpy(X_test.transpose(0, 2, 1)).float()
    yv  = torch.from_numpy(y_test).long()

    train_dl = DataLoader(TensorDataset(Xt, yt),
                          batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(TensorDataset(Xv, yv),
                          batch_size=batch_size, shuffle=False)

    n_classes  = int(y_train.max()) + 1
    input_size = X_train.shape[1]   # C

    model = RNNBaseline(input_size=input_size,
                        hidden_size=hidden_size,
                        n_classes=n_classes,
                        n_layers=n_layers,
                        rnn_type=rnn_type).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().tolist())

    p = np.array(all_preds)
    l = np.array(all_labels)
    return {
        'f1':       float(f1_score(l, p, average='weighted', zero_division=0)),
        'accuracy': float((p == l).mean()),
    }


# ---------------------------------------------------------------------------
# 2. SKMSW Baseline  (SK + multi-scale window, recognition only)
# ---------------------------------------------------------------------------

class SKMSWBaseline:
    """
    Placeholder for the SKMSW method shown in the NED bar chart (Fig. 5).

    Full SKMSW = the SK backbone with multiscale windows but WITHOUT the
    joint segmentation training (i.e., recognition-only mode).
    We approximate it by evaluating MTHARS with the segmentation loss
    weight β = 0 (effectively disabling the localization branch).

    To run:  set beta=0 in the trainer config.
    """

    @staticmethod
    def note():
        print(
            '[SKMSW] Run MTHARS trainer with --beta 0 to get the '
            'recognition-only baseline shown in Fig. 5 / Table V.'
        )


# ---------------------------------------------------------------------------
# 3. Confusion Matrix Plotter  (Fig. 6)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
        y_true:      np.ndarray,
        y_pred:      np.ndarray,
        class_names: List[str],
        title:       str = 'Confusion Matrix',
        save_path:   Optional[str] = None,
        figsize:     Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a normalised confusion matrix matching Fig. 6 of the paper.

    Args:
        y_true      : (N,) true labels
        y_pred      : (N,) predicted labels
        class_names : list of activity name strings
        title       : figure title
        save_path   : if given, save figure to this path (.png / .pdf)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('matplotlib not installed; skipping plot.')
        return

    cm = confusion_matrix(y_true, y_pred, normalize=None)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # Annotate cells with raw counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=7, color=color)

    # Compute accuracy for title
    acc = cm.diagonal().sum() / cm.sum()
    mis = 1.0 - acc
    ax.set_title(f'{title}\naccuracy={acc:.3f}; misclass={mis:.3f}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Confusion matrix saved → {save_path}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. NED Bar Chart  (Fig. 5)
# ---------------------------------------------------------------------------

def plot_ned_bar(
        ned_results: Dict[str, Dict[str, float]],
        datasets:    List[str],
        save_path:   Optional[str] = None,
) -> None:
    """
    Bar chart of NED values matching Fig. 5 of the paper.

    Args:
        ned_results : {method_name: {dataset_name: ned_value}}
        datasets    : ordered list of dataset names (x-axis)
        save_path   : optional output path
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        warnings.warn('matplotlib not installed; skipping plot.')
        return

    methods = list(ned_results.keys())
    n_d     = len(datasets)
    n_m     = len(methods)
    width   = 0.8 / n_m
    x       = np.arange(n_d)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4878CF', '#D65F5F', '#6ACC65', '#B47CC7']

    for i, method in enumerate(methods):
        vals = [ned_results[method].get(d, float('nan')) for d in datasets]
        ax.bar(x + i * width, vals, width=width,
               label=method, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (n_m - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('NED (lower is better)')
    ax.set_title('Segmentation NED – Fig. 5 of MTHARS paper')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'NED bar chart saved → {save_path}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. End-to-End Inference Demo
# ---------------------------------------------------------------------------

def inference_demo(
        model_path:  str,
        raw_signal:  np.ndarray,
        in_channels: int,
        n_classes:   int,
        scales:      List[float],
        feat_dim:    int,
        data_len:    int,
        activity_names: Optional[List[str]] = None,
        device:      Optional[torch.device] = None,
) -> List[Dict]:
    """
    Run MTHARS inference on a continuous raw sensor stream.

    The stream is split into fixed-length chunks of `data_len` samples,
    each chunk is passed through the model, and the resulting segments
    are concatenated using the Concatenation Algorithm (Algorithm 1).

    Args:
        model_path   : path to a saved MTHARS checkpoint (.pt)
        raw_signal   : (T_total, C) raw sensor data (un-normalised)
        in_channels  : C (must match the saved model)
        n_classes    : K
        scales       : window scale list
        feat_dim     : backbone feature dimension
        data_len     : fixed input length the model expects
        activity_names : optional list mapping class index → name
        device       : inference device

    Returns:
        all_segments : list of merged segment dicts
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.recognition_segmentation import MTHARS, ConcatenateAlgorithm

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MTHARS(in_channels=in_channels, n_classes=n_classes,
                   scales=scales, feat_dim=feat_dim, data_len=data_len)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f'Loaded checkpoint (epoch {ckpt.get("epoch","?")}, '
          f'F1={ckpt.get("f1", "?"):.4f})')

    # Normalise
    signal = raw_signal.astype(np.float32)
    mean   = signal.mean(axis=0, keepdims=True)
    std    = signal.std(axis=0, keepdims=True) + 1e-8
    signal = (signal - mean) / std

    # Chunk and predict
    T_total = signal.shape[0]
    concat  = ConcatenateAlgorithm()
    offset  = 0
    all_segments: List[Dict] = []

    while offset < T_total:
        chunk = signal[offset: offset + data_len]   # (<=data_len, C)
        if len(chunk) < data_len:
            # Zero-pad the last chunk
            pad   = np.zeros((data_len - len(chunk), signal.shape[1]),
                             dtype=np.float32)
            chunk = np.vstack([chunk, pad])

        # (C, T) → (1, C, T)
        x_t = torch.from_numpy(chunk.T).unsqueeze(0).to(device)

        with torch.no_grad():
            batch_segs = model.predict(x_t)   # [[seg, seg, …]]

        # Adjust absolute positions
        for seg in batch_segs[0]:
            seg['start'] += offset
            seg['end']   += offset

        merged = concat.merge(batch_segs[0])
        all_segments.extend(merged)
        offset += data_len

    # Final merge across chunks
    final = concat.merge(all_segments)

    print(f'\nDetected {len(final)} activity segment(s):')
    for s in final:
        name = (activity_names[s['class'] - 1]
                if activity_names and 0 < s['class'] <= len(activity_names)
                else str(s['class']))
        print(f'  [{s["start"]:6d} – {s["end"]:6d}]  '
              f'{name}  (score={s["score"]:.3f})')

    return final


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # RNN baseline quick test (random data)
    np.random.seed(0)
    N, C, T, K = 200, 6, 128, 4
    Xtr = np.random.randn(N, C, T).astype(np.float32)
    ytr = np.random.randint(0, K, N).astype(np.int64)
    Xte = np.random.randn(50, C, T).astype(np.float32)
    yte = np.random.randint(0, K, 50).astype(np.int64)

    res = train_rnn_baseline(Xtr, ytr, Xte, yte,
                             rnn_type='LSTM', epochs=5)
    print(f'LSTM baseline: {res}')

    res = train_rnn_baseline(Xtr, ytr, Xte, yte,
                             rnn_type='GRU',  epochs=5)
    print(f'GRU  baseline: {res}')

    # Confusion matrix (no actual plot in headless env)
    y_true = np.array([0,1,2,0,1,2,0])
    y_pred = np.array([0,1,1,0,2,2,0])
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    SKMSWBaseline.note()