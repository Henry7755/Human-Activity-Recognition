# =============================================================================
#  MTHARS – Multi-Task HAR with Activity Segmentation & Recognition
#  Faithful TensorFlow implementation of Duan et al. (2023)
#  Backbone: SKNet  |  Heads: Conv1D cls + loc  |  Post-process: NMS + concat
#
#  UCI HAR Dataset layout expected:
#   <DATA_ROOT>/
#     train/
#       Inertial Signals/  (total_acc_x_train.txt … body_gyro_z_train.txt)
#       y_train.txt
#       subject_train.txt
#     test/
#       Inertial Signals/  (total_acc_x_test.txt  … body_gyro_z_test.txt)
#       y_test.txt
#       subject_test.txt
# =============================================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = '/path/to/UCI HAR Dataset'   # <-- change this
NUM_CLASSES = 6                             # UCI HAR has 6 activity classes
WINDOW_LEN  = 128                           # samples per pre-segmented window
OVERLAP     = 64                            # 50 % overlap
BATCH_SIZE  = 64
EPOCHS      = 150
LR          = 1e-3
WD          = 1e-4
ALPHA       = 1.0   # weight for classification loss  (eq. 8 in paper)
BETA        = 1.0   # weight for localisation loss    (eq. 8 in paper)

# Anchor scales  s ∈ {s1, s2, …, sm}  – we use s=2,3 (best on UCI per Table VIII)
SCALES      = [2.0, 3.0]

# IOU threshold for window-to-ground-truth matching
IOU_MATCH_THRESHOLD = 0.5
# NMS IOU threshold
NMS_IOU_THRESHOLD   = 0.4
# Hard-negative mining ratio  neg:pos = 3:1
NEG_POS_RATIO       = 3

SIGNAL_NAMES_TRAIN = [
    'total_acc_x_train','total_acc_y_train','total_acc_z_train',
    'body_acc_x_train', 'body_acc_y_train', 'body_acc_z_train',
    'body_gyro_x_train','body_gyro_y_train','body_gyro_z_train',
]
SIGNAL_NAMES_TEST = [s.replace('_train','_test') for s in SIGNAL_NAMES_TRAIN]

ACTIVITY_NAMES = {
    1:'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS',
    4:'SITTING', 5:'STANDING', 6:'LAYING'
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & CONTINUOUS STREAM RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_signals(folder, names):
    """Load inertial signal .txt files → (N_windows, 128, 9)."""
    signals = []
    for name in names:
        path = os.path.join(folder, f'{name}.txt')
        data = pd.read_csv(path, header=None, sep=r'\s+').values
        signals.append(data)
    # signals[i] shape: (N, 128)  → stack on axis-2 → (N, 128, 9)
    return np.stack(signals, axis=2).astype(np.float32)


def reconstruct_continuous_stream(X_windows, y_windows, subject_ids, overlap=64):
    """
    Stitch pre-segmented windows back into per-subject continuous streams.

    Returns
    -------
    dict  {subject_id: {'signal': (T,9), 'labels': (T,), 'boundaries': list}}

    Each boundary: {'activity': int, 'tx': float, 'tl': float,
                    'start': int, 'end': int}
    """
    result = {}
    for sub in np.unique(subject_ids):
        mask  = (subject_ids == sub)
        sub_X = X_windows[mask]          # (n_win, 128, 9)
        sub_y = y_windows[mask]          # (n_win,)
        n_win = sub_X.shape[0]
        total = n_win * overlap + overlap

        # Accumulate + count for overlap averaging
        stream = np.zeros((total, 9), dtype=np.float64)
        counts = np.zeros((total, 1), dtype=np.float64)
        for i in range(n_win):
            s, e = i * overlap, i * overlap + WINDOW_LEN
            stream[s:e] += sub_X[i]
            counts[s:e] += 1
        stream /= counts

        # Sample-level labels: window label covers its 64-sample stride
        sample_labels = np.zeros(total, dtype=np.int32)
        for i in range(n_win):
            sample_labels[i * overlap : (i + 1) * overlap] = int(sub_y[i])
        sample_labels[-overlap:] = int(sub_y[-1])

        # Extract ground-truth activity boundaries
        boundaries = []
        change_pts = np.where(np.diff(sample_labels) != 0)[0]
        seg_start  = 0
        for cp in change_pts:
            seg_end  = cp
            length   = seg_end - seg_start + 1
            center   = seg_start + length / 2.0
            act_label = sample_labels[seg_start]
            boundaries.append({
                'activity': act_label,
                'tx': center, 'tl': float(length),
                'start': seg_start, 'end': seg_end
            })
            seg_start = seg_end + 1
        # Last segment
        seg_end  = total - 1
        boundaries.append({
            'activity': sample_labels[seg_start],
            'tx': seg_start + (seg_end - seg_start) / 2.0,
            'tl': float(seg_end - seg_start + 1),
            'start': seg_start, 'end': seg_end
        })

        result[sub] = {
            'signal': stream.astype(np.float32),
            'labels': sample_labels,
            'boundaries': boundaries
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MULTI-SCALE WINDOW GENERATION  (Section III-B of paper)
# ─────────────────────────────────────────────────────────────────────────────

def generate_windows(feature_seq_len, scales):
    """
    Generate anchor windows centred on every unit of the feature sequence.

    For each scale s and each position x, two window lengths are produced:
        l1 = n * sqrt(s),   l2 = n / sqrt(s)
    where n = feature_seq_len.

    Returns
    -------
    windows : np.ndarray  (n_windows, 2)  columns = [centre_x, length]
    """
    n   = feature_seq_len
    windows = []
    for pos in range(n):
        for s in scales:
            for l in [n * np.sqrt(s), n / np.sqrt(s)]:
                windows.append([float(pos), float(l)])
    return np.array(windows, dtype=np.float32)  # (n_win, 2)


def compute_iou_1d(w_x, w_l, t_x, t_l):
    """1-D Intersection-over-Union between window W and truth T."""
    w_start, w_end = w_x - w_l / 2, w_x + w_l / 2
    t_start, t_end = t_x - t_l / 2, t_x + t_l / 2
    inter = max(0.0, min(w_end, t_end) - max(w_start, t_start))
    union = max(w_end, t_end) - min(w_start, t_start)
    return inter / union if union > 0 else 0.0


def compute_offsets(truth_tx, truth_tl, win_x, win_l):
    """Encode ground-truth boundary as offset from anchor (eq. 1-2 in paper)."""
    fx = (truth_tx - win_x) / win_l
    fl = np.log(truth_tl / win_l + 1e-6)
    return fx, fl


def decode_offsets(pred_fx, pred_fl, win_x, win_l):
    """Decode predicted offsets to boundary (eq. 3-4 in paper)."""
    tx = pred_fx * win_l + win_x
    tl = win_l * np.exp(pred_fl)
    return tx, tl


def label_windows(windows, boundaries, num_classes, iou_threshold=0.5):
    """
    Match each anchor window to a ground-truth boundary box.
    Uses the greedy matching algorithm described in Section III-B.

    Returns
    -------
    cls_labels  : (n_windows,)  int  – 0 = background, 1..K = activity class
    loc_targets : (n_windows, 2) float  – (fx, fl) offset targets
    pos_mask    : (n_windows,)  bool
    """
    na = len(windows)
    nb = len(boundaries)
    cls_labels  = np.zeros(na, dtype=np.int32)
    loc_targets = np.zeros((na, 2), dtype=np.float32)

    if nb == 0:
        return cls_labels, loc_targets, np.zeros(na, dtype=bool)

    # Build IOU matrix M ∈ R(na × nb)
    M = np.zeros((na, nb), dtype=np.float32)
    for i, w in enumerate(windows):
        for j, b in enumerate(boundaries):
            M[i, j] = compute_iou_1d(w[0], w[1], b['tx'], b['tl'])

    assigned_w = [-1] * na   # window → truth index
    used_cols  = set()
    used_rows  = set()

    # Step 1: greedy best-match (ensure every GT gets at least one window)
    M_copy = M.copy()
    for _ in range(nb):
        if M_copy.size == 0:
            break
        flat_idx = np.argmax(M_copy)
        ri, ci   = np.unravel_index(flat_idx, M_copy.shape)
        if M_copy[ri, ci] <= 0:
            break
        assigned_w[ri] = ci
        used_rows.add(ri); used_cols.add(ci)
        M_copy[ri, :] = -1
        M_copy[:, ci] = -1

    # Step 2: threshold-based assignment for remaining windows
    for i in range(na):
        if i in used_rows:
            continue
        best_iou = -1; best_j = -1
        for j in range(nb):
            if j in used_cols:
                continue
            if M[i, j] > best_iou:
                best_iou = M[i, j]; best_j = j
        if best_iou >= iou_threshold:
            assigned_w[i] = best_j

    # Encode labels and offsets
    pos_mask = np.zeros(na, dtype=bool)
    for i, j in enumerate(assigned_w):
        if j >= 0:
            b = boundaries[j]
            cls_labels[i]     = int(b['activity'])
            fx, fl            = compute_offsets(b['tx'], b['tl'], windows[i, 0], windows[i, 1])
            loc_targets[i]    = [fx, fl]
            pos_mask[i]       = True

    return cls_labels, loc_targets, pos_mask


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(reconstructed, feature_seq_len, scales, num_classes,
                  fixed_window_samples=600, step=300, training=True):
    """
    Slide a fixed-length window over each subject's continuous stream.
    For each chunk, generate anchor windows and produce supervision targets.

    Returns
    -------
    X_chunks   : (N, fixed_window_samples, 9, 1)
    cls_labels : (N, n_anchors)  int32
    loc_targets: (N, n_anchors, 2)  float32
    pos_masks  : (N, n_anchors)  bool
    """
    windows_template = generate_windows(feature_seq_len, scales)
    n_anchors = len(windows_template)

    X_list, cls_list, loc_list, pos_list = [], [], [], []

    for sub, data in reconstructed.items():
        stream = data['signal']        # (T, 9)
        bounds = data['boundaries']
        T = len(stream)

        for start in range(0, T - fixed_window_samples + 1, step):
            end    = start + fixed_window_samples
            chunk  = stream[start:end]           # (fixed_window_samples, 9)

            # Filter boundaries overlapping this chunk
            local_bounds = []
            for b in bounds:
                b_start = b['start']; b_end = b['end']
                if b_end < start or b_start > end:
                    continue
                # Clip to chunk and re-centre
                cs = max(b_start, start) - start
                ce = min(b_end,   end)   - start
                tl = float(ce - cs + 1)
                tx = cs + tl / 2.0
                local_bounds.append({
                    'activity': b['activity'],
                    'tx': tx, 'tl': tl,
                    'start': cs, 'end': ce
                })

            # Scale window centres from feature-seq coords to chunk coords
            # The feature seq is feature_seq_len units covering fixed_window_samples
            scale_factor = fixed_window_samples / feature_seq_len
            scaled_windows = windows_template.copy()
            scaled_windows[:, 0] *= scale_factor   # centre
            scaled_windows[:, 1] *= scale_factor   # length

            cls_lbl, loc_tgt, pos_mask = label_windows(
                scaled_windows, local_bounds,
                num_classes, iou_threshold=IOU_MATCH_THRESHOLD
            )

            # Reshape chunk: (T, 9) → (T, 9, 1)
            X_list.append(chunk[:, :, np.newaxis])
            cls_list.append(cls_lbl)
            loc_list.append(loc_tgt)
            pos_list.append(pos_mask)

    return (
        np.array(X_list,   dtype=np.float32),    # (N, T, 9, 1)
        np.array(cls_list, dtype=np.int32),       # (N, n_anchors)
        np.array(loc_list, dtype=np.float32),     # (N, n_anchors, 2)
        np.array(pos_list, dtype=bool),           # (N, n_anchors)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL ARCHITECTURE  (Figure 3 & Table I)
# ─────────────────────────────────────────────────────────────────────────────

class SKConv(layers.Layer):
    """Selective-Kernel Convolution (channels-last, 2-D temporal signal)."""

    def __init__(self, features, M=3, G=32, r=32, stride=1, L=32, **kwargs):
        super().__init__(**kwargs)
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        # Split: M branches with different dilation rates
        self.branches = []
        for i in range(M):
            b = keras.Sequential([
                layers.Conv2D(features, kernel_size=(3, 1), strides=(stride, 1),
                              padding='same', dilation_rate=(1 + i, 1),
                              groups=G, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ], name=f'{self.name}_branch_{i}')
            self.branches.append(b)

        # Fuse
        self.gap = layers.GlobalAveragePooling2D(keepdims=False)   # → (B, C)
        self.fc  = keras.Sequential([
            layers.Dense(d, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name=f'{self.name}_fc')

        # Select
        self.fcs      = [layers.Dense(features, name=f'{self.name}_sel_{i}')
                         for i in range(M)]
        self.softmax  = layers.Softmax(axis=1)

    def call(self, x, training=False):
        feats = [b(x, training=training) for b in self.branches]   # list (B,H,W,C)
        feats_stack = tf.stack(feats, axis=1)                       # (B,M,H,W,C)

        feats_U = tf.reduce_sum(feats_stack, axis=1)                # (B,H,W,C)
        feats_S = self.gap(feats_U)                                  # (B,C)
        feats_Z = self.fc(feats_S, training=training)               # (B,d)

        attn = tf.stack([fc(feats_Z) for fc in self.fcs], axis=1)  # (B,M,C)
        attn = self.softmax(attn)                                    # (B,M,C)
        attn = attn[:, :, tf.newaxis, tf.newaxis, :]                # (B,M,1,1,C)

        return tf.reduce_sum(feats_stack * attn, axis=1)             # (B,H,W,C)


def build_mthars(input_shape, num_classes, n_anchors_per_pos,
                 M=3, G=32, r=32, L=32):
    """
    Construct the MTHARS model.

    Architecture (Figure 3, Table I):
      Input → Conv2D(5×1, s=3) → BN → ReLU
            → SKConv(128) → SKConv(256)
            → GlobalAvgPool2D → Dense (bridge to feature-seq length)
            ┌── Conv1D cls head → (B, n_anchors, num_classes+1)
            └── Conv1D loc head → (B, n_anchors, 2)
    """
    inputs = keras.Input(shape=input_shape, name='input')

    # ── Backbone ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, kernel_size=(5, 1), strides=(3, 1),
                      padding='same', use_bias=True, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)

    x = SKConv(128, M=M, G=G, r=r, L=L, name='skconv1')(x)
    x = SKConv(256, M=M, G=G, r=r, L=L, name='skconv2')(x)

    # Collapse spatial (sensor) axis: average over the 9-channel width dim
    # x shape: (B, T_feat, n_sensors, 256)
    x = layers.Lambda(lambda timport tensorflow as tf)
from tensorflow import keras
