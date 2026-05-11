"""
evaluation/metrics.py
======================
Evaluation metrics used in Sections IV-C, IV-D, IV-E of the paper.

1. NED – Normalised Edit Distance (Equations 9 & 10)
   Used to measure activity *segmentation* accuracy.

2. WeightedF1Score – Weighted F1 (Equation 11)
   Used to measure activity *recognition* accuracy (handles class imbalance).

3. SegmentationEvaluator
   – wraps NED for batch evaluation.

4. RecognitionEvaluator
   – wraps WeightedF1Score for batch evaluation.

5. FullEvaluator
   – runs both evaluators and produces the combined result table
     matching Tables IV & V in the paper.

6. StaticWindowBaseline
   – replicates Section IV-C experiments: train/test several classifiers
     (NB, DT, SVM, LSTM, GRU) with fixed sliding-window lengths
     and report F1 (Table III).

7. DynamicSegmentationBaseline
   – replicates Section IV-D: compare MTHARS against Dynp, BottomUP,
     BinaryCPD baselines (Table IV / Fig. 5).
"""

from __future__ import annotations

import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Normalised Edit Distance  (Equations 9 & 10)
# ---------------------------------------------------------------------------

def levenshtein(seq_a: List[int], seq_b: List[int]) -> int:
    """
    Compute the Levenshtein distance between two integer sequences.

    Equation (10): considers insertion, deletion, and substitution.
    """
    m, n = len(seq_a), len(seq_b)
    # dp[i][j] = edit distance between seq_a[:i] and seq_b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )
    return dp[m][n]


def normalised_edit_distance(pred_seq: List[int],
                             true_seq: List[int]) -> float:
    """
    NED = lev(T̂, T) / length(T)   (Equation 9).

    Args:
        pred_seq : predicted activity label sequence
        true_seq : ground-truth activity label sequence

    Returns:
        NED ∈ [0, ∞)   (lower is better; 0 = perfect)
    """
    if not true_seq:
        return 0.0 if not pred_seq else float(len(pred_seq))
    lev = levenshtein(pred_seq, true_seq)
    return lev / len(true_seq)


def segments_to_label_sequence(segments: List[Dict]) -> List[int]:
    """
    Convert a list of segment dicts to the ordered label sequence
    used for NED computation.

    Deduplicates consecutive identical labels (as per the paper's
    formulation where the sequence is the activity order, not
    frame-level labels).
    """
    seq = []
    for seg in sorted(segments, key=lambda s: s.get('start', 0)):
        lbl = seg.get('label', seg.get('class', 0))
        if not seq or seq[-1] != lbl:
            seq.append(int(lbl))
    return seq


# ---------------------------------------------------------------------------
# 2. Weighted F1 Score  (Equation 11)
# ---------------------------------------------------------------------------

class WeightedF1Score:
    """
    Weighted F1 as defined in Equation (11):
        F1 = 2 * Σ_c (N_c / N_total) * (P_c * R_c) / (P_c + R_c)

    Accumulates predictions across multiple batches before final compute.
    """

    def __init__(self, n_classes: int):
        self.n = n_classes
        self.reset()

    def reset(self):
        self.all_preds  = []
        self.all_labels = []

    def update(self, preds: np.ndarray, labels: np.ndarray):
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())

    def compute(self) -> Dict[str, float]:
        if not self.all_labels:
            return {'f1': 0.0, 'accuracy': 0.0}

        p = np.array(self.all_preds)
        l = np.array(self.all_labels)

        f1  = f1_score(l, p, average='weighted', zero_division=0)
        acc = accuracy_score(l, p)
        return {'f1': float(f1), 'accuracy': float(acc)}


# ---------------------------------------------------------------------------
# 3. Segmentation Evaluator
# ---------------------------------------------------------------------------

class SegmentationEvaluator:
    """
    Evaluate segmentation using NED (Fig. 5 / Table IV of the paper).

    Usage:
        ev = SegmentationEvaluator()
        ev.update(predicted_segments_list, gt_segments_list)
        print(ev.compute())
    """

    def __init__(self):
        self.ned_scores: List[float] = []

    def update(self,
               pred_segs_batch: List[List[Dict]],
               gt_segs_batch:   List[List[Dict]]):
        for pred, gt in zip(pred_segs_batch, gt_segs_batch):
            pred_seq = segments_to_label_sequence(pred)
            gt_seq   = segments_to_label_sequence(gt)
            self.ned_scores.append(normalised_edit_distance(pred_seq, gt_seq))

    def compute(self) -> Dict[str, float]:
        if not self.ned_scores:
            return {'ned_mean': 0.0, 'ned_std': 0.0}
        arr = np.array(self.ned_scores)
        return {'ned_mean': float(arr.mean()),
                'ned_std':  float(arr.std())}

    def reset(self):
        self.ned_scores = []


# ---------------------------------------------------------------------------
# 4. Recognition Evaluator
# ---------------------------------------------------------------------------

class RecognitionEvaluator:
    """Wraps WeightedF1Score for recognition evaluation (Table V)."""

    def __init__(self, n_classes: int):
        self.f1_meter = WeightedF1Score(n_classes=n_classes)

    def update(self, preds: np.ndarray, labels: np.ndarray):
        self.f1_meter.update(preds, labels)

    def compute(self) -> Dict[str, float]:
        return self.f1_meter.compute()

    def reset(self):
        self.f1_meter.reset()


# ---------------------------------------------------------------------------
# 5. Full Evaluator  (recognition + segmentation combined)
# ---------------------------------------------------------------------------

class FullEvaluator:
    """
    Combined evaluator that mirrors the results in Tables IV & V.

    Instantiate once, call update() after every inference step, then
    call compute() at the end.
    """

    def __init__(self, n_classes: int):
        self.seg_ev = SegmentationEvaluator()
        self.rec_ev = RecognitionEvaluator(n_classes)

    def update(self,
               pred_segs:   List[List[Dict]],
               gt_segs:     List[List[Dict]],
               rec_preds:   np.ndarray,
               rec_labels:  np.ndarray):
        self.seg_ev.update(pred_segs, gt_segs)
        self.rec_ev.update(rec_preds, rec_labels)

    def compute(self) -> Dict[str, float]:
        seg = self.seg_ev.compute()
        rec = self.rec_ev.compute()
        return {**seg, **rec}

    def reset(self):
        self.seg_ev.reset()
        self.rec_ev.reset()


# ---------------------------------------------------------------------------
# 6. Static Sliding-Window Baseline  (Section IV-C, Table III)
# ---------------------------------------------------------------------------

class StaticWindowBaseline:
    """
    Replicate the static time-based sliding-window experiments.

    For each combination of (window length t, classifier), trains on
    the training split and evaluates on the test split, reporting F1.

    Classifiers implemented here:
        NB   – Naïve Bayes (GaussianNB)
        DT   – Decision Tree
        SVM  – Linear SVM
        LSTM – delegated to PyTorch (see train_rnn_baseline)
        GRU  – delegated to PyTorch

    Args:
        X_train, X_test : (N, C, T) float32
        y_train, y_test : (N,)      int64
    """

    SKLEARN_MODELS = {
        'NB':  lambda: GaussianNB(),
        'DT':  lambda: DecisionTreeClassifier(random_state=42),
        'SVM': lambda: LinearSVC(max_iter=2000, random_state=42),
    }

    def __init__(self,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test:  np.ndarray, y_test:  np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """(N, C, T) → (N, C*T)"""
        return X.reshape(X.shape[0], -1)

    def run(self, methods: List[str] = None) -> Dict[str, float]:
        """
        Run all sklearn classifiers.

        Returns:
            dict mapping method_name → weighted F1
        """
        if methods is None:
            methods = list(self.SKLEARN_MODELS.keys())

        X_tr = self._flatten(self.X_train)
        X_te = self._flatten(self.X_test)

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        results = {}
        for name in methods:
            if name not in self.SKLEARN_MODELS:
                warnings.warn(f'Unknown model {name}, skipping.')
                continue
            clf = self.SKLEARN_MODELS[name]()
            try:
                clf.fit(X_tr, self.y_train)
                preds = clf.predict(X_te)
                f1 = f1_score(self.y_test, preds,
                              average='weighted', zero_division=0)
                results[name] = float(f1)
            except Exception as e:
                results[name] = float('nan')
                warnings.warn(f'{name} failed: {e}')
        return results


# ---------------------------------------------------------------------------
# 7. Dynamic Segmentation Baseline  (Section IV-D, Table IV)
# ---------------------------------------------------------------------------

class DynamicSegmentationBaseline:
    """
    Wrapper to evaluate the dynamic segmentation baselines described in
    Section IV-D: Dynp, BottomUP, BinaryCPD.

    These methods require external libraries (ruptures for Dynp/BinaryCPD,
    custom BottomUP).  This class provides a unified interface so results
    can be collected into Table IV format.

    If a library is not installed, the method is skipped gracefully.
    """

    @staticmethod
    def run_dynp(signal: np.ndarray, n_bkps: int) -> List[int]:
        """
        Dynp change-point detection (Ref [23] in the paper).
        Requires: pip install ruptures
        """
        try:
            import ruptures as rpt
            model = rpt.Dynp(model='l2').fit(signal)
            bkps  = model.predict(n_bkps=n_bkps)
            return bkps
        except ImportError:
            warnings.warn('ruptures not installed; Dynp skipped.')
            return []

    @staticmethod
    def run_bottomup(signal: np.ndarray, n_bkps: int) -> List[int]:
        """
        BottomUP segmentation (Ref [53]).
        Requires: pip install ruptures
        """
        try:
            import ruptures as rpt
            model = rpt.BottomUp(model='l2').fit(signal)
            return model.predict(n_bkps=n_bkps)
        except ImportError:
            warnings.warn('ruptures not installed; BottomUP skipped.')
            return []

    @staticmethod
    def run_binarycpd(signal: np.ndarray, n_bkps: int) -> List[int]:
        """
        Binary segmentation / Wild Binary Segmentation (Ref [54]).
        Requires: pip install ruptures
        """
        try:
            import ruptures as rpt
            model = rpt.Binseg(model='l2').fit(signal)
            return model.predict(n_bkps=n_bkps)
        except ImportError:
            warnings.warn('ruptures not installed; BinaryCPD skipped.')
            return []

    @staticmethod
    def breakpoints_to_segments(bkps: List[int],
                                labels_per_frame: np.ndarray
                                ) -> List[Dict]:
        """
        Convert a list of break-point indices to segment dicts with
        majority-voted activity labels.
        """
        segs = []
        prev = 0
        for bp in sorted(bkps):
            chunk = labels_per_frame[prev:bp]
            if len(chunk):
                vals, counts = np.unique(chunk, return_counts=True)
                lbl = int(vals[counts.argmax()])
                segs.append({'start': prev, 'end': bp - 1, 'label': lbl})
            prev = bp
        return segs

    def evaluate(self,
                 method:            str,
                 signal:            np.ndarray,
                 frame_labels:      np.ndarray,
                 gt_segments:       List[Dict],
                 n_bkps:            int = 5
                 ) -> float:
        """
        Evaluate one method on one sample and return NED.
        """
        runners = {
            'Dynp':      self.run_dynp,
            'BottomUP':  self.run_bottomup,
            'BinaryCPD': self.run_binarycpd,
        }
        if method not in runners:
            raise ValueError(f'Unknown method {method}')

        bkps = runners[method](signal, n_bkps)
        if not bkps:
            return 1.0    # worst-case NED when method unavailable

        pred_segs = self.breakpoints_to_segments(bkps, frame_labels)
        pred_seq  = segments_to_label_sequence(pred_segs)
        gt_seq    = segments_to_label_sequence(gt_segments)
        return normalised_edit_distance(pred_seq, gt_seq)


# ---------------------------------------------------------------------------
# Utility: print result table
# ---------------------------------------------------------------------------

def print_result_table(results: Dict[str, Dict[str, float]],
                       title: str = 'Results'):
    """
    Pretty-print a nested results dict as a table.
    Outer key = dataset/method, inner key = metric.
    """
    print(f'\n{"="*60}')
    print(f' {title}')
    print(f'{"="*60}')

    # Collect all metric names
    all_metrics = sorted({m for v in results.values() for m in v})
    header = f'{"Method":<20}' + ''.join(f'{m:>12}' for m in all_metrics)
    print(header)
    print('-' * len(header))

    for method, metrics in results.items():
        row = f'{method:<20}'
        for m in all_metrics:
            val = metrics.get(m, float('nan'))
            row += f'{val:>12.4f}'
        print(row)
    print('=' * 60)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # NED test
    pred = [1, 2, 2, 3]
    gt   = [1, 2, 3]
    print(f'NED test: pred={pred} gt={gt} → {normalised_edit_distance(pred, gt):.4f}')

    # Perfect prediction
    print(f'NED perfect: {normalised_edit_distance([1,2,3],[1,2,3]):.4f}')

    # F1 test
    ev = WeightedF1Score(n_classes=6)
    rng = np.random.default_rng(0)
    p = rng.integers(0, 6, 200)
    l = rng.integers(0, 6, 200)
    ev.update(p, l)
    print(f'F1 random baseline: {ev.compute()}')

    # Segmentation evaluator
    seg_ev = SegmentationEvaluator()
    seg_ev.update(
        [[{'start': 0, 'end': 50, 'label': 1},
          {'start': 51, 'end': 100, 'label': 2}]],
        [[{'start': 0, 'end': 48, 'label': 1},
          {'start': 49, 'end': 100, 'label': 2}]]
    )
    print(f'Seg evaluator: {seg_ev.compute()}')