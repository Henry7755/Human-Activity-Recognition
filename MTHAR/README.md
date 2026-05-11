# MTHARS – Full Replication

**Paper**: *A Multi-Task Deep Learning Approach for Sensor-based Human Activity Recognition and Segmentation*  
Duan et al., 2023 · [arXiv:2303.11100](https://arxiv.org/abs/2303.11100)

---

## Project layout

```
MTHARS/
├── backbone/
│   └── sknet.py                  SKNet backbone (2-D and 1-D variants)
├── dataset/
│   └── har_datasets.py           Loaders for all 8 benchmark datasets
├── model/
│   ├── multiscale_windows.py     Window generation, IOU, offset encode/decode, matching
│   └── recognition_segmentation.py  NMS, R&S head, full MTHARS model, Algorithm 1
├── training/
│   ├── losses.py                 SmoothL1 + Cross-entropy + MTHARSLoss (Eq. 5-8)
│   └── trainer.py                Training loop + ablation runner
├── evaluation/
│   └── metrics.py                NED (Eq. 9-10), Weighted-F1 (Eq. 11), baselines
├── utils/
│   └── baselines_and_inference.py  LSTM/GRU baselines, plots, inference demo
├── requirements.txt
└── README.md
```

---

## Paper → Code mapping

| Paper Section | Script | Key class / function |
|---|---|---|
| II-B (SK convolution, Ref [44]) | `backbone/sknet.py` | `SKConv`, `SKConv1D` |
| II-B (SK residual unit) | `backbone/sknet.py` | `SKUnit`, `SKNet`, `SKNet1D` |
| III-A (Problem definition) | `model/multiscale_windows.py` | module docstring |
| III-B (Window generation) | `model/multiscale_windows.py` | `WindowGenerator` |
| III-B (IOU, Jaccard index) | `model/multiscale_windows.py` | `iou_1d`, `iou_matrix` |
| III-B (Window matching) | `model/multiscale_windows.py` | `WindowMatcher` |
| III-B (Eq. 1-2, offset encode) | `model/multiscale_windows.py` | `offset_encode` |
| III-B (Eq. 3-4, offset decode) | `model/multiscale_windows.py` | `offset_decode` |
| III-B (NMS) | `model/recognition_segmentation.py` | `NonMaximumSuppression` |
| III-C (Model overview, Fig. 3) | `model/recognition_segmentation.py` | `MTHARS` |
| III-D (R&S module, Table I) | `model/recognition_segmentation.py` | `RecognitionSegmentationNet` |
| III-E (Loss, Eq. 5-6) | `training/losses.py` | `SmoothL1Loss1D` |
| III-E (Loss, Eq. 7) | `training/losses.py` | `ClassificationLoss` |
| III-E (Loss, Eq. 8) | `training/losses.py` | `MTHARSLoss` |
| III-E (Hard-negative mining, 3:1) | `training/losses.py` | `ClassificationLoss` |
| III-F (Algorithm 1) | `model/recognition_segmentation.py` | `ConcatenateAlgorithm` |
| IV-A (Datasets, Table II) | `dataset/har_datasets.py` | `load_dataset`, `DATASET_INFO` |
| IV-B (Metrics, Eq. 9-11) | `evaluation/metrics.py` | `normalised_edit_distance`, `WeightedF1Score` |
| IV-C (Static window, Table III) | `evaluation/metrics.py` | `StaticWindowBaseline` |
| IV-C (LSTM/GRU baselines) | `utils/baselines_and_inference.py` | `RNNBaseline`, `train_rnn_baseline` |
| IV-D (Dynamic seg, Table IV) | `evaluation/metrics.py` | `DynamicSegmentationBaseline` |
| IV-D (NED bar chart, Fig. 5) | `utils/baselines_and_inference.py` | `plot_ned_bar` |
| IV-E (Recognition results, Table V) | `training/trainer.py` | `Trainer.run` |
| IV-F (Ablation α/β, Table VII) | `training/trainer.py` | `run_ablation_study` |
| IV-F (Ablation scale s, Table VIII) | `training/trainer.py` | `run_ablation_study` |
| Fig. 6 (Confusion matrices) | `utils/baselines_and_inference.py` | `plot_confusion_matrix` |

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download a dataset
Example: UCI HAR Dataset  
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

### 3. Train MTHARS
```bash
# Basic training (UCI, paper default settings)
python training/trainer.py \
    --dataset UCI \
    --data_root /path/to/UCI \
    --epochs 100 \
    --alpha 1.0 --beta 1.0 \
    --scales 2.0 3.0

# PAMAP2 (80/20 split, best scale = s=2,3,4)
python training/trainer.py \
    --dataset PAMAP2 \
    --data_root /path/to/PAMAP2 \
    --epochs 100 \
    --scales 2.0 3.0 4.0

# WISDM best loss weights (Table VII: α=2, β=3)
python training/trainer.py \
    --dataset WISDM \
    --data_root /path/to/WISDM \
    --alpha 2.0 --beta 3.0
```

### 4. Run ablation study (Section IV-F)
```bash
python training/trainer.py \
    --dataset OPPORTUNITY \
    --data_root /path/to/OPPORTUNITY \
    --ablation \
    --epochs 30
```

### 5. Run static window baselines (Section IV-C, Table III)
```python
from dataset.har_datasets import load_dataset, get_dataloaders
from evaluation.metrics import StaticWindowBaseline
import numpy as np

X, y, segs = load_dataset('SKODA', '/path/to/SKODA')
n = len(y)
split = int(0.7 * n)
idx = np.random.permutation(n)
bl = StaticWindowBaseline(X[idx[:split]], y[idx[:split]],
                           X[idx[split:]], y[idx[split:]])
print(bl.run())       # NB, DT, SVM
```

### 6. Run LSTM/GRU baselines (Section IV-C, Table III)
```python
from utils.baselines_and_inference import train_rnn_baseline
result = train_rnn_baseline(X_train, y_train, X_test, y_test,
                             rnn_type='LSTM', epochs=50)
print(result)
```

### 7. Inference on a raw sensor stream
```python
from utils.baselines_and_inference import inference_demo
import numpy as np

raw = np.random.randn(3000, 9).astype(np.float32)   # 3000 frames, 9 axes
segments = inference_demo(
    model_path='checkpoints/UCI/best_model.pt',
    raw_signal=raw,
    in_channels=9, n_classes=6,
    scales=[2.0, 3.0], feat_dim=256, data_len=128,
    activity_names=['walking','upstairs','downstairs',
                    'sitting','standing','lying'],
)
```

---

## Reproducing Table V (Recognition F1)

Run training on each of the 8 datasets and collect the best F1:

```bash
for DS in SKODA HCI PS WISDM UCI OPPORTUNITY PAMAP2 UNIMIB_SHAR; do
    python training/trainer.py \
        --dataset $DS \
        --data_root /data/$DS \
        --epochs 100 \
        --scales 2.0 3.0 \
        --output_dir ./checkpoints
done
```

Expected results (from the paper, Table V):

| Dataset     | SK [44] | MTHARS |
|-------------|---------|--------|
| SKODA       | 0.9510  | 0.9632 |
| HCI         | 0.9377  | 0.9524 |
| PS          | 0.9574  | 0.9721 |
| WISDM       | 0.9725  | 0.9877 |
| UCI         | 0.9558  | 0.9723 |
| OPPORTUNITY | 0.9074  | 0.9213 |
| PAMAP2      | 0.9338  | 0.9480 |
| UNIMIB SHAR | 0.7463  | 0.7571 |

---

## Notes

- The backbone (`SKNet1D`) downsamples the time axis by 8×
  (3 stride-2 stages), so `data_len` must be ≥ 8.
- The window generator mirrors the paper's anchor concept: centers
  are placed at each feature-sequence unit and mapped back to data coords.
- Hard-negative mining uses the paper's 3:1 ratio (configurable via
  `--n_neg_ratio`).
- Mixed-precision training is supported via `--amp` (requires CUDA).

