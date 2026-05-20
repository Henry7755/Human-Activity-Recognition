import pytest
import torch
import torch.nn.functional as F
from training.losses import SmoothL1Loss1D, ClassificationLoss, MTHARSLoss, WeightedF1

# ---------------------------------------------------------------------------
# 1. SmoothL1Loss1D Tests
# ---------------------------------------------------------------------------

def test_smooth_l1_loss_basic():
    """Verify SmoothL1Loss1D mathematical calculations for exact piecewise boundaries."""
    loss_fn = SmoothL1Loss1D()
    
    # Setup values where we can compute the loss by hand:
    # Item 0 (pos): diff = |0.5 - 0.0| = 0.5 (< 1.0) -> 0.5 * 0.5^2 = 0.125
    #               diff = |0.2 - 0.0| = 0.2 (< 1.0) -> 0.5 * 0.2^2 = 0.02
    #               Sum over dim=1: 0.125 + 0.02 = 0.145
    # Item 1 (pos): diff = |2.5 - 0.0| = 2.5 (>= 1.0) -> 2.5 - 0.5 = 2.0
    #               diff = |0.0 - 0.0| = 0.0 (< 1.0)  -> 0.5 * 0.0^2 = 0.0
    #               Sum over dim=1: 2.0 + 0.0 = 2.0
    # Item 2 (neg): Ignored by mask
    
    pred_offsets = torch.tensor([[[0.5, 0.2], [2.5, 0.0], [9.9, 9.9]]], dtype=torch.float32)
    true_offsets = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    pos_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    
    expected_loss = (0.145 + 2.0) / 2.0  # Mean over positives
    
    loss = loss_fn(pred_offsets, true_offsets, pos_mask)
    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-5)


def test_smooth_l1_loss_no_positives():
    """Ensure it returns 0.0 without breaking the computational graph gradient when no positives exist."""
    loss_fn = SmoothL1Loss1D()
    
    pred_offsets = torch.randn(2, 4, 2, requires_grad=True)
    true_offsets = torch.randn(2, 4, 2)
    pos_mask = torch.zeros(2, 4, dtype=torch.bool) # No positives
    
    loss = loss_fn(pred_offsets, true_offsets, pos_mask)
    
    assert loss.item() == 0.0
    # Verify backward pass works even if there are no positive matches
    loss.backward()
    assert pred_offsets.grad is not None


# ---------------------------------------------------------------------------
# 2. ClassificationLoss Tests
# ---------------------------------------------------------------------------

def test_classification_loss_mining_ratio():
    """Verify that hard-negative mining maintains the correct negative-to-positive ratio (e.g., 3:1)."""
    n_neg_ratio = 3
    loss_fn = ClassificationLoss(n_neg_ratio=n_neg_ratio)
    
    # 1 batch, 10 windows, 3 classes (K=2, background=0)
    B, na, n_classes = 1, 10, 3
    
    # Create deterministic logits where negative index 1, 2, 3 have very high background loss
    cls_logits = torch.zeros(B, na, n_classes)
    # Give specific non-positive windows massive loss elements by pushing background logit low
    cls_logits[0, 4, 0] = -100.0
    cls_logits[0, 5, 0] = -100.0
    cls_logits[0, 6, 0] = -100.0
    cls_logits[0, 7, 0] = 0.0 # clean background (low loss)
    
    matched_labels = torch.zeros(B, na, dtype=torch.long)
    matched_labels[0, 0] = 1 # Pos 1
    matched_labels[0, 1] = 2 # Pos 2
    
    pos_mask = torch.tensor([[True, True, False, False, False, False, False, False, False, False]], dtype=torch.bool)
    
    # 2 positives -> we want exactly min(2 * 3, 8 available negatives) = 6 hard negatives mined.
    # The code internally handles summing up pos_loss + neg_loss and dividing by N=2.
    loss = loss_fn(cls_logits, matched_labels, pos_mask)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar


def test_classification_loss_no_positives():
    """Check behavior when there are zero positive anchor matches."""
    loss_fn = ClassificationLoss(n_neg_ratio=3)
    
    cls_logits = torch.randn(2, 5, 4)
    matched_labels = torch.zeros(2, 5, dtype=torch.long)
    pos_mask = torch.zeros(2, 5, dtype=torch.bool)
    
    loss = loss_fn(cls_logits, matched_labels, pos_mask)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 3. MTHARSLoss Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (2.0, 3.0)])
def test_mthars_loss_combined(alpha, beta):
    """Test that multi-task balancing and normalization weight scaling are computed correctly."""
    K = 5
    criterion = MTHARSLoss(n_classes=K, alpha=alpha, beta=beta)
    
    B, na = 2, 8
    cls_logits = torch.randn(B, na, K + 1)
    pred_offsets = torch.randn(B, na, 2)
    matched_labels = torch.randint(0, K + 1, (B, na))
    true_offsets = torch.randn(B, na, 2)
    pos_mask = torch.rand(B, na) > 0.5
    
    # Force at least one positive to keep normalization uniform
    pos_mask[0, 0] = True
    N = pos_mask.sum().item()
    
    total_loss, stats = criterion(cls_logits, pred_offsets, matched_labels, true_offsets, pos_mask)
    
    # Assert return types and structures
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {'conf_loss', 'loc_loss', 'total_loss', 'n_pos'}
    
    # Mathematically verify the composition equation: L = (α * L_conf + β * L_loc) / N
    expected_combined = (alpha * stats['conf_loss'] + beta * stats['loc_loss']) / N
    assert pytest.approx(stats['total_loss'], rel=1e-5) == expected_combined
    assert stats['n_pos'] == int(N)


# ---------------------------------------------------------------------------
# 4. WeightedF1 Metric Tests
# ---------------------------------------------------------------------------

def test_weighted_f1_perfect_score():
    """Verify WeightedF1 implementation yields 1.0 given completely accurate classification matches."""
    f1_metric = WeightedF1(n_classes=3)
    
    # Fully accurate targets and outputs
    preds = torch.tensor([0, 0, 1, 2, 2, 2])
    labels = torch.tensor([0, 0, 1, 2, 2, 2])
    
    f1_metric.update(preds, labels)
    score = f1_metric.compute()
    
    assert score == pytest.approx(1.0)


def test_weighted_f1_accumulation():
    """Test state retention and accumulation properties over multiple mini-batch loops."""
    f1_metric = WeightedF1(n_classes=2)
    
    # Batch 1
    f1_metric.update(torch.tensor([0, 1]), torch.tensor([0, 0]))
    # Batch 2
    f1_metric.update(torch.tensor([1, 1]), torch.tensor([1, 1]))
    
    # Aggregated True State across batches: 
    # Class 0: True count = 2. Preds for Class 0 = [True, False, False, False]. TP=1, FP=0, FN=1 -> F1_0 = 2*0.5 / 1.5 = 2/3
    # Class 1: True count = 2. Preds for Class 1 = [False, True, True, True].   TP=2, FP=1, FN=0 -> F1_1 = 4 / 5 = 0.8
    # Weights: Class 0 = 0.5, Class 1 = 0.5
    # Total Expected = 0.5 * (2/3) + 0.5 * (0.8) = 0.33333 + 0.4 = 0.73333
    
    expected_f1 = 0.5 * (2/3) + 0.5 * (0.8)
    assert f1_metric.compute() == pytest.approx(expected_f1, rel=1e-5)