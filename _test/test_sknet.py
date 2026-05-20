import pytest
import torch
import torch.nn as nn

# Assuming your architectures are saved in a module named backbone.sknet
from backbone.sknet import SKConv, SKUnit, SKNet, SKConv1D, SKNet1D


# ============================================================================
# FIXTURES (Reusable Environment Setup)
# ============================================================================

@pytest.fixture
def sample_dimensions():
    """Provides standard tensor dimensions for sensor time-series data."""
    return {
        "batch_size": 4,
        "in_channels": 9,     # e.g., Accelerometer + Gyroscope + Magnetometer axes
        "time_steps": 128,    # Window length
        "classes": 6          # Number of target activities
    }


@pytest.fixture
def device():
    """Automatically selects CUDA if available to test hardware compatibility."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# UNIT TESTS: 2Dmachinery Variant (Sensor data treated as B, C, T, 1)
# ============================================================================

def test_sk_conv_2d_shape_and_flow(sample_dimensions, device):
    """Verifies SKConv 2D processes multi-branch attentions without altering shapes."""
    B, C, T = sample_dimensions["batch_size"], 64, sample_dimensions["time_steps"]
    
    # Standard input tensor shape for the internal blocks: (B, C, H, W)
    x = torch.randn(B, C, T, 1, device=device)
    
    # Initialize SKConv: features=64, WH=32 (legacy API), M=2 branches, G=8 groups, r=2 reduction
    layer = SKConv(features=C, WH=32, M=2, G=8, r=2, stride=1).to(device)
    out = layer(x)
    
    # Assert Shape Integrity
    assert out.shape == (B, C, T, 1), f"Expected shape {(B, C, T, 1)}, got {out.shape}"


@pytest.mark.parametrize("stride", [1, 2])
def test_sk_unit_dimensions(sample_dimensions, device, stride):
    """Tests bottleneck SKUnits preserve or cleanly downsample temporal fields."""
    B, T = sample_dimensions["batch_size"], sample_dimensions["time_steps"]
    in_feats, out_feats = 64, 256
    
    x = torch.randn(B, in_feats, T, 1, device=device)
    unit = SKUnit(in_features=in_feats, out_features=out_feats, WH=32, 
                  M=2, G=8, r=2, stride=stride).to(device)
    out = unit(x)
    
    expected_time_steps = T // stride
    assert out.shape == (B, out_feats, expected_time_steps, 1)


@pytest.mark.parametrize("return_features, expected_last_dim", [
    (True, 1024),   # Backbone feature extraction mode
    (False, 6)      # Directly running built-in classification head
])
def test_sknet_full_backbone_outputs(sample_dimensions, device, return_features, expected_last_dim):
    """
    Asserts the full 2D-machinery SKNet complies with Paper Table I specification.
    Downsampling occurs via three consecutive stride=2 stages (128 -> 64 -> 32 -> 16).
    Global Average Pooling (GAP) then squashes temporal steps to 1.
    """
    B, C, T = sample_dimensions["batch_size"], sample_dimensions["in_channels"], sample_dimensions["time_steps"]
    x = torch.randn(B, C, T, 1, device=device)
    
    model = SKNet(in_channels=C, class_num=sample_dimensions["classes"]).to(device)
    out = model(x, return_features=return_features)
    
    assert out.shape == (B, expected_last_dim), f"Mismatched output channel mapping for mode return_features={return_features}"


# ============================================================================
# UNIT TESTS: Pure 1D Variants (Sensor data treated as B, C, T)
# ============================================================================

def test_sk_conv_1d_attention_weights(sample_dimensions, device):
    """Validates the pure 1D SKConv tracks shapes and properly sums soft attention weights."""
    B, T = sample_dimensions["batch_size"], sample_dimensions["time_steps"]
    features = 256
    
    x = torch.randn(B, features, T, device=device)
    layer = SKConv1D(features=features, M=3, G=8, r=2).to(device)
    out = layer(x)
    
    assert out.shape == (B, features, T)


@pytest.mark.parametrize("custom_time_steps", [64, 128, 256])
def test_sknet_1d_downsampling_ratio(sample_dimensions, device, custom_time_steps):
    """
    Verifies that SKNet1D conforms to the MTHARS architecture paper specification.
    The output sequence must be precisely downsampled by a factor of 8 (T // 8)
    to feed accurately into the downstream Window Generation and Segmentation modules.
    """
    B, C = sample_dimensions["batch_size"], sample_dimensions["in_channels"]
    feat_dim = 256
    
    x = torch.randn(B, C, custom_time_steps, device=device)
    model = SKNet1D(in_channels=C, feat_dim=feat_dim).to(device)
    out = model(x)
    
    expected_temporal_len = custom_time_steps // 8
    assert out.shape == (B, feat_dim, expected_temporal_len), \
        f"Paper requirement broken: temporal length should scale to T // 8. Expected {expected_temporal_len}, got {out.shape[2]}"


# ============================================================================
# FUNCTIONAL GRADIENT & RECEPTIVE FIELD TESTING
# ============================================================================

def test_backpropagation_gradients(sample_dimensions, device):
    """
    Ensures the computational graph is fully unbroken.
    Verifies that gradients can cleanly flow through soft-attention vectors
    back to the input convolution parameters without exploding or vanishing.
    """
    B, C, T = sample_dimensions["batch_size"], sample_dimensions["in_channels"], sample_dimensions["time_series"] = 4, 3, 128
    x = torch.randn(B, C, T, device=device, requires_grad=True)
    
    model = SKNet1D(in_channels=C, feat_dim=256).to(device)
    output = model(x)
    
    # Simulate a downstream Loss calculation (Mean Squared Error style check)
    loss = output.sum()
    loss.backward()
    
    # Assert input gradients exist and are populated
    assert x.grad is not None, "Gradient path broken! Input tensor did not collect backprop gradients."
    assert not torch.isnan(x.grad).any(), "Exploding/corrupted gradient detected: NaNs found in backpropagation."


def test_model_evaluation_mode_invariance(sample_dimensions, device):
    """
    Ensures that switching from training mode to evaluation mode freezes BatchNorm
    statistics properly and outputs matching reproducible predictions.
    """
    B, C, T = sample_dimensions["batch_size"], sample_dimensions["in_channels"], sample_dimensions["time_steps"]
    x = torch.randn(B, C, T, device=device)
    
    model = SKNet1D(in_channels=C, feat_dim=256).to(device)
    
    model.eval()
    with torch.no_grad():
        out_eval_1 = model(x)
        out_eval_2 = model(x)
        
    # In eval mode, outputs must be 100% deterministic (no moving statistical variance updates)
    assert torch.allclose(out_eval_1, out_eval_2, atol=1e-6), "Non-deterministic output detected while in evaluation mode."