"""
backbone/sknet.py
=================
Selective Kernel Network (SKNet) backbone used in MTHARS.

Paper reference:
    Gao et al., "Deep Neural Networks for Sensor-Based Human Activity
    Recognition Using Selective Kernel Convolution," IEEE TIM, 2021. [Ref 44]

Architecture adapted for 1-D sensor time-series:
    - Layer1   : Conv2D (temporal stem)
    - SKConv   : three parallel dilated Conv2D branches + SE-style attention
    - SKUnit   : bottleneck block with SK convolution
    - SKNet    : full backbone (stem → 3 stages → GAP)

The original user-supplied code targets 2-D image input (H×W).
Here we keep Conv2D but treat sensor data as (C_in, T, 1) so the
network is effectively 1-D while reusing all the 2-D machinery,
exactly as the paper describes for sensor HAR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SKConv  (Selective Kernel Convolution)
# ---------------------------------------------------------------------------

class SKConv(nn.Module):
    """
    Multi-branch convolution with soft attention over kernel sizes.

    Args:
        features   (int): number of input (= output) channels.
        WH         (int): spatial size for legacy GAP kernel (kept for API
                          compatibility; we use adaptive GAP instead).
        M          (int): number of branches (= number of kernel sizes).
        G          (int): group convolution groups.
        r          (int): reduction ratio for the FC bottleneck.
        stride     (int): spatial stride applied to every branch.
        L          (int): minimum bottleneck width (default 32).
    """

    def __init__(self, features: int, WH: int, M: int, G: int,
                 r: int, stride: int = 1, L: int = 32):
        super().__init__()

        # bottleneck dimension  d = max(features/r, L)
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        # M parallel branches with increasing dilation (→ different RF sizes)
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features,
                          kernel_size=3 + i * 2,          # 3, 5, 7, …
                          stride=stride,
                          padding=1 + i,                  # same-padding
                          groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))

        # Global Average Pooling → FC bottleneck
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(features, d)

        # M branch-specific linear layers to produce attention logits
        self.fcs = nn.ModuleList([nn.Linear(d, features) for _ in range(M)])

        self.softmax = nn.Softmax(dim=1)   # over the M branch dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Run all branches, stack along a new dim-1
        branch_outs = []
        for conv in self.convs:
            branch_outs.append(conv(x).unsqueeze_(dim=1))   # (B,1,C,H,W)
        feas = torch.cat(branch_outs, dim=1)                 # (B,M,C,H,W)

        # 2. Fuse branches by element-wise sum, then GAP
        fea_U = feas.sum(dim=1)                              # (B,C,H,W)
        fea_s = self.gap(fea_U).squeeze(-1).squeeze(-1)      # (B,C)

        # 3. FC bottleneck
        fea_z = self.fc(fea_s)                               # (B,d)

        # 4. Compute per-branch attention vectors
        attn_vecs = []
        for fc in self.fcs:
            attn_vecs.append(fc(fea_z).unsqueeze_(dim=1))    # (B,1,C)
        attention = torch.cat(attn_vecs, dim=1)              # (B,M,C)
        attention = self.softmax(attention)                   # soft-max over M
        attention = attention.unsqueeze(-1).unsqueeze(-1)    # (B,M,C,1,1)

        # 5. Weighted sum over branches
        fea_v = (feas * attention).sum(dim=1)                # (B,C,H,W)
        return fea_v


# ---------------------------------------------------------------------------
# SKUnit  (bottleneck residual block with SKConv in the middle)
# ---------------------------------------------------------------------------

class SKUnit(nn.Module):
    """
    Bottleneck residual block:
        1×1 → SKConv → 1×1
    with a skip-connection that projects dimensions when needed.

    Args:
        in_features  (int): input channel count.
        out_features (int): output channel count.
        WH           (int): passed through to SKConv.
        M            (int): SK branches.
        G            (int): group convolution groups.
        r            (int): SK reduction ratio.
        mid_features (int): inner channel width (default: out_features // 2).
        stride       (int): stride for the SK conv (halves spatial size).
        L            (int): minimum SK bottleneck width.
    """

    def __init__(self, in_features: int, out_features: int, WH: int,
                 M: int, G: int, r: int, mid_features: int = None,
                 stride: int = 1, L: int = 32):
        super().__init__()

        if mid_features is None:
            mid_features = out_features // 2

        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
        )

        # Skip connection
        if in_features == out_features:
            self.shortcut = nn.Sequential()                  # identity
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_features),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feas(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# SKNet  (full backbone used as the MTHARS feature extractor)
# ---------------------------------------------------------------------------

class SKNet(nn.Module):
    """
    Three-stage SKNet backbone for sensor HAR.

    Input  : (B, C_sensor, T, 1)  –  treat sensor streams as 2-D
    Output : (B, 1024)            –  flattened feature vector

    The paper's Table I uses Conv2D throughout, so we follow the same
    convention; callers must reshape raw sensor windows accordingly.

    Args:
        in_channels (int): number of sensor input channels (axes × sensors).
        class_num   (int): number of activity classes (used only when the
                           built-in classifier head is active).
    """

    def __init__(self, in_channels: int = 3, class_num: int = 6):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(5, 1),
                      stride=1, padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )

        # Stage 1 : 64 → 256,  stride-2 on the first unit
        self.stage_1 = nn.Sequential(
            SKUnit(64,  256, 32, 2, 8, 2, stride=2), nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),            nn.ReLU(),
        )

        # Stage 2 : 256 → 512,  stride-2
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 8, 2, stride=2), nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),            nn.ReLU(),
        )

        # Stage 3 : 512 → 1024,  stride-2
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 8, 2, stride=2), nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),           nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),           nn.ReLU(),
        )

        # Global Average Pooling → 1024-d feature vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Optional classification head (not used when plugged into MTHARS)
        self.classifier = nn.Linear(1024, class_num)

    def forward(self, x: torch.Tensor,
                return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x               : (B, C, T, 1) sensor tensor.
            return_features : if True, return the 1024-d vector before the
                              classifier (used by MTHARS backbone).
        """
        x = self.stem(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.gap(x)
        feat = x.view(x.size(0), -1)        # (B, 1024)

        if return_features:
            return feat
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# Lightweight 1-D backbone variant
# ---------------------------------------------------------------------------

class SKConv1D(nn.Module):
    """
    Pure 1-D version of SKConv for when sensor data is (B, C, T).
    Easier to plug into the Recognition & Segmentation module which
    operates on 1-D feature sequences.
    """

    def __init__(self, features: int, M: int = 3, G: int = 8,
                 r: int = 2, stride: int = 1, L: int = 32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv1d(features, features,
                          kernel_size=3 + i * 2,
                          stride=stride,
                          padding=1 + i,
                          dilation=1,
                          groups=G),
                nn.BatchNorm1d(features),
                nn.ReLU(inplace=False),
            ))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(features, d)
        self.fcs = nn.ModuleList([nn.Linear(d, features) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [conv(x).unsqueeze(1) for conv in self.convs]  # (B,1,C,T)
        feas  = torch.cat(branch_outs, dim=1)                         # (B,M,C,T)
        fea_U = feas.sum(dim=1)                                        # (B,C,T)
        fea_s = self.gap(fea_U).squeeze(-1)                            # (B,C)
        fea_z = self.fc(fea_s)                                         # (B,d)

        attn_vecs = [fc(fea_z).unsqueeze(1) for fc in self.fcs]       # (B,1,C)
        attention = torch.cat(attn_vecs, dim=1)                        # (B,M,C)
        attention = self.softmax(attention).unsqueeze(-1)              # (B,M,C,1)

        fea_v = (feas * attention).sum(dim=1)                          # (B,C,T)
        return fea_v


class SKNet1D(nn.Module):
    """
    1-D variant of SKNet acting as the MTHARS backbone.

    Input  : (B, C_sensor, T)    raw windowed sensor signals
    Output : (B, feat_dim, T//8) feature sequence passed to the
             Windows Generate + Recognition & Segmentation modules.
    """

    def __init__(self, in_channels: int, feat_dim: int = 256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
        )

        self.stage_1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            SKConv1D(128, M=2, G=4, r=2),
            nn.BatchNorm1d(128), nn.ReLU(),
        )

        self.stage_2 = nn.Sequential(
            nn.Conv1d(128, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            SKConv1D(feat_dim, M=2, G=8, r=2),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
        )

        self.stage_3 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            SKConv1D(feat_dim, M=3, G=8, r=2),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        return x   # (B, feat_dim, T//8)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C, T = 4, 9, 128     # batch=4, 9 sensor axes, 128 time-steps

    # 2-D backbone
    x2d = torch.randn(B, C, T, 1)
    net2d = SKNet(in_channels=C, class_num=6)
    out2d = net2d(x2d)
    print(f"[SKNet 2D]  input {tuple(x2d.shape)} → output {tuple(out2d.shape)}")

    # 1-D backbone
    x1d = torch.randn(B, C, T)
    net1d = SKNet1D(in_channels=C, feat_dim=256)
    feat_seq = net1d(x1d)
    print(f"[SKNet1D]   input {tuple(x1d.shape)} → feat_seq {tuple(feat_seq.shape)}")

    # SKConv1D standalone
    skc = SKConv1D(features=256, M=3, G=8, r=2)
    y   = skc(feat_seq)
    print(f"[SKConv1D]  input {tuple(feat_seq.shape)} → output {tuple(y.shape)}")