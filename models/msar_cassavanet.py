import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation Channel Attention Module."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SpatialAttentionGate(nn.Module):
    """CBAM-style Spatial Attention Gate."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_map = torch.cat([mean_map, max_map], dim=1)
        gate = torch.sigmoid(self.bn(self.conv(spatial_map)))
        return x * gate


class MultiScaleFusionBlock(nn.Module):
    """Parallel multi-scale feature extraction and fusion block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_dim = out_channels // 3
        rem = out_channels - (branch_dim * 3)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, branch_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim + rem, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, branch_dim + rem, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_dim + rem),
            nn.SiLU()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.use_skip = (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)
        out = self.proj(concat)
        return out + x if self.use_skip else out


class DepthwiseSeparableConv(nn.Module):
    """Standard Depthwise-Separable Convolution Layer."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class InvertedSeparableResidualBlock(nn.Module):
    """Inverted Separable Residual Block with optional SE Module."""
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 3, stride: int = 1, use_se: bool = False):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ])

        if use_se:
            layers.append(SqueezeAndExcitation(hidden_dim, reduction=4))

        self.proj_bn = nn.BatchNorm2d(out_channels)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            self.proj_bn
        ])

        self.conv = nn.Sequential(*layers)
        
        # Zero-init last BN in residual branch
        if self.use_residual:
            nn.init.constant_(self.proj_bn.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_residual else self.conv(x)


class MSARCassavaNet(nn.Module):
    """MSAR-CassavaNet Architecture (1.01M Parameters, 147M MACs)."""
    def __init__(self, num_classes: int = 5, use_se: bool = True, use_spatial_gate: bool = True, use_multiscale: bool = True):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )
        self.early_down = DepthwiseSeparableConv(24, 32, stride=2)

        # Stage A (56x56 -> 28x28)
        self.stage_a_down = DepthwiseSeparableConv(32, 64, stride=2)
        self.stage_a_res1 = InvertedSeparableResidualBlock(64, 64, expansion=3, stride=1, use_se=False)
        self.stage_a_res2 = InvertedSeparableResidualBlock(64, 64, expansion=3, stride=1, use_se=use_se)

        # Stage B (28x28 -> 14x14)
        self.stage_b_down = DepthwiseSeparableConv(64, 128, stride=2)
        if use_multiscale:
            self.stage_b_block = MultiScaleFusionBlock(128, 128)
        else:
            self.stage_b_block = InvertedSeparableResidualBlock(128, 128, expansion=3, stride=1, use_se=False)

        # Stage C (14x14)
        self.stage_c_res1 = InvertedSeparableResidualBlock(128, 128, expansion=4, stride=1, use_se=False)
        self.stage_c_res2 = InvertedSeparableResidualBlock(128, 128, expansion=4, stride=1, use_se=use_se)
        self.spatial_gate = SpatialAttentionGate(kernel_size=7) if use_spatial_gate else nn.Identity()

        # Stage D (14x14 -> 7x7)
        self.stage_d_down = DepthwiseSeparableConv(128, 256, stride=2)
        self.stage_d_res1 = InvertedSeparableResidualBlock(256, 256, expansion=3, stride=1, use_se=use_se)

        # Classifier Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.early_down(x)

        x = self.stage_a_down(x)
        x = self.stage_a_res1(x)
        x = self.stage_a_res2(x)

        x = self.stage_b_down(x)
        x = self.stage_b_block(x)

        x = self.stage_c_res1(x)
        x = self.stage_c_res2(x)
        x = self.spatial_gate(x)

        x = self.stage_d_down(x)
        x = self.stage_d_res1(x)

        x = self.head_conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
