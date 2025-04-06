import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCA(nn.Module):
    """
    Gated Channel Attention as described in the architecture diagram.
    """

    def __init__(self, in_channels):
        super(GCA, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))  # usually start from zero
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()

        # 1. L2-norm along (H, W)
        l2_norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)

        # 2. Divide by sqrt(C) (channel-wise norm scaling)
        norm_scale = math.sqrt(C)
        scaled_norm = l2_norm / norm_scale  # Shape: (B, C, 1, 1)

        # 3. α * scaled_norm * β + γ
        gated = self.alpha * scaled_norm * self.beta + self.gamma

        # 4. Tanh
        gated = self.tanh(gated)

        # 5. Add residual: R^B = F^B + F^B * Gated
        out = x + x * gated

        return out


class DAFF(nn.Module):
    """
    Dynamic Adaptive Feature Fusion module
    Used to fuse semantic and boundary features dynamically
    """
    def __init__(self, in_channels):
        super(DAFF, self).__init__()

        self.channel_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, sem_feat, bound_feat):
        # Ensure same spatial shape
        if sem_feat.shape[2:] != bound_feat.shape[2:]:
            bound_feat = F.interpolate(bound_feat, size=sem_feat.shape[2:], mode='bilinear', align_corners=True)

        fused = torch.cat([sem_feat, bound_feat], dim=1)

        # Channel attention
        channel_weight = self.channel_mixer(fused)

        # Spatial attention
        spatial_weight = self.spatial_mixer(fused)

        # Apply attention
        sem_feat = sem_feat * channel_weight * spatial_weight
        bound_feat = bound_feat * (1 - channel_weight) * (1 - spatial_weight)

        out = self.fusion_conv(torch.cat([sem_feat, bound_feat], dim=1))
        return out
