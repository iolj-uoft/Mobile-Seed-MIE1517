import torch
import torch.nn as nn
import torch.nn.functional as F


class GCA(nn.Module):
    """
    Gated Channel Adaptive module
    Lightweight enhancement for boundary features
    """
    def __init__(self, in_channels, reduction=16):
        super(GCA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global context embedding
        y = self.global_avg_pool(x)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y  # Gated channel-wise enhancement


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
