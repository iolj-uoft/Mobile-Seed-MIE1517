import torch
import torch.nn as nn
import torch.nn.functional as F

class DAFF_AFDStyle(nn.Module):
    """
    A hybrid fusion module that combines DAFF's dynamic fusion with AFD-style learned scalar weights.
    """
    def __init__(self, in_channels):
        super(DAFF_AFDStyle, self).__init__()

        # DAFF-style attention (simplified)
        self.channel_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # AFD-style learned fusion scalars
        self.weight_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels * 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, 1, bias=False),
        )

        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, sem_feat, bound_feat):
        if sem_feat.shape[2:] != bound_feat.shape[2:]:
            bound_feat = F.interpolate(bound_feat, size=sem_feat.shape[2:], mode='bilinear', align_corners=True)

        fused = torch.cat([sem_feat, bound_feat], dim=1)  # [B, 2C, H, W]

        # Channel-wise weighting like DAFF
        channel_att = self.channel_mixer(fused)  # [B, C, 1, 1]

        # AFD-style projection for learned weights
        scalar_weight = self.weight_proj(fused).mean(dim=[2, 3], keepdim=True)  # [B, 2C, 1, 1]
        sem_w, bound_w = torch.split(scalar_weight, sem_feat.shape[1], dim=1)  # [B, C, 1, 1] each

        # Apply gated attention + scalar weighting
        sem_feat = sem_feat * channel_att * (1 + sem_w)
        bound_feat = bound_feat * (1 - channel_att) * (1 + bound_w)

        out = self.fusion_conv(torch.cat([sem_feat, bound_feat], dim=1))
        return out
