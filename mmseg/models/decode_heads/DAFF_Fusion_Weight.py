import torch
import torch.nn as nn
import torch.nn.functional as F

class DAFF_AFDStyle(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DAFF_AFDStyle, self).__init__()

        # Projection for each stream
        self.conv_sem = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_bound = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Spatial gate: sigmoid(Rs + Rb)
        self.spatial_gate = nn.Sigmoid()

        # Channel attention weights (AFD-style lightweight QKV)
        mid_channels = in_channels // reduction
        self.qkv_generator = nn.Sequential(
            nn.Linear(in_channels * 2, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, (in_channels * 2) * 3, bias=False)  # Q, K, V
        )

        self.proj = nn.Linear(in_channels * 2, in_channels * 2)
        self.dropout = nn.Dropout(0.1)

        # Final fusion conv
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, sem_feat, bound_feat):
        B, C, H, W = sem_feat.shape

        # Resize if needed
        if sem_feat.shape[2:] != bound_feat.shape[2:]:
            bound_feat = F.interpolate(bound_feat, size=sem_feat.shape[2:], mode='bilinear', align_corners=True)

        # 1×1 conv projections
        r_s = self.conv_sem(sem_feat)
        r_b = self.conv_bound(bound_feat)

        # Spatial gate from Rs + Rb
        spatial_attention = self.spatial_gate(r_s + r_b)

        # Global average pooled descriptors
        gap_r_s = torch.mean(r_s, dim=(2, 3))  # [B, C]
        gap_r_b = torch.mean(r_b, dim=(2, 3))  # [B, C]
        context_vector = torch.cat([gap_r_s, gap_r_b], dim=1)  # [B, 2C]

        # Generate Q, K, V for attention
        qkv = self.qkv_generator(context_vector)  # [B, 6C]
        q, k, v = torch.chunk(qkv, 3, dim=1)      # each: [B, 2C]
        q = q.view(B, 1, -1)                      # [B, 1, 2C]
        k = k.view(B, -1, 1)                      # [B, 2C, 1]
        attention = torch.bmm(q, k) / (C ** 0.5)  # [B, 1, 1]
        attention = torch.softmax(attention, dim=-1)  # still scalar but learned

        # Apply attention to V
        fusion_weights = (v.view(B, -1) * attention.view(B, 1))  # [B, 2C]
        fusion_weights = self.dropout(self.proj(fusion_weights)).view(B, 2 * C, 1, 1)
        w_s, w_b = torch.split(fusion_weights, C, dim=1)  # [B, C, 1, 1] each

        # Residual-style modulation
        r_s_mod = r_s * (1 + w_s)
        r_b_mod = r_b * (1 + w_b)

        # Final fusion: concat → conv → spatial mask
        fused = torch.cat([r_s_mod, r_b_mod], dim=1)       # [B, 2C, H, W]
        fused = self.fusion_conv(fused)                    # [B, C, H, W]
        fused = fused * spatial_attention                  # apply spatial gate

        return fused
