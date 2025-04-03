import torch
import torch.nn as nn
import torch.nn.functional as F

class DAFF_AFDStyle(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DAFF_AFDStyle, self).__init__()

        # print("Fuse channel: ", in_channels)

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
        ifPrint = False
        if ifPrint:  # Add a switch for printing only during training
            print(f"Input sem_feat shape: {sem_feat.shape}")
            print(f"Input bound_feat shape: {bound_feat.shape}")

        # Resize if needed
        if sem_feat.shape[2:] != bound_feat.shape[2:]:
            bound_feat = F.interpolate(bound_feat, size=sem_feat.shape[2:], mode='bilinear', align_corners=True)
            if ifPrint:
                print(f"Resized bound_feat shape: {bound_feat.shape}")

        # 1×1 conv projections
        r_s = self.conv_sem(sem_feat)
        r_b = self.conv_bound(bound_feat)
        if ifPrint:
            print(f"r_s shape: {r_s.shape}")
            print(f"r_b shape: {r_b.shape}")

        # Spatial gate from Rs + Rb
        spatial_attention = self.spatial_gate(r_s + r_b)
        if ifPrint:
            print(f"spatial_attention shape: {spatial_attention.shape}")

        # Global average pooled descriptors
        gap_r_s = torch.mean(r_s, dim=(2, 3))  # [B, C]
        gap_r_b = torch.mean(r_b, dim=(2, 3))  # [B, C]
        if ifPrint:
            print(f"gap_r_s shape: {gap_r_s.shape}")
            print(f"gap_r_b shape: {gap_r_b.shape}")

        context_vector = torch.cat([gap_r_s, gap_r_b], dim=1)  # [B, 2C]
        if ifPrint:
            print(f"context_vector shape: {context_vector.shape}")

        # Generate Q, K, V for attention
        qkv = self.qkv_generator(context_vector)  # [B, 6C]
        if ifPrint:
            print(f"qkv shape: {qkv.shape}")

        q, k, v = torch.chunk(qkv, 3, dim=1)      # each: [B, 2C]
        if ifPrint:
            print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        q = q.view(B, 1, -1)                      # [B, 1, 2C]
        k = k.view(B, -1, 1)                      # [B, 2C, 1]
        if ifPrint:
            print(f"q reshaped shape: {q.shape}, k reshaped shape: {k.shape}")

        attention = torch.bmm(q, k) / (C ** 0.5)  # [B, 1, 1]
        attention = torch.softmax(attention, dim=-1)  # still scalar but learned
        if ifPrint:
            print(f"attention shape: {attention.shape}")

        # Apply attention to V
        fusion_weights = (v.view(B, -1) * attention.view(B, 1))  # [B, 2C]
        if ifPrint:
            print(f"fusion_weights shape before projection: {fusion_weights.shape}")

        fusion_weights = self.dropout(self.proj(fusion_weights)).view(B, 2 * C, 1, 1)
        if ifPrint:
            print(f"fusion_weights shape after projection: {fusion_weights.shape}")

        w_s, w_b = torch.split(fusion_weights, C, dim=1)  # [B, C, 1, 1] each
        if ifPrint:
            print(f"w_s shape: {w_s.shape}, w_b shape: {w_b.shape}")

        # Residual-style modulation
        r_s_mod = r_s * (1 + w_s)
        r_b_mod = r_b * (1 + w_b)
        if ifPrint:
            print(f"r_s_mod shape: {r_s_mod.shape}, r_b_mod shape: {r_b_mod.shape}")

        # Final fusion: concat → conv → spatial mask
        fused = torch.cat([r_s_mod, r_b_mod], dim=1)       # [B, 2C, H, W]
        if ifPrint:
            print(f"fused shape after concat: {fused.shape}")

        fused = self.fusion_conv(fused)                    # [B, C, H, W]
        if ifPrint:
            print(f"fused shape after fusion_conv: {fused.shape}")

        fused = fused * spatial_attention                  # apply spatial gate
        if ifPrint:
            print(f"fused shape after spatial_attention: {fused.shape}\n")

        return fused

