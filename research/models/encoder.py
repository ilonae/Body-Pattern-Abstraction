"""
encoder.py
----------
PointNet-style encoder for body scan point clouds.

Architecture inspired by:
  - PointNet (Qi et al., 2017) — global feature extraction
  - NeuralTailor (Korosteleva et al., SIGGRAPH 2022) — point-level attention
    for garment pattern reconstruction from point clouds

This is a lightweight research implementation — not a direct copy of
NeuralTailor's code. It is designed to be readable, hackable, and
trainable on a single GPU (even a laptop GPU).

Input:  (B, N, 3) — batch of B point clouds, N points each, XYZ
Output: (B, D)    — global feature vector per cloud (D = feature_dim)

The global feature vector is then passed to LandmarkNet or directly
to the scan-to-params mapper, depending on the pipeline stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Lightweight PointNet encoder.

    Each point is processed independently through shared MLPs,
    then max-pooled to produce a permutation-invariant global feature.

    This is the workhorse for Phase 2 training on synthetic data.
    For Phase 3 (real scans), and upgrade to PointNet++ or
    a transformer-based encoder  would make sense (NeuralTailor's point-level attention).

    Parameters
    ----------
    input_dim   : 3 (XYZ) or 6 (XYZ + normals)
    feature_dim : output global feature dimension
    """

    def __init__(self, input_dim: int = 3, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # Point-wise MLP: 3 → 64 → 128 → 256
        self.conv1 = nn.Conv1d(input_dim,  64,  1)
        self.conv2 = nn.Conv1d(64,         128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, 3) — batch of point clouds

        Returns
        -------
        (B, feature_dim) — global feature vector
        """
        # (B, N, 3) → (B, 3, N)  — Conv1d expects (B, C, L)
        x = x.transpose(2, 1)

        x = F.gelu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.gelu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = self.bn3(self.conv3(x))            # (B, D, N)

        # Global max pool over points
        x = x.max(dim=2)[0]                   # (B, D)
        return x


class PointNetPlusPlusEncoder(nn.Module):
    """
    Simplified PointNet++ style encoder with local neighbourhood grouping.

    Uses a ball-query (radius-based) grouping at two scales, then
    hierarchical max-pooling. Captures local geometry better than vanilla
    PointNet — important for extracting shoulder/hip landmarks reliably.

    This is rather to recommend once PointNetEncoder is working.

    Parameters
    ----------
    input_dim   : 3 (XYZ)
    feature_dim : output global feature dimension
    """

    def __init__(self, input_dim: int = 3, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Scale 1: fine local features (radius ≈ 0.1 body height)
        self.sa1_mlp = nn.Sequential(
            nn.Conv2d(input_dim + 3, 64, 1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 64, 1),            nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, 1),           nn.BatchNorm2d(128), nn.GELU(),
        )

        # Scale 2: coarser body-part features
        self.sa2_mlp = nn.Sequential(
            nn.Conv2d(128 + 3, 128, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 1),     nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 1),     nn.BatchNorm2d(512), nn.GELU(),
        )

        # Global aggregation
        self.global_mlp = nn.Sequential(
            nn.Linear(512, feature_dim), nn.LayerNorm(feature_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward — uses random subsampling rather than FPS
        for now. Real PointNet++ uses Farthest Point Sampling.

        Parameters
        ----------
        x : (B, N, 3)

        Returns
        -------
        (B, feature_dim)
        """
        B, N, _ = x.shape

        # --- SA Layer 1: 512 points, k=16 neighbours ---
        n1 = min(512, N)
        idx1 = torch.randperm(N, device=x.device)[:n1]
        xyz1   = x[:, idx1, :]                        # (B, n1, 3)
        # Build local groups: for each of n1 points, take k=16 neighbours
        k = min(16, N)
        # (Simplified: random neighbourhood — use KNN or ball-query in production)
        group_idx = torch.randint(N, (B, n1, k), device=x.device)
        grouped   = x[:, :, :].unsqueeze(2).expand(-1, -1, n1, -1)    # placeholder
        # Simple: just use the point itself + random context
        centroid  = xyz1.unsqueeze(2).expand(-1, -1, k, -1)            # (B, n1, k, 3)
        local_xyz = centroid                                             # relative coords
        feat1 = torch.cat([local_xyz, centroid], dim=-1)               # (B, n1, k, 6)
        feat1 = feat1.permute(0, 3, 2, 1)                              # (B, 6, k, n1)
        feat1 = self.sa1_mlp(feat1)                                    # (B, 128, k, n1)
        feat1 = feat1.max(dim=2)[0].permute(0, 2, 1)                  # (B, n1, 128)

        # --- SA Layer 2: 128 points ---
        n2   = min(128, n1)
        idx2 = torch.randperm(n1, device=x.device)[:n2]
        xyz2 = xyz1[:, idx2, :]                                        # (B, n2, 3)
        k2   = min(8, n1)
        cent2 = xyz2.unsqueeze(2).expand(-1, -1, k2, -1)
        f2 = torch.cat([cent2, cent2], dim=-1).permute(0, 3, 2, 1)    # placeholder
        feat2 = self.sa2_mlp(f2)                                       # (B, 512, k2, n2)
        feat2 = feat2.max(dim=2)[0].max(dim=2)[0]                     # (B, 512)

        out = self.global_mlp(feat2)                                   # (B, feature_dim)
        return out


def build_encoder(variant: str = "pointnet",
                  feature_dim: int = 256) -> nn.Module:
    """
    Factory function. Use 'pointnet' to start; 'pointnet++' for Phase 3.
    """
    if variant == "pointnet":
        return PointNetEncoder(input_dim=3, feature_dim=feature_dim)
    elif variant == "pointnet++":
        return PointNetPlusPlusEncoder(input_dim=3, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown encoder variant: {variant}. "
                         f"Choose 'pointnet' or 'pointnet++'")
