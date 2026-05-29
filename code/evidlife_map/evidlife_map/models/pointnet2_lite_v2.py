"""PointNet2Lite-v2 — Path 4 backbone: stacked kNN-context blocks.

Extends ``pointnet2_lite.PointNet2Lite`` from a single kNN-context layer to
a stack of ``num_blocks`` dilated residual blocks (RandLA-Net spirit:
Local Spatial Encoding + Attentive Pooling, repeated). All blocks operate
at the same N points (no random downsampling), so per-point label gradient
flow stays trivial.

Block structure (per block ``b``, channels ``c_in -> c_out``)::

    feats0 = MLP(c_in -> c_mid)(x)                   # per-point lift
    knn_idx = kNN(xyz, k)                            # shared across blocks
    nb = gather(feats0, knn_idx)                     # (N, k, c_mid)
    # attentive pooling: w = softmax(MLP_att(nb_with_pos)) — weighted sum
    pos_rel = xyz[knn_idx] - xyz[:, None]            # (N, k, 3)
    nb_aug = cat(nb, pos_rel)                        # (N, k, c_mid+3)
    att_logits = MLP_att(nb_aug)                     # (N, k, 1)
    att = softmax(att_logits, dim=k)
    pooled = (nb * att).sum(dim=k)                   # (N, c_mid)
    feats_out = MLP_out(cat(feats0, pooled))         # (N, c_out)
    # residual if c_in == c_out
    return feats_out + x if c_in == c_out else feats_out

Stack ``num_blocks`` such blocks at increasing channel widths. Final
classifier is per-point MLP to (C,) logits.

Param-budget at default config (channels 64/128/256, num_blocks=3,
k=16): ~ 0.6 M params (≈ 2× PointNet2Lite). Wall-clock per frame at
20 000 points: ≈ 2× PointNet2Lite (so ~ 4 min per epoch on 300 frames /
seq × 10 seqs at the 5090).

Note: kNN is computed once per forward and reused across all blocks
because xyz is the same. This is the chief efficiency vs RandLA-Net's
per-block kNN. Trade-off: we lose the *dilation* effect of RandLA-Net.
For a 1-day implementation this is acceptable.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from evidlife_map.models.pointnet2_lite import (
    chunked_pairwise_distances,
    gather_neighbors,
    knn,
)

__all__ = ["PointNet2Lite_v2"]


def _mlp(channels: list[int], *, bn: bool = True, last_relu: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        is_last = (i == len(channels) - 2)
        if not is_last or last_relu:
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DilatedResidualBlock(nn.Module):
    """One LSE + Attentive-Pooling block at fixed N points.

    Parameters
    ----------
    in_dim, out_dim : int
        Channel dimensions in / out.
    mid_dim : int
        Per-point feature dim before pooling. Defaults to in_dim.
    """

    def __init__(self, in_dim: int, out_dim: int, *, mid_dim: int | None = None) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.mid_dim = int(mid_dim if mid_dim is not None else in_dim)
        self.point_mlp = _mlp([self.in_dim, self.mid_dim])
        # Attention MLP: input mid_dim + 3 (relative xyz)
        self.att_mlp = nn.Sequential(
            nn.Linear(self.mid_dim + 3, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, 1),
        )
        # Output MLP: input cat(per_point_feat, pooled) = 2 * mid_dim
        self.out_mlp = _mlp([2 * self.mid_dim, self.out_dim])
        self.residual = (in_dim == out_dim)

    def forward(
        self, feats: Tensor, xyz: Tensor, knn_idx: Tensor,
    ) -> Tensor:
        """Apply one block.

        Parameters
        ----------
        feats : (N, in_dim) Tensor
            Per-point features.
        xyz : (N, 3) Tensor
            Point coordinates (unchanged by this block).
        knn_idx : (N, k) Tensor
            Pre-computed kNN indices (shared across all blocks).

        Returns
        -------
        Tensor
            (N, out_dim) Tensor.
        """
        N, _ = feats.shape
        k = knn_idx.shape[1]
        per_pt = self.point_mlp(feats)                      # (N, mid)
        nb = gather_neighbors(per_pt, knn_idx)              # (N, k, mid)
        # relative positional offsets to neighbours
        nb_xyz = xyz[knn_idx]                                # (N, k, 3)
        pos_rel = nb_xyz - xyz.unsqueeze(1)                  # (N, k, 3)
        att_in = torch.cat([nb, pos_rel], dim=-1)            # (N, k, mid+3)
        # Flatten N*k for BatchNorm1d
        att_logits = self.att_mlp(att_in.reshape(N * k, -1)).reshape(N, k, 1)
        att = torch.softmax(att_logits, dim=1)               # (N, k, 1)
        pooled = (nb * att).sum(dim=1)                       # (N, mid)
        out = self.out_mlp(torch.cat([per_pt, pooled], dim=-1))  # (N, out)
        if self.residual:
            out = out + feats
        return out


class PointNet2Lite_v2(nn.Module):
    """Path 4 backbone: stacked LSE + Attentive-Pooling blocks.

    Parameters
    ----------
    in_channels : int
        Input feature dim (4 = xyzi).
    num_classes : int
        Output class count.
    k : int
        kNN size (shared across blocks).
    channels : tuple[int, ...]
        Channel widths at each block boundary. The number of blocks =
        len(channels) - 1.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 19,
        k: int = 16,
        channels: tuple[int, ...] = (64, 128, 256, 256),
        cls_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.channels = tuple(int(c) for c in channels)
        self.cls_dims = tuple(int(d) for d in cls_dims)
        # First MLP lifts in_channels -> channels[0]
        self.stem = _mlp([self.in_channels, self.channels[0]])
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(self.channels[i], self.channels[i + 1])
            for i in range(len(self.channels) - 1)
        ])
        self.classifier = nn.Sequential(
            _mlp([self.channels[-1], *self.cls_dims], last_relu=True),
            nn.Linear(self.cls_dims[-1], self.num_classes),
        )

    def forward(self, points: Tensor) -> Tensor:
        """Map (N, in_channels) → (N, num_classes) logits."""
        if points.dim() != 2 or points.shape[1] != self.in_channels:
            raise ValueError(
                f"points must be (N, {self.in_channels}); got {tuple(points.shape)}"
            )
        N = points.shape[0]
        xyz = points[:, :3]
        # Shared kNN once
        k_use = min(self.k, N)
        knn_idx = knn(xyz, k=k_use)  # (N, k)

        feats = self.stem(points)
        for block in self.blocks:
            feats = block(feats, xyz, knn_idx)
        logits = self.classifier(feats)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
