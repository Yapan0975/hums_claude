"""PointNet-Vanilla — per-point semantic segmentation head.

A minimal Charles-Qi-2017 PointNet (without T-Net), suitable for an R2-style
LiDAR semantic head when sparse-conv (spconv / Cylinder3D) isn't available.

Architecture
------------
    (N, 4 xyzi)
        ↓ per-point MLP [64, 128, 256]
    (N, 256)
        ↓ global max-pool  →  (1, 256)
        ↓ broadcast        →  (N, 256)
    (N, 256 + 256 = 512)
        ↓ classifier MLP [256, 128, num_classes]
    (N, num_classes) logits

~200k parameters; fits easily on a 5060 / 5090.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


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


class PointNetVanilla(nn.Module):
    """Per-point semantic segmentation PointNet (no T-Net).

    Parameters
    ----------
    in_channels
        Input feature dim per point (4 = xyzi).
    num_classes
        Output classes (19 = SemanticKITTI learning set).
    feat_dims
        Per-point feature extractor channel ladder.
    cls_dims
        Classifier head channel ladder (input dim = ``feat_dims[-1] * 2``).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 19,
        feat_dims: tuple[int, ...] = (64, 128, 256),
        cls_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.feat_dims = tuple(int(d) for d in feat_dims)
        self.cls_dims = tuple(int(d) for d in cls_dims)

        self.point_mlp = _mlp([self.in_channels, *self.feat_dims])
        cls_in = self.feat_dims[-1] * 2
        self.classifier = nn.Sequential(
            _mlp([cls_in, *self.cls_dims], last_relu=True),
            nn.Linear(self.cls_dims[-1], self.num_classes),
        )

    def forward(self, points: Tensor) -> Tensor:
        """Map ``(N, C)`` points to ``(N, num_classes)`` logits.

        Internally re-shapes to a fake batch of 1 so :class:`nn.BatchNorm1d`
        works on the ``(B*N, F)`` flat tensor.
        """
        if points.dim() != 2 or points.shape[1] != self.in_channels:
            raise ValueError(
                f"points must be (N, in_channels={self.in_channels}); "
                f"got {tuple(points.shape)}"
            )
        per_pt = self.point_mlp(points)                     # (N, F)
        global_feat = per_pt.max(dim=0).values             # (F,)
        global_b = global_feat.unsqueeze(0).expand_as(per_pt)  # (N, F)
        fused = torch.cat([per_pt, global_b], dim=-1)       # (N, 2F)
        logits = self.classifier(fused)                     # (N, C)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
