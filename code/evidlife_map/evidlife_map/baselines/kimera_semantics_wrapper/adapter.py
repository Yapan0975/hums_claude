"""Kimera-Semantics adapter (P1 skeleton)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class KimeraSemanticsAdapterConfig:
    num_classes: int = 19
    voxel_size_m: float = 0.25
    rosbag_path: str | None = None     # filled by ``scripts/eval_rq4_lifelong.sh``


class KimeraSemanticsAdapter:
    """Skeleton adapter to the MIT-SPARK Kimera-Semantics docker container."""

    def __init__(self, config: KimeraSemanticsAdapterConfig | None = None) -> None:
        self.config = config or KimeraSemanticsAdapterConfig()

    def forward(self, voxel_class_probs: Tensor) -> Tensor:
        if voxel_class_probs.dim() != 2 or voxel_class_probs.shape[-1] != self.config.num_classes:
            raise ValueError(
                f"voxel_class_probs must be (V, C={self.config.num_classes})"
            )
        return voxel_class_probs.clone()
