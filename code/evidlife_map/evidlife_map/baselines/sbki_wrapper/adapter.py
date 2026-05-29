"""S-BKI [gan2020sbki] adapter (P1 skeleton)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(slots=True)
class SBKIAdapterConfig:
    """Match S-BKI defaults from the deep-read (§T1.6)."""

    num_classes: int = 19
    voxel_size_m: float = 0.40
    kernel_length_scale_m: float = 0.4
    prior_concentration: float = 1.0
    upstream_root: Path | str = "external/sbki"


class SBKIAdapter:
    """Skeleton adapter around the upstream C++ binary."""

    def __init__(self, config: SBKIAdapterConfig | None = None) -> None:
        self.config = config or SBKIAdapterConfig()

    def forward(self, voxel_class_probs: Tensor) -> Tensor:
        if voxel_class_probs.dim() != 2 or voxel_class_probs.shape[-1] != self.config.num_classes:
            raise ValueError(
                f"voxel_class_probs must be (V, C={self.config.num_classes})"
            )
        # P1 skeleton: identity. Real path = subprocess call in W2.
        return voxel_class_probs.clone()
