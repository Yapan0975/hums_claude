"""Khronos [schmid2024khronos] adapter (P1 skeleton, R-17 high risk).

The W20 G-6 checkpoint either lights this up or punts to a published-numbers
fallback (see ``risk_register_v3.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(slots=True)
class KhronosAdapterConfig:
    num_classes: int = 19
    voxel_size_m: float = 0.25
    upstream_root: Path | str = "external/khronos"
    use_published_numbers_fallback: bool = False
    cpu_only: bool = True   # Khronos is CPU-only by default; matches deep-read §T1.1


class KhronosAdapter:
    """Skeleton adapter. Real path runs the Khronos docker via subprocess."""

    def __init__(self, config: KhronosAdapterConfig | None = None) -> None:
        self.config = config or KhronosAdapterConfig()

    def forward(self, voxel_class_probs: Tensor) -> Tensor:
        if self.config.use_published_numbers_fallback:
            raise RuntimeError(
                "Adapter configured to use published numbers only; "
                "do not invoke forward() — read published metrics from "
                "risk_log/khronos_published_numbers.json instead."
            )
        if voxel_class_probs.dim() != 2 or voxel_class_probs.shape[-1] != self.config.num_classes:
            raise ValueError(
                f"voxel_class_probs must be (V, C={self.config.num_classes})"
            )
        return voxel_class_probs.clone()
