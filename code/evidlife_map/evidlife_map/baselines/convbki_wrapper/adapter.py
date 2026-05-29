"""ConvBKI [wilson2024convbki] adapter (P1 skeleton).

We deliberately do **not** vendor the upstream code; the W2 bring-up clones
``UMich-CURLY/BKI_ROS`` into ``external/convbki/`` and this adapter wraps
the Python side via the subprocess-launched ROS node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(slots=True)
class ConvBKIAdapterConfig:
    """Parameters that match the ConvBKI deep-read (§T1.4)."""

    num_classes: int = 19
    voxel_size_m: float = 0.20
    kernel_size: int = 5
    learnable: bool = True
    upstream_root: Path | str = "external/convbki"
    device: str = "cuda:0"


class ConvBKIAdapter:
    """Skeleton adapter to ``UMich-CURLY/BKI_ROS``.

    The real `forward()` will call the upstream depthwise-conv-BKI kernel via
    its Python bindings; for the P1 skeleton we emit a uniform posterior so
    the surrounding harness can be exercised without the C++ build.
    """

    def __init__(self, config: ConvBKIAdapterConfig | None = None) -> None:
        self.config = config or ConvBKIAdapterConfig()
        self._upstream_loaded = False

    def forward(self, voxel_class_probs: Tensor) -> Tensor:
        """Run one BKI smoothing pass over the per-voxel class probabilities.

        Parameters
        ----------
        voxel_class_probs
            ``(V, C)`` per-voxel class probability vector.

        Returns
        -------
        Tensor
            ``(V, C)`` smoothed posterior (placeholder = identity in skeleton).
        """
        if voxel_class_probs.dim() != 2 or voxel_class_probs.shape[-1] != self.config.num_classes:
            raise ValueError(
                f"voxel_class_probs must be (V, C={self.config.num_classes}); "
                f"got {tuple(voxel_class_probs.shape)}"
            )
        if not self._upstream_loaded:
            # TODO(W2): import the C++ bindings via
            #   sys.path.insert(0, str(Path(self.config.upstream_root) / "src"))
            # then `from bki_ros import ConvBKILayer` etc.
            return voxel_class_probs.clone()
        # Real path is filled in W2 (research_plan v3 §7 P2).
        raise NotImplementedError("ConvBKI forward path lives in W2 bring-up")
