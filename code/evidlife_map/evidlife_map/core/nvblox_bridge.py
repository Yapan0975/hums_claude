"""Bridge to the C++ ``nvblox_evidential`` plug-in.

The pure-Python :class:`VoxelMap` is the reference implementation used in
tests; in production we delegate to the C++ ``EvidentialLayer`` (paper §III.F)
through pybind11 bindings emitted by ``nvblox_evidential/CMakeLists.txt``.

This module imports the bindings lazily so that the package still works on
machines where the C++ side has not been built (CI, the documentation builder,
the W1 sanity script). All public methods raise :class:`NvbloxBridgeError`
with a clear remediation hint if the bindings are missing.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


class NvbloxBridgeError(RuntimeError):
    """Raised when the C++ ``nvblox_evidential`` extension is unavailable."""


def _load_bindings() -> ModuleType:
    try:
        return import_module("nvblox_evidential")
    except ImportError as exc:
        raise NvbloxBridgeError(
            "nvblox_evidential C++ extension not importable. "
            "Build it first: \n"
            "    cd nvblox_evidential && cmake -B build && cmake --build build -j"
        ) from exc


class NvbloxBridge:
    """Lazy wrapper around the C++ ``EvidentialLayer``.

    The bridge keeps the same public surface as :class:`evidlife_map.core.voxel_map.VoxelMap`
    so that swapping is a one-line config change.

    Parameters
    ----------
    voxel_size_m
        Voxel edge length in metres.
    num_classes_plus_one
        ``C + 1`` per paper §III.A.
    block_size
        nvblox block edge length in voxels (default ``8``).
    """

    def __init__(
        self,
        *,
        voxel_size_m: float = 0.25,
        num_classes_plus_one: int = 20,
        block_size: int = 8,
    ) -> None:
        if voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be > 0")
        self.voxel_size_m = float(voxel_size_m)
        self.num_classes_plus_one = int(num_classes_plus_one)
        self.block_size = int(block_size)
        self._layer: Any | None = None  # populated by :meth:`_ensure_layer`

    def _ensure_layer(self) -> Any:
        if self._layer is None:
            module = _load_bindings()
            ctor = getattr(module, "EvidentialLayer", None)
            if ctor is None:
                raise NvbloxBridgeError(
                    "nvblox_evidential extension imported but lacks "
                    "EvidentialLayer; rebuild from "
                    "nvblox_evidential/include/nvblox_evidential/evidential_layer.h"
                )
            self._layer = ctor(
                voxel_size_m=self.voxel_size_m,
                num_classes_plus_one=self.num_classes_plus_one,
                block_size=self.block_size,
            )
        return self._layer

    # ------------------------------------------------------------------ I/O

    def add_evidence(self, voxel_keys: list[int], evidence: "Tensor", timestamp: float) -> None:
        """Forward a batch of evidence vectors to the GPU side (paper Eq 4)."""
        layer = self._ensure_layer()
        # The C++ side accepts a contiguous CUDA tensor; we don't enforce here
        # because the bindings will surface a clear error if dtype/device wrong.
        layer.add_evidence(voxel_keys, evidence, float(timestamp))

    def get_alpha(self, voxel_key: int) -> "Tensor":
        layer = self._ensure_layer()
        return layer.get_alpha(int(voxel_key))

    def get_vacuity(self, voxel_key: int) -> float:
        layer = self._ensure_layer()
        return float(layer.get_vacuity(int(voxel_key)))

    def decay(self, timestamp: float) -> int:
        """Trigger the periodic decay kernel (paper Eq 12).

        Returns
        -------
        int
            Number of voxels touched by the sweep.
        """
        layer = self._ensure_layer()
        return int(layer.decay(float(timestamp)))

    def num_voxels(self) -> int:
        layer = self._ensure_layer()
        return int(layer.num_voxels())
