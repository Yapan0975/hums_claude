"""Python-side voxel map abstraction.

This is the Python-facing API that M1 / M2 / M3 talk to. The ground-truth
implementation in production lives in the C++ ``nvblox_evidential`` plug-in
(under ``../../nvblox_evidential/``); the pure-Python ``VoxelMap`` here is a
reference implementation used by unit tests and by the W1 R2-reimpl bring-up.

Voxel keys are 64-bit packed integers ``(x, y, z)`` with each coordinate stored
in a signed 21-bit field — this is the same packing nvblox uses for its
internal block index. Helper functions :func:`xyz_to_key` and :func:`key_to_xyz`
expose the packing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from evidlife_map.m1_evidential.fusion import EvidenceAccumulator

_COORD_BITS = 21
_COORD_MASK = (1 << _COORD_BITS) - 1
_SIGN_BIT = 1 << (_COORD_BITS - 1)
_COORD_MAX = _SIGN_BIT - 1
_COORD_MIN = -_SIGN_BIT


@dataclass(frozen=True, slots=True)
class VoxelKey:
    """3-D integer voxel coordinate (block / sub-block agnostic)."""

    x: int
    y: int
    z: int

    def packed(self) -> int:
        return xyz_to_key(self.x, self.y, self.z)


def xyz_to_key(x: int, y: int, z: int) -> int:
    """Pack ``(x, y, z)`` integers into a 64-bit voxel key."""
    for name, v in (("x", x), ("y", y), ("z", z)):
        if not _COORD_MIN <= v <= _COORD_MAX:
            raise ValueError(
                f"voxel coord {name}={v} out of range "
                f"[{_COORD_MIN}, {_COORD_MAX}] for a 21-bit field"
            )
    ux = x & _COORD_MASK
    uy = y & _COORD_MASK
    uz = z & _COORD_MASK
    return (ux << (2 * _COORD_BITS)) | (uy << _COORD_BITS) | uz


def _unpack_signed(u: int) -> int:
    return u - (1 << _COORD_BITS) if u & _SIGN_BIT else u


def key_to_xyz(key: int) -> tuple[int, int, int]:
    """Inverse of :func:`xyz_to_key`."""
    z = _unpack_signed(key & _COORD_MASK)
    y = _unpack_signed((key >> _COORD_BITS) & _COORD_MASK)
    x = _unpack_signed((key >> (2 * _COORD_BITS)) & _COORD_MASK)
    return x, y, z


class VoxelMap:
    """Pure-Python voxel map (P1 reference; nvblox is the production path).

    Parameters
    ----------
    voxel_size_m
        Edge length of a single voxel in metres (paper §III.B default 0.25 m).
    num_classes_plus_one
        Dimension of the per-voxel ``α`` vector (paper ``C + 1``).
    device, dtype
        Forwarded to :class:`EvidenceAccumulator`.
    """

    def __init__(
        self,
        *,
        voxel_size_m: float = 0.25,
        num_classes_plus_one: int = 20,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be > 0")
        self.voxel_size_m = float(voxel_size_m)
        self.num_classes_plus_one = int(num_classes_plus_one)
        self.accumulator = EvidenceAccumulator(
            num_classes_plus_one=num_classes_plus_one,
            device=device,
            dtype=dtype,
        )

    # ----------------------------------------------------------- coordinates

    def world_to_voxel(self, world_xyz: Tensor) -> Tensor:
        """Convert world-space points to integer voxel coordinates.

        Parameters
        ----------
        world_xyz
            ``(N, 3)`` tensor of metric points.

        Returns
        -------
        Tensor
            ``(N, 3)`` int64 voxel coordinates.
        """
        if world_xyz.dim() != 2 or world_xyz.shape[-1] != 3:
            raise ValueError(f"world_xyz must be (N, 3); got {tuple(world_xyz.shape)}")
        return torch.floor(world_xyz / self.voxel_size_m).to(torch.int64)

    def integrate(
        self,
        world_xyz: Tensor,
        evidence: Tensor,
        *,
        timestamp: float,
    ) -> int:
        """Splat per-point evidence into the voxel grid (paper Eq 4).

        Parameters
        ----------
        world_xyz
            ``(N, 3)`` metric points (LiDAR / camera frame transformed to world).
        evidence
            ``(N, C+1)`` non-negative per-point evidence from :class:`EDLHead`.
        timestamp
            Wall-clock time for the M3 age tracker.

        Returns
        -------
        int
            Number of unique voxels touched.
        """
        if world_xyz.shape[0] != evidence.shape[0]:
            raise ValueError("world_xyz and evidence must have the same N")
        voxel_int = self.world_to_voxel(world_xyz)
        keys = [
            xyz_to_key(int(voxel_int[i, 0]), int(voxel_int[i, 1]), int(voxel_int[i, 2]))
            for i in range(voxel_int.shape[0])
        ]
        # Aggregate per-voxel before pushing to the accumulator, for speed.
        agg: dict[int, Tensor] = {}
        for k, e in zip(keys, evidence, strict=True):
            if k in agg:
                agg[k] = agg[k] + e
            else:
                agg[k] = e.clone()
        if agg:
            keys_list = list(agg.keys())
            stacked = torch.stack([agg[k] for k in keys_list], dim=0)
            self.accumulator.accumulate_batch(keys_list, stacked, timestamp=timestamp)
        return len(agg)

    # ------------------------------------------------------------- accessors

    def get_alpha(self, voxel: VoxelKey | int) -> Tensor:
        key = voxel.packed() if isinstance(voxel, VoxelKey) else voxel
        return self.accumulator.get_alpha(key)

    def __len__(self) -> int:
        return len(self.accumulator)

    def keys(self) -> Iterable[int]:
        return self.accumulator.keys()
