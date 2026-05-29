"""Per-voxel age tracker (paper §III.D Eq 10).

    a_v = t_now − t_{v, last-update}

The tracker holds the timestamps the EvidenceAccumulator stores under its own
key — we deliberately keep the two structures separate so that the decay kernel
can sweep ages without holding the accumulator's lock.
"""

from __future__ import annotations

import torch
from torch import Tensor


def age_in_seconds(t_now: float, t_last_update: float) -> float:
    """Implement Eq 10 scalar version."""
    age = float(t_now) - float(t_last_update)
    if age < 0.0:
        raise ValueError(
            f"t_now ({t_now}) is before t_last_update ({t_last_update}); "
            "clock went backwards"
        )
    return age


class VoxelAgeTracker:
    """Lightweight age bookkeeping for the M3 decay sweep.

    Parameters
    ----------
    initial_capacity
        Hint for the underlying tensor pre-allocation. Grown geometrically.
    device
        Storage device. Stay on CPU on the workstation; move to ``cuda:0`` on
        the Jetson where the decay kernel runs on-device.
    """

    def __init__(
        self,
        *,
        initial_capacity: int = 1024,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self._key_to_idx: dict[int, int] = {}
        self._timestamps: Tensor = torch.full(
            (initial_capacity,), float("nan"), device=self.device, dtype=torch.float64
        )
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _grow(self) -> None:
        new_size = self._timestamps.numel() * 2
        bigger = torch.full((new_size,), float("nan"),
                            device=self.device, dtype=torch.float64)
        bigger[: self._timestamps.numel()] = self._timestamps
        self._timestamps = bigger

    def touch(self, voxel_key: int, timestamp: float) -> None:
        """Set ``t_{v, last-update}`` for ``voxel_key``."""
        idx = self._key_to_idx.get(voxel_key)
        if idx is None:
            if self._size >= self._timestamps.numel():
                self._grow()
            idx = self._size
            self._key_to_idx[voxel_key] = idx
            self._size += 1
        self._timestamps[idx] = float(timestamp)

    def ages_at(self, t_now: float) -> Tensor:
        """Return per-voxel ages aligned with insertion order (``len()`` items)."""
        if self._size == 0:
            return torch.empty(0, device=self.device, dtype=torch.float64)
        view = self._timestamps[: self._size]
        ages = float(t_now) - view
        if torch.any(ages < 0.0):
            raise ValueError(
                "clock went backwards relative to a stored timestamp"
            )
        return ages

    def keys_in_order(self) -> list[int]:
        """Return keys in insertion order, aligned with :meth:`ages_at` output."""
        # Inverse of self._key_to_idx (idx -> key).
        inverse: list[int | None] = [None] * self._size
        for k, idx in self._key_to_idx.items():
            inverse[idx] = k
        return [k for k in inverse if k is not None]
