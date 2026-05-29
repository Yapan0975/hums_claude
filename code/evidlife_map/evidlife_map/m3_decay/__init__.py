"""M3 — Vacuity-driven voxel decay (paper §III.D, Eq 10–12).

    Eq 10 — voxel age a_v = t_now − t_v,last           :mod:`voxel_age`
    Eq 11 — decay time-constant τ(u_v) linear interp   :mod:`decay_rate`
    Eq 12 — conjugate decay α := (α-1)·exp(−Δ/τ) + 1   :mod:`conjugate_decay`

All three together realise the "stale voxels become re-writable" property
called out in the paper's §III.D side-effect paragraph.
"""

from __future__ import annotations

from evidlife_map.m3_decay.conjugate_decay import conjugate_decay_step, decay_concentration
from evidlife_map.m3_decay.decay_rate import tau_from_vacuity
from evidlife_map.m3_decay.voxel_age import VoxelAgeTracker, age_in_seconds

__all__ = [
    "VoxelAgeTracker",
    "age_in_seconds",
    "conjugate_decay_step",
    "decay_concentration",
    "tau_from_vacuity",
]
