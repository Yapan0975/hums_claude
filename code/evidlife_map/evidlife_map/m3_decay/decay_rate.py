"""Decay time-constant τ(u_v) — paper §III.D Eq 11.

    τ(u_v) = τ_min + (τ_max − τ_min) · (1 − u_v)

* ``u_v → 0`` (confidently-known voxel) ⇒ ``τ → τ_max`` (ages slowly).
* ``u_v → 1`` (maximally-uncertain voxel) ⇒ ``τ → τ_min`` (ages fast).

The default values ``τ_min = 60 s`` and ``τ_max = 3600 s`` come from the
research_plan v3 §3.4 narrative ("one minute" and "one hour").
"""

from __future__ import annotations

import torch
from torch import Tensor


def tau_from_vacuity(
    vacuity: Tensor | float,
    *,
    tau_min_s: float = 60.0,
    tau_max_s: float = 3600.0,
) -> Tensor:
    """Implement Eq 11 — linear interpolation in vacuity.

    Parameters
    ----------
    vacuity
        Per-voxel vacuity, scalar or tensor of any shape, values in ``(0, 1]``.
    tau_min_s
        Lower bound of the time-constant (seconds).
    tau_max_s
        Upper bound of the time-constant (seconds).

    Returns
    -------
    Tensor
        Per-voxel decay time-constant in seconds, matching the input shape.
    """
    if tau_min_s <= 0.0 or tau_max_s <= 0.0:
        raise ValueError("tau_min_s and tau_max_s must be > 0")
    if tau_min_s > tau_max_s:
        raise ValueError(f"tau_min_s ({tau_min_s}) must be ≤ tau_max_s ({tau_max_s})")

    if not isinstance(vacuity, Tensor):
        vacuity = torch.tensor(float(vacuity))
    if torch.any(vacuity < 0.0) or torch.any(vacuity > 1.0 + 1e-6):
        raise ValueError("vacuity must lie in [0, 1]")

    return tau_min_s + (tau_max_s - tau_min_s) * (1.0 - vacuity)
