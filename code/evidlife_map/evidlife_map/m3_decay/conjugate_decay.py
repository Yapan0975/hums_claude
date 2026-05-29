"""Conjugate Dirichlet decay (paper §III.D Eq 12).

    α_v^{(t+Δ)} = ( α_v^{(t)} − 1 ) · exp(−Δ / τ(u_v)) + 1

This is the time-discretisation of the Dirichlet-conjugate pull toward the
uniform prior. Two structural invariants must hold:

1. ``α_v ≥ 1`` is preserved (otherwise Eq 2 vacuity is undefined).
2. The decay preserves the *direction* of ``α_v − 1`` — i.e. the posterior
   mean is unchanged when no class dominates the prior — and inflates vacuity
   monotonically with ``a_v``.

Implementation: the decay multiplier is computed from the *pre-decay* vacuity
(numerically stable; matches the reading on which the M3 trigger fired). The
post-decay clamp at 1 + ε defends against floating-point rounding only.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import vacuity_from_alpha
from evidlife_map.m3_decay.decay_rate import tau_from_vacuity


def decay_concentration(
    alpha: Tensor,
    delta_t_s: float | Tensor,
    *,
    tau_min_s: float = 60.0,
    tau_max_s: float = 3600.0,
) -> Tensor:
    """Apply Eq 12 in-place-safe form (returns a new tensor).

    Parameters
    ----------
    alpha
        Per-voxel concentration of shape ``(..., C+1)`` with values ``≥ 1``.
    delta_t_s
        Time elapsed since the last update, scalar or tensor broadcastable to
        ``alpha.shape[:-1]``.
    tau_min_s, tau_max_s
        Decay-rate bounds (paper §III.D narrative).

    Returns
    -------
    Tensor
        Decayed concentration of the same shape; values clamped at
        ``≥ 1`` to defend against floating-point underflow.
    """
    if torch.any(alpha < 1.0 - 1e-6):
        raise ValueError("input alpha must satisfy α ≥ 1 (paper invariant)")
    if not isinstance(delta_t_s, Tensor):
        delta_t_s = alpha.new_tensor(float(delta_t_s))
    if torch.any(delta_t_s < 0.0):
        raise ValueError("delta_t_s must be ≥ 0")

    vacuity = vacuity_from_alpha(alpha)
    tau = tau_from_vacuity(vacuity, tau_min_s=tau_min_s, tau_max_s=tau_max_s)

    # Broadcast scalar Δ to the right shape, then unsqueeze the class axis.
    if delta_t_s.dim() == 0:
        ratio = delta_t_s / tau
    else:
        ratio = (delta_t_s / tau).reshape(alpha.shape[:-1])
    multiplier = torch.exp(-ratio).unsqueeze(-1)

    decayed = (alpha - 1.0) * multiplier + 1.0
    return torch.clamp(decayed, min=1.0)


def conjugate_decay_step(
    alpha: Tensor,
    *,
    age_seconds: Tensor,
    age_hysteresis_s: float,
    tau_min_s: float,
    tau_max_s: float,
) -> Tensor:
    """Periodic decay sweep used by :mod:`core.nvblox_bridge`.

    Voxels younger than ``age_hysteresis_s`` are left untouched so that
    fresh evidence integrated in the current frame is not immediately decayed.

    Parameters
    ----------
    alpha
        Concentrations of shape ``(V, C+1)``.
    age_seconds
        Per-voxel ages, shape ``(V,)``.
    age_hysteresis_s
        Hysteresis threshold (paper §III.F implementation paragraph).
    tau_min_s, tau_max_s
        Decay-rate bounds.

    Returns
    -------
    Tensor
        Concentrations after the sweep, shape ``(V, C+1)``.
    """
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be (V, C+1); got {tuple(alpha.shape)}")
    if age_seconds.shape != (alpha.shape[0],):
        raise ValueError(
            f"age_seconds shape {tuple(age_seconds.shape)} != (V={alpha.shape[0]},)"
        )

    apply_mask = age_seconds > age_hysteresis_s
    if not bool(apply_mask.any()):
        return alpha.clone()

    out = alpha.clone()
    out[apply_mask] = decay_concentration(
        alpha[apply_mask],
        age_seconds[apply_mask],
        tau_min_s=tau_min_s,
        tau_max_s=tau_max_s,
    )
    return out
