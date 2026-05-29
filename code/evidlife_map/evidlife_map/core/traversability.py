"""Vacuity → traversability conversion (RQ3).

The simplest credible "uncertainty-aware traversability head" — described in
research_plan v3 §2 RQ3 — maps per-voxel vacuity to one of three labels:
``{ traversable, deferred, untraversable }``. A voxel with low vacuity and a
ground class is traversable; high vacuity defers the decision; the class label
itself overrides everything else.

This file implements the conversion function; the eval-side metric (safe-region
recall + false-traversable rate) lives in :mod:`evidlife_map.eval.traversability_metrics`.
"""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import dirichlet_mean, vacuity_from_alpha


class TraversabilityLabel(IntEnum):
    """Three-class traversability output."""

    UNTRAVERSABLE = 0
    DEFERRED = 1
    TRAVERSABLE = 2


def vacuity_to_traversability(
    alpha: Tensor,
    *,
    ground_classes: list[int] | tuple[int, ...],
    vacuity_traversable_max: float = 0.30,
    vacuity_defer_max: float = 0.70,
) -> Tensor:
    """Convert per-voxel concentrations to a traversability label.

    Parameters
    ----------
    alpha
        Concentrations of shape ``(V, C+1)``.
    ground_classes
        Class indices considered traversable terrain (e.g. on SemKITTI:
        ``road``, ``sidewalk``, ``parking``, ``other-ground``, ``vegetation``).
    vacuity_traversable_max
        ``u_v ≤`` this threshold AND argmax in ``ground_classes`` → traversable.
    vacuity_defer_max
        ``u_v ≤`` this threshold AND not ground → defer; otherwise
        un-traversable.

    Returns
    -------
    Tensor
        Per-voxel labels of shape ``(V,)`` taking values from
        :class:`TraversabilityLabel`.
    """
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be (V, C+1); got {tuple(alpha.shape)}")
    if not 0.0 < vacuity_traversable_max < vacuity_defer_max <= 1.0:
        raise ValueError("require 0 < vacuity_traversable_max < vacuity_defer_max ≤ 1")
    if not ground_classes:
        raise ValueError("ground_classes must be non-empty")
    if any(c < 0 or c >= alpha.shape[-1] for c in ground_classes):
        raise ValueError("ground_classes contains an out-of-range index")

    vacuity = vacuity_from_alpha(alpha)
    pred = dirichlet_mean(alpha).argmax(dim=-1)

    ground_set = torch.tensor(list(ground_classes), device=alpha.device, dtype=torch.int64)
    is_ground = torch.isin(pred, ground_set)

    labels = torch.full(
        (alpha.shape[0],), int(TraversabilityLabel.UNTRAVERSABLE),
        device=alpha.device, dtype=torch.int64,
    )
    labels = torch.where(
        (vacuity <= vacuity_traversable_max) & is_ground,
        torch.tensor(int(TraversabilityLabel.TRAVERSABLE), device=alpha.device),
        labels,
    )
    labels = torch.where(
        (labels == int(TraversabilityLabel.UNTRAVERSABLE)) & (vacuity <= vacuity_defer_max),
        torch.tensor(int(TraversabilityLabel.DEFERRED), device=alpha.device),
        labels,
    )
    return labels
