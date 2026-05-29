"""Lifelong / multi-session metrics for RQ4 (research_plan v3 §2 H4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def stale_voxel_removal_precision(
    pred_removed: Tensor,
    gt_stale: Tensor,
) -> float:
    """Precision of stale-voxel removal predictions.

    Parameters
    ----------
    pred_removed
        Boolean mask: ``True`` for voxels the system removed (decayed below
        threshold), shape ``(V,)``.
    gt_stale
        Boolean mask: ``True`` for voxels GT-known to be stale (object moved,
        scene changed), shape ``(V,)``.

    Returns
    -------
    float
        Precision in ``[0, 1]``; ``NaN`` if no voxel was removed.
    """
    if pred_removed.shape != gt_stale.shape:
        raise ValueError("pred_removed and gt_stale must have the same shape")
    n_pred = int(pred_removed.sum().item())
    if n_pred == 0:
        return float("nan")
    tp = int((pred_removed & gt_stale).sum().item())
    return float(tp) / float(n_pred)


def stale_voxel_removal_recall(
    pred_removed: Tensor,
    gt_stale: Tensor,
) -> float:
    n_gt = int(gt_stale.sum().item())
    if n_gt == 0:
        return float("nan")
    tp = int((pred_removed & gt_stale).sum().item())
    return float(tp) / float(n_gt)


@dataclass(slots=True)
class ECEDriftReport:
    """Bookkeeping for the RQ4 ECE-drift metric."""

    session_eces: list[float]
    single_session_baseline_ece: float

    @property
    def max_drift_multiplier(self) -> float:
        if not self.session_eces:
            return float("nan")
        return max(self.session_eces) / max(self.single_session_baseline_ece, 1e-12)

    def passes_h4(self, max_multiplier_threshold: float = 1.5) -> bool:
        return self.max_drift_multiplier <= max_multiplier_threshold


def map_size_multiplier(
    session_voxel_counts: list[int],
    *,
    single_session_baseline: int,
) -> float:
    """Inter-session map growth (paper H4 (iii))."""
    if single_session_baseline <= 0:
        raise ValueError("single_session_baseline must be > 0")
    if not session_voxel_counts:
        return float("nan")
    return float(max(session_voxel_counts)) / float(single_session_baseline)


def _avg(items: list[float]) -> float:
    if not items:
        return float("nan")
    return float(torch.tensor(items, dtype=torch.float64).mean().item())
