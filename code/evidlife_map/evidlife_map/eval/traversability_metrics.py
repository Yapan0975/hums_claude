"""Traversability metrics for RQ3 (research_plan v3 §2 H3)."""

from __future__ import annotations

import torch
from torch import Tensor

from evidlife_map.core.traversability import TraversabilityLabel


def safe_region_recall(pred: Tensor, gt: Tensor) -> float:
    """Recall of the TRAVERSABLE label restricted to GT-traversable voxels.

    Parameters
    ----------
    pred, gt
        Integer label tensors of shape ``(V,)`` taking values from
        :class:`TraversabilityLabel`.

    Returns
    -------
    float
        Recall in ``[0, 1]``; ``NaN`` if the GT has no traversable voxels.
    """
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have the same shape")
    gt_pos = gt == int(TraversabilityLabel.TRAVERSABLE)
    if not bool(gt_pos.any()):
        return float("nan")
    tp = ((pred == int(TraversabilityLabel.TRAVERSABLE)) & gt_pos).sum().item()
    return float(tp) / float(gt_pos.sum().item())


def false_traversable_rate(pred: Tensor, gt: Tensor) -> float:
    """Fraction of voxels predicted TRAVERSABLE that are actually UNTRAVERSABLE.

    Returns
    -------
    float
        Rate in ``[0, 1]``; ``NaN`` if no voxel is predicted traversable.
    """
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have the same shape")
    pred_pos = pred == int(TraversabilityLabel.TRAVERSABLE)
    if not bool(pred_pos.any()):
        return float("nan")
    false_pos = (pred_pos & (gt == int(TraversabilityLabel.UNTRAVERSABLE))).sum().item()
    return float(false_pos) / float(pred_pos.sum().item())


def deferral_rate(pred: Tensor) -> float:
    """Fraction of voxels with the DEFERRED label."""
    if pred.numel() == 0:
        return float("nan")
    return float((pred == int(TraversabilityLabel.DEFERRED)).sum().item()) / float(pred.numel())
