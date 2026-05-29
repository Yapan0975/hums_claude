"""Evaluation metrics for §IV.

* :mod:`miou` — closed-set mIoU + per-class IoU (RQ1, RQ4).
* :mod:`ece` — re-export of the closed-form ECE from M1.
* :mod:`auroc` — AUROC / AUPR of vacuity-as-OOD (RQ2 lead).
* :mod:`traversability_metrics` — safe-region recall, false-traversable rate (RQ3).
* :mod:`lifelong_metrics` — stale-voxel removal, ECE drift across sessions (RQ4).
"""

from __future__ import annotations

from evidlife_map.eval.auroc import vacuity_aupr, vacuity_auroc
from evidlife_map.eval.ece import expected_calibration_error
from evidlife_map.eval.lifelong_metrics import ECEDriftReport, stale_voxel_removal_precision
from evidlife_map.eval.miou import per_class_iou
from evidlife_map.eval.traversability_metrics import (
    false_traversable_rate,
    safe_region_recall,
)

__all__ = [
    "ECEDriftReport",
    "expected_calibration_error",
    "false_traversable_rate",
    "per_class_iou",
    "safe_region_recall",
    "stale_voxel_removal_precision",
    "vacuity_aupr",
    "vacuity_auroc",
]
