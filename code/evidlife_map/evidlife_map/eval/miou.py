"""mIoU and per-class IoU (RQ1 / RQ4 metric — paper Tables III / VII)."""

from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(
    preds: Tensor,
    labels: Tensor,
    num_classes: int,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Compute the ``(C, C)`` confusion matrix.

    Parameters
    ----------
    preds
        Predicted labels, shape ``(N,)`` int64.
    labels
        Ground-truth labels, shape ``(N,)`` int64. ``ignore_index`` is skipped.
    num_classes
        Total number of classes.

    Returns
    -------
    Tensor
        ``(C, C)`` int64 confusion matrix.
    """
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")
    valid = labels != ignore_index
    p = preds[valid]
    l = labels[valid]
    if l.numel() == 0:
        return torch.zeros(num_classes, num_classes, dtype=torch.int64, device=preds.device)
    # In-range check.
    if torch.any(p < 0) or torch.any(p >= num_classes):
        raise ValueError("preds out of range")
    if torch.any(l < 0) or torch.any(l >= num_classes):
        raise ValueError("labels out of range (after ignore mask)")
    flat = l * num_classes + p
    bincount = torch.bincount(flat, minlength=num_classes * num_classes)
    return bincount.view(num_classes, num_classes)


def per_class_iou(
    preds: Tensor,
    labels: Tensor,
    num_classes: int,
    *,
    ignore_index: int = -100,
) -> tuple[Tensor, float]:
    """Compute per-class IoU and mean IoU.

    Returns
    -------
    iou
        ``(C,)`` per-class IoU; classes absent from both preds and labels get ``NaN``.
    miou
        Mean IoU over the classes whose IoU is finite.
    """
    cm = confusion_matrix(preds, labels, num_classes, ignore_index=ignore_index).to(torch.float64)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))
    finite = iou[~iou.isnan()]
    miou = float(finite.mean().item()) if finite.numel() > 0 else float("nan")
    return iou, miou


class IoUTracker:
    """Streaming IoU accumulator.

    Accumulates a confusion matrix across multiple ``update(preds, labels)``
    calls so per-frame inference can stream into the same eval object.
    Used by the RQ1 / RQ4 eval runners and ``scripts/smoke_test_real_kitti.py``.

    Parameters
    ----------
    num_classes
        Number of semantic classes (e.g. 19 for SemanticKITTI).
    ignore_index
        Label value to skip (default -100 to match torch CE convention).
    """

    def __init__(self, num_classes: int, *, ignore_index: int = -100) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self._cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, preds: Tensor, labels: Tensor) -> None:
        """Add a (preds, labels) frame to the running confusion matrix."""
        cm = confusion_matrix(preds.cpu(), labels.cpu(),
                              self.num_classes, ignore_index=self.ignore_index)
        self._cm += cm

    def compute(self) -> dict[str, object]:
        """Return ``{"miou": float, "per_class_iou": list[float|None]}``.

        ``per_class_iou[c]`` is ``None`` if class ``c`` appears in neither
        preds nor labels (avoids penalising the metric for classes that are
        legitimately absent from the test split).
        """
        cm = self._cm.to(torch.float64)
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = tp + fp + fn
        per_class: list[float | None] = []
        finite_vals: list[float] = []
        for c in range(self.num_classes):
            d = float(denom[c].item())
            if d > 0:
                v = float(tp[c].item()) / d
                per_class.append(v)
                finite_vals.append(v)
            else:
                per_class.append(None)
        miou = sum(finite_vals) / len(finite_vals) if finite_vals else float("nan")
        return {"miou": miou, "per_class_iou": per_class}

    def reset(self) -> None:
        self._cm.zero_()
