"""ECE — re-export of paper §III.B Eq 6 + streaming calculator wrapper.

The closed-form Eq 6 implementation lives in
:mod:`evidlife_map.m1_evidential.calibration` so that training-time inference
and §IV evaluation share one code path. ``ECECalculator`` is a thin streaming
wrapper used by ``evidlife_map/eval/rq1.py`` and ``smoke_test_real_kitti.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evidlife_map.m1_evidential.calibration import (
    brier_score,
    expected_calibration_error,
    nll_dirichlet,
)

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "nll_dirichlet",
    "ECECalculator",
]


class ECECalculator:
    """Streaming expected-calibration-error accumulator.

    Buffers per-frame (probabilities, labels) pairs, then concatenates and
    delegates to :func:`expected_calibration_error` at :meth:`compute` time.
    Memory cost is ``O(N_total × C × 4 bytes)`` — bounded by the eval set.

    Parameters
    ----------
    num_bins
        Equal-width confidence bin count ``M`` (Eq 6 default 15).
    ignore_index
        Label value to skip (default ``-100`` to match torch CE).
    """

    def __init__(self, num_bins: int = 15, *, ignore_index: int = -100) -> None:
        self.num_bins = int(num_bins)
        self.ignore_index = int(ignore_index)
        self._probs: list[Tensor] = []
        self._labels: list[Tensor] = []

    def update(self, probs: Tensor, labels: Tensor) -> None:
        """Add one (per-point probs, labels) frame.

        Parameters
        ----------
        probs
            ``(N, C)`` non-negative probabilities (per-point softmax / Dirichlet mean).
        labels
            ``(N,)`` int64 ground-truth labels.
        """
        if probs.dim() != 2 or labels.dim() != 1 or probs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"shape mismatch: probs {tuple(probs.shape)} vs labels {tuple(labels.shape)}"
            )
        self._probs.append(probs.detach().cpu().to(torch.float32))
        self._labels.append(labels.detach().cpu().to(torch.int64))

    def compute(self) -> float:
        """Return scalar ECE over all accumulated samples."""
        if not self._probs:
            return float("nan")
        all_probs = torch.cat(self._probs, dim=0)
        all_labels = torch.cat(self._labels, dim=0)
        # Fake Dirichlet: alpha = probs * (C+1); dirichlet_mean(alpha) = probs.
        c = all_probs.shape[1]
        alpha = all_probs * float(c)
        # Eq 6 requires α >= 1; clamp to avoid degenerate bins.
        alpha = alpha.clamp_min(1e-3)
        ece = expected_calibration_error(alpha, all_labels, n_bins=self.num_bins, ignore_index=self.ignore_index)
        return float(ece.item())

    def reset(self) -> None:
        self._probs.clear()
        self._labels.clear()
