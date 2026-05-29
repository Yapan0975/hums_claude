"""Calibration metrics (paper §III.B Eq 6 + standard companion metrics).

Implements:

* :func:`expected_calibration_error` — paper Eq 6, equal-width binning over
  the per-voxel maximum expected probability ``max_k E[p_k|α_v]``.
* :func:`brier_score` — multi-class generalisation of Brier score.
* :func:`nll_dirichlet` — predictive negative log-likelihood under the
  Dirichlet posterior (used in §IV for ablation-grid calibration plots).

All three metrics take ``alpha`` (post-fusion concentrations) and integer
labels, so they can be invoked equivalently from voxel-level and point-level
evaluators.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import dirichlet_mean


def expected_calibration_error(
    alpha: Tensor,
    labels: Tensor,
    *,
    n_bins: int = 15,
    ignore_index: int = -100,
) -> Tensor:
    """Implement paper Eq 6: equal-width ECE over confidence bins.

    Parameters
    ----------
    alpha
        Per-sample Dirichlet concentrations, shape ``(N, C+1)``.
    labels
        Per-sample integer class labels, shape ``(N,)``.
    n_bins
        Number of equal-width confidence bins ``M`` (Eq 6).
    ignore_index
        Label value to skip.

    Returns
    -------
    Tensor
        Scalar ECE in ``[0, 1]``.
    """
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be (N, C+1); got {tuple(alpha.shape)}")
    if labels.dim() != 1 or labels.shape[0] != alpha.shape[0]:
        raise ValueError("labels must be (N,) and match alpha[0]")
    valid = labels != ignore_index
    if not bool(valid.any()):
        return alpha.new_zeros(())
    alpha = alpha[valid]
    labels = labels[valid]

    probs = dirichlet_mean(alpha)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).to(probs.dtype)

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=alpha.device, dtype=alpha.dtype)
    ece = alpha.new_zeros(())
    total = float(alpha.shape[0])
    for m in range(n_bins):
        lo, hi = float(bin_edges[m].item()), float(bin_edges[m + 1].item())
        # Right-open bins except for the last, which is closed on both ends.
        if m < n_bins - 1:
            in_bin = (confidences >= lo) & (confidences < hi)
        else:
            in_bin = (confidences >= lo) & (confidences <= hi)
        bin_count = int(in_bin.sum().item())
        if bin_count == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece = ece + (bin_count / total) * (bin_acc - bin_conf).abs()
    return ece


def brier_score(
    alpha: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Multi-class Brier score under the Dirichlet posterior mean.

    Returns
    -------
    Tensor
        Scalar in ``[0, 2]`` (multi-class convention).
    """
    valid = labels != ignore_index
    if not bool(valid.any()):
        return alpha.new_zeros(())
    alpha = alpha[valid]
    labels = labels[valid]
    probs = dirichlet_mean(alpha)
    y = torch.nn.functional.one_hot(labels, num_classes=alpha.shape[-1]).to(probs.dtype)
    return ((probs - y) ** 2).sum(dim=-1).mean()


def nll_dirichlet(
    alpha: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Categorical NLL evaluated at the Dirichlet posterior mean.

    Returns
    -------
    Tensor
        Mean per-sample NLL.
    """
    valid = labels != ignore_index
    if not bool(valid.any()):
        return alpha.new_zeros(())
    alpha = alpha[valid]
    labels = labels[valid]
    probs = dirichlet_mean(alpha).clamp_min(1e-12)
    return -torch.log(probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)).mean()
