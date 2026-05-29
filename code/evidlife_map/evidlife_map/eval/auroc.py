"""AUROC + AUPR for vacuity-as-OOD detector (RQ2 lead).

Used by ``scripts/eval_rq2_openset.sh`` to compute the headline numbers of
H2 (AUROC ≥ 0.80, AUPR ≥ 0.60).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _check_inputs(scores: Tensor, is_ood: Tensor) -> tuple[Tensor, Tensor]:
    if scores.dim() != 1 or is_ood.dim() != 1:
        raise ValueError("scores and is_ood must be 1-D")
    if scores.shape != is_ood.shape:
        raise ValueError("scores and is_ood must match in length")
    return scores.detach().cpu().to(torch.float64), is_ood.detach().cpu().to(torch.int64)


def vacuity_auroc(
    vacuity: Tensor,
    is_ood: Tensor,
) -> float:
    """AUROC of vacuity as OOD score.

    Higher vacuity should correspond to higher OOD probability. We implement
    the Mann-Whitney–U formulation that does not need scikit-learn so this
    file stays import-safe in the minimal CI container.

    Parameters
    ----------
    vacuity
        Per-sample vacuity (or any OOD score), shape ``(N,)``.
    is_ood
        Binary mask, ``1`` = unknown / OOD, ``0`` = known / in-distribution.

    Returns
    -------
    float
        AUROC in ``[0, 1]``.
    """
    s, y = _check_inputs(vacuity, is_ood)
    n_pos = int(y.sum().item())
    n_neg = int(y.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank scores ascending. For tied ranks use the average-rank convention.
    sorted_idx = torch.argsort(s)
    sorted_y = y[sorted_idx]
    # Assign average ranks for ties.
    ranks = torch.empty_like(s, dtype=torch.float64)
    n = s.numel()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[sorted_idx[j + 1]] == s[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    sum_rank_pos = float(ranks[y == 1].sum().item())
    u = sum_rank_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def vacuity_aupr(
    vacuity: Tensor,
    is_ood: Tensor,
) -> float:
    """Area under the precision-recall curve, OOD = positive class.

    Returns
    -------
    float
        AUPR in ``[0, 1]`` (chance level = positive-class prevalence).
    """
    s, y = _check_inputs(vacuity, is_ood)
    if int(y.sum().item()) == 0:
        return float("nan")
    # Sort by score descending.
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    tp_cum = y_sorted.cumsum(dim=0).to(torch.float64)
    fp_cum = (1 - y_sorted).cumsum(dim=0).to(torch.float64)
    n_pos = float(y.sum().item())
    recall = tp_cum / n_pos
    precision = tp_cum / (tp_cum + fp_cum).clamp_min(1.0)
    # Trapezoidal AP via interpolating step function.
    recall_prev = torch.cat([torch.zeros(1, dtype=torch.float64), recall[:-1]])
    delta = recall - recall_prev
    return float((precision * delta).sum().item())
