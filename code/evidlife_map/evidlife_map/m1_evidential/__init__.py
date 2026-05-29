"""M1 — Dirichlet evidential per-voxel semantic fusion.

Covers paper_v1.md §III.B equations 1–6:

    Eq 1 (alpha = e + 1)        → :mod:`edl_head`
    Eq 2 (vacuity u_v)          → :mod:`edl_head`
    Eq 3 (Dirichlet mean)       → :mod:`edl_head`
    Eq 4 (conjugate accumulate) → :mod:`fusion`
    Eq 5 (EDL MSE + KL loss)    → :mod:`loss`
    Eq 6 (ECE)                  → :mod:`calibration`

Module :mod:`prior` exposes the uninformed Dirichlet prior ``Dir(1)`` and the
per-voxel prior-strength scalar ``s_0 = C + 1``.
"""

from __future__ import annotations

from evidlife_map.m1_evidential.calibration import (
    brier_score,
    expected_calibration_error,
    nll_dirichlet,
)
from evidlife_map.m1_evidential.edl_head import EDLHead, evidence_to_alpha, vacuity_from_alpha
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator
from evidlife_map.m1_evidential.loss import EDLLoss
from evidlife_map.m1_evidential.prior import dirichlet_prior, uniform_prior_strength

__all__ = [
    "EDLHead",
    "EDLLoss",
    "EvidenceAccumulator",
    "brier_score",
    "dirichlet_prior",
    "evidence_to_alpha",
    "expected_calibration_error",
    "nll_dirichlet",
    "uniform_prior_strength",
    "vacuity_from_alpha",
]
