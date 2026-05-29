"""Unit tests for calibration metrics (paper §III.B Eq 6)."""

from __future__ import annotations

import torch

from evidlife_map.m1_evidential.calibration import (
    brier_score,
    expected_calibration_error,
    nll_dirichlet,
)


def _perfect_alpha(num_classes_plus_one: int, n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(alpha, labels)`` for a perfectly calibrated, perfectly accurate model."""
    rng = torch.Generator().manual_seed(0)
    labels = torch.randint(0, num_classes_plus_one, (n,), generator=rng)
    alpha = torch.ones(n, num_classes_plus_one)
    # Spike the ground-truth class with very high evidence ⇒ confidence ≈ 1.0
    alpha[torch.arange(n), labels] = 1000.0
    return alpha, labels


def test_ece_perfectly_confident_correct_is_low(num_classes_plus_one: int) -> None:
    alpha, labels = _perfect_alpha(num_classes_plus_one)
    ece = expected_calibration_error(alpha, labels, n_bins=15)
    assert float(ece.item()) < 0.05


def test_ece_uniform_prior_close_to_chance_gap(num_classes_plus_one: int) -> None:
    n = 256
    alpha = torch.ones(n, num_classes_plus_one)
    rng = torch.Generator().manual_seed(1)
    labels = torch.randint(0, num_classes_plus_one, (n,), generator=rng)
    ece = expected_calibration_error(alpha, labels, n_bins=15)
    # At uniform posterior, confidence = 1/(C+1); accuracy ≈ 1/(C+1) too,
    # so |conf-acc| is tiny.
    assert float(ece.item()) < 0.3


def test_brier_score_is_finite_and_in_range(
    alpha_batch: torch.Tensor,
    integer_labels: torch.Tensor,
) -> None:
    bs = brier_score(alpha_batch, integer_labels)
    assert torch.isfinite(bs)
    assert 0.0 <= float(bs.item()) <= 2.0


def test_nll_finite(
    alpha_batch: torch.Tensor,
    integer_labels: torch.Tensor,
) -> None:
    nll = nll_dirichlet(alpha_batch, integer_labels)
    assert torch.isfinite(nll)
    assert float(nll.item()) >= 0.0
