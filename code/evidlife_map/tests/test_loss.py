"""Unit tests for the M1 EDL loss (paper §III.B Eq 5)."""

from __future__ import annotations

import pytest
import torch

from evidlife_map.m1_evidential.loss import EDLLoss


def test_edl_loss_returns_finite_scalar(
    alpha_batch: torch.Tensor,
    integer_labels: torch.Tensor,
    num_classes_plus_one: int,
) -> None:
    loss_mod = EDLLoss(num_classes_plus_one=num_classes_plus_one)
    out = loss_mod(alpha_batch, integer_labels, epoch=0)
    assert torch.isfinite(out["loss"])
    assert out["loss"].ndim == 0


def test_edl_loss_handles_all_ignored(num_classes_plus_one: int) -> None:
    loss_mod = EDLLoss(num_classes_plus_one=num_classes_plus_one)
    alpha = torch.ones(4, num_classes_plus_one) + 0.1
    labels = torch.full((4,), -100, dtype=torch.int64)
    out = loss_mod(alpha, labels, epoch=0)
    assert float(out["loss"].item()) == 0.0


def test_edl_loss_kl_annealing_grows_with_epoch(
    alpha_batch: torch.Tensor,
    integer_labels: torch.Tensor,
    num_classes_plus_one: int,
) -> None:
    loss_mod = EDLLoss(num_classes_plus_one=num_classes_plus_one, kl_anneal_epochs=10)
    early = loss_mod(alpha_batch, integer_labels, epoch=0)
    late = loss_mod(alpha_batch, integer_labels, epoch=9)
    assert float(late["lambda"].item()) >= float(early["lambda"].item())


def test_edl_loss_rejects_shape_mismatch(num_classes_plus_one: int) -> None:
    loss_mod = EDLLoss(num_classes_plus_one=num_classes_plus_one)
    with pytest.raises(ValueError, match="last dim"):
        loss_mod(torch.ones(4, num_classes_plus_one - 1) + 0.1, torch.zeros(4, dtype=torch.int64))


def test_edl_loss_gradient_flows(num_classes_plus_one: int) -> None:
    loss_mod = EDLLoss(num_classes_plus_one=num_classes_plus_one)
    # Build alpha as a leaf tensor that requires grad; the `+ 1.0` ensures the
    # α ≥ 1 invariant without creating an intermediate non-leaf.
    raw = torch.rand(8, num_classes_plus_one)
    alpha = (raw + 1.0).detach().clone().requires_grad_(True)
    labels = torch.randint(0, num_classes_plus_one, (8,))
    out = loss_mod(alpha, labels, epoch=5)
    out["loss"].backward()
    assert alpha.grad is not None
    assert torch.all(torch.isfinite(alpha.grad))
