"""Unit tests for the M1 evidential head (paper §III.B Eq 1–3)."""

from __future__ import annotations

import pytest
import torch

from evidlife_map.m1_evidential.edl_head import (
    EDLHead,
    dirichlet_mean,
    evidence_to_alpha,
    vacuity_from_alpha,
)


def test_evidence_to_alpha_eq1_shifts_by_one() -> None:
    evidence = torch.zeros(4, 20)
    alpha = evidence_to_alpha(evidence)
    assert torch.allclose(alpha, torch.ones(4, 20))


def test_evidence_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        evidence_to_alpha(torch.tensor([-0.1, 0.0, 0.0]))


def test_vacuity_uniform_prior_equals_one(num_classes_plus_one: int) -> None:
    alpha = torch.ones(3, num_classes_plus_one)
    u = vacuity_from_alpha(alpha)
    assert torch.allclose(u, torch.ones(3))


def test_vacuity_in_unit_interval(alpha_batch: torch.Tensor) -> None:
    u = vacuity_from_alpha(alpha_batch)
    assert torch.all(u > 0.0)
    assert torch.all(u <= 1.0 + 1e-6)


def test_dirichlet_mean_sums_to_one(alpha_batch: torch.Tensor) -> None:
    mean = dirichlet_mean(alpha_batch)
    sums = mean.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_edl_head_forward_shapes(num_classes: int) -> None:
    head = EDLHead(in_features=32, num_classes=num_classes, activation="softplus")
    x = torch.randn(8, 32)
    alpha, vacuity = head(x)
    assert alpha.shape == (8, num_classes + 1)
    assert vacuity.shape == (8,)
    assert torch.all(alpha >= 1.0)
    assert torch.all(vacuity > 0.0)
    assert torch.all(vacuity <= 1.0 + 1e-6)


def test_edl_head_gradient_flows(num_classes: int) -> None:
    head = EDLHead(in_features=8, num_classes=num_classes)
    x = torch.randn(4, 8, requires_grad=True)
    alpha, _ = head(x)
    loss = alpha.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_edl_head_rejects_too_few_classes() -> None:
    with pytest.raises(ValueError, match="num_classes must be"):
        EDLHead(in_features=8, num_classes=1)


@pytest.mark.parametrize("activation", ["softplus", "relu", "exp"])
def test_edl_head_supports_all_activations(activation: str, num_classes: int) -> None:
    head = EDLHead(in_features=8, num_classes=num_classes, activation=activation)
    alpha, _ = head(torch.randn(2, 8))
    assert torch.all(alpha >= 1.0)
