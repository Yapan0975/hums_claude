"""Unit tests for M3 vacuity-driven decay (paper §III.D Eq 11-12).

The key invariant is ``α_v ≥ 1`` — vacuity is undefined otherwise.
"""

from __future__ import annotations

import pytest
import torch

from evidlife_map.m1_evidential.edl_head import vacuity_from_alpha
from evidlife_map.m3_decay.conjugate_decay import conjugate_decay_step, decay_concentration
from evidlife_map.m3_decay.decay_rate import tau_from_vacuity


@pytest.mark.parametrize("u_v", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_tau_endpoints(u_v: float) -> None:
    tau = tau_from_vacuity(torch.tensor(u_v), tau_min_s=60.0, tau_max_s=3600.0)
    assert 60.0 - 1e-6 <= float(tau.item()) <= 3600.0 + 1e-6


def test_tau_is_decreasing_in_vacuity() -> None:
    low_v = tau_from_vacuity(torch.tensor(0.1))
    high_v = tau_from_vacuity(torch.tensor(0.9))
    assert float(low_v.item()) > float(high_v.item())


def test_decay_preserves_alpha_ge_one(num_classes_plus_one: int) -> None:
    rng = torch.Generator().manual_seed(0)
    alpha = 1.0 + torch.rand(16, num_classes_plus_one, generator=rng) * 50.0
    decayed = decay_concentration(alpha, delta_t_s=10.0)
    assert torch.all(decayed >= 1.0 - 1e-6)


def test_decay_increases_vacuity_monotonically(num_classes_plus_one: int) -> None:
    rng = torch.Generator().manual_seed(0)
    alpha = 1.0 + torch.rand(32, num_classes_plus_one, generator=rng) * 50.0
    u0 = vacuity_from_alpha(alpha)
    u1 = vacuity_from_alpha(decay_concentration(alpha, delta_t_s=1.0))
    u2 = vacuity_from_alpha(decay_concentration(alpha, delta_t_s=100.0))
    assert torch.all(u1 >= u0 - 1e-6)
    assert torch.all(u2 >= u1 - 1e-6)


def test_decay_zero_time_is_identity(num_classes_plus_one: int) -> None:
    alpha = 1.0 + torch.rand(8, num_classes_plus_one) * 10.0
    decayed = decay_concentration(alpha, delta_t_s=0.0)
    assert torch.allclose(decayed, alpha, atol=1e-6)


def test_decay_rejects_negative_time(num_classes_plus_one: int) -> None:
    alpha = torch.ones(4, num_classes_plus_one) + 0.5
    with pytest.raises(ValueError, match="delta_t_s"):
        decay_concentration(alpha, delta_t_s=-1.0)


def test_decay_rejects_alpha_below_one(num_classes_plus_one: int) -> None:
    alpha = torch.full((4, num_classes_plus_one), 0.5)
    with pytest.raises(ValueError, match="α ≥ 1"):
        decay_concentration(alpha, delta_t_s=1.0)


def test_conjugate_decay_step_respects_hysteresis(num_classes_plus_one: int) -> None:
    alpha = 1.0 + torch.rand(8, num_classes_plus_one) * 10.0
    ages = torch.tensor([0.0, 0.5, 1.5, 5.0, 10.0, 0.0, 0.0, 0.0])
    out = conjugate_decay_step(
        alpha,
        age_seconds=ages,
        age_hysteresis_s=1.0,
        tau_min_s=60.0,
        tau_max_s=3600.0,
    )
    # Voxels below the hysteresis must be untouched.
    untouched_mask = ages <= 1.0
    assert torch.allclose(out[untouched_mask], alpha[untouched_mask])
