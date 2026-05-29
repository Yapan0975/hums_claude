"""Shared pytest fixtures.

Every test in this directory uses synthetic 4-D tensors; we never touch a
real dataset so the suite runs on CPU in CI in seconds.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def seed() -> int:
    return 20260528


@pytest.fixture()
def rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@pytest.fixture()
def num_classes() -> int:
    """SemanticKITTI 19 closed-set classes (research_plan v3 §4.1)."""
    return 19


@pytest.fixture()
def num_classes_plus_one(num_classes: int) -> int:
    """``C + 1`` with the explicit unknown channel (paper §III.A)."""
    return num_classes + 1


@pytest.fixture()
def evidence_batch(
    rng: torch.Generator,
    num_classes_plus_one: int,
) -> torch.Tensor:
    """Synthetic non-negative evidence batch of shape ``(N=64, C+1)``."""
    return torch.rand(64, num_classes_plus_one, generator=rng) * 5.0


@pytest.fixture()
def alpha_batch(evidence_batch: torch.Tensor) -> torch.Tensor:
    """``α = evidence + 1`` so the α ≥ 1 invariant holds."""
    return evidence_batch + 1.0


@pytest.fixture()
def integer_labels(
    rng: torch.Generator,
    num_classes_plus_one: int,
) -> torch.Tensor:
    return torch.randint(0, num_classes_plus_one, (64,), generator=rng)
