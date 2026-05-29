"""Dirichlet prior (paper §III.B implementation paragraph).

The paper pins the uninformed posterior to the uniform Dirichlet ``Dir(1)``
with prior strength ``s_0 = C + 1``. Concretely, every voxel is initialised
with ``α_v = 1``; the very first observation lifts ``α_v`` away from the
uniform prior and lowers vacuity from its ``1.0`` ceiling.

This module exposes the two helpers M1 / M3 need:

* :func:`dirichlet_prior` — the prior concentration tensor itself.
* :func:`uniform_prior_strength` — the scalar ``s_0 = C + 1``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def dirichlet_prior(
    num_classes_plus_one: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the uniform Dirichlet prior ``α = 1`` over ``C+1`` classes.

    Parameters
    ----------
    num_classes_plus_one
        Output dimension (``C + 1``).

    Returns
    -------
    Tensor
        Tensor of ones, shape ``(C+1,)``.
    """
    if num_classes_plus_one < 2:
        raise ValueError("num_classes_plus_one must be ≥ 2")
    return torch.ones(num_classes_plus_one, device=device, dtype=dtype)


def uniform_prior_strength(num_classes_plus_one: int) -> float:
    """Return the prior strength ``s_0 = C + 1`` (paper §III.B impl)."""
    if num_classes_plus_one < 2:
        raise ValueError("num_classes_plus_one must be ≥ 2")
    return float(num_classes_plus_one)
