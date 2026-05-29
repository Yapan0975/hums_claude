"""Submap similarity scoring (paper §III.C Eq 8).

    sim(S₁, S₂) = (1 − β) · cos(h_class¹, h_class²) − β · H(h_vac¹ ∥ h_vac²)

The convex combination of cosine similarity on the class channel and
cross-entropy on the vacuity channel is the single hyperparameter of M2 (the
mixing weight ``β`` is set by ablation A-3).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from evidlife_map.m2_loop_closure.descriptor import SubmapDescriptor


def descriptor_similarity(
    d1: SubmapDescriptor,
    d2: SubmapDescriptor,
    *,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """Implement Eq 8: cosine class minus cross-entropy vacuity.

    Parameters
    ----------
    d1, d2
        Two submap descriptors built by :func:`build_submap_descriptor`.
    beta
        Mixing weight in ``[0, 1]``. ``β = 0`` recovers cosine-only matching
        (the Hydra-style descriptor); ``β = 1`` is the vacuity-only matching.
    eps
        Numerical floor for log + division.

    Returns
    -------
    float
        Similarity score in roughly ``(−∞, 1]``; higher is better.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1]; got {beta!r}")

    # Cosine similarity on the class channel.
    n1 = float(d1.h_class.norm().clamp_min(eps).item())
    n2 = float(d2.h_class.norm().clamp_min(eps).item())
    cos = float((d1.h_class * d2.h_class).sum().item()) / (n1 * n2)

    # Cross-entropy on the vacuity channel.
    # H(p ∥ q) = − Σ p_i · log(q_i + eps).
    cross_entropy = float(
        -(d1.h_vac * torch.log(d2.h_vac.clamp_min(eps))).sum().item()
    )

    return (1.0 - beta) * cos - beta * cross_entropy


def top_k_matches(
    query: SubmapDescriptor,
    gallery: Sequence[SubmapDescriptor],
    *,
    k: int = 5,
    beta: float = 0.5,
    exclude_self: bool = True,
) -> list[tuple[SubmapDescriptor, float]]:
    """Return the ``k`` highest-similarity gallery entries.

    Parameters
    ----------
    query
        The submap we are looking for a loop closure for.
    gallery
        Sequence of candidate submaps (e.g. all sealed submaps in the session).
    k
        Number of top matches to return.
    beta
        Mixing weight in Eq 8.
    exclude_self
        If ``True``, drop descriptors whose ``submap_id`` equals the query's.

    Returns
    -------
    list[tuple[SubmapDescriptor, float]]
        Sorted descending by similarity score.
    """
    scored: list[tuple[SubmapDescriptor, float]] = []
    for cand in gallery:
        if exclude_self and cand.submap_id == query.submap_id:
            continue
        scored.append((cand, descriptor_similarity(query, cand, beta=beta)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def _stable_cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    """Numerically-stable cosine similarity for two 1-D tensors (test helper)."""
    return float(
        ((a * b).sum() / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))).item()
    )
