"""Vacuity-conditioned submap descriptor (paper §III.C Eq 7).

    d(S) = ( h_class(S), h_vac(S) )

* ``h_class`` ∈ R^{C+1} — ``L¹``-normalised class histogram over **confident**
  voxels (those with vacuity below the per-submap median).
* ``h_vac`` ∈ R^{B} — histogram of per-voxel vacuity ``u_v`` bucketed into
  ``B = 10`` uniform bins over ``(0, 1]`` (paper §III.C, second paragraph).

Together they form the descriptor that :mod:`matcher` consumes via Eq 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import dirichlet_mean, vacuity_from_alpha


@dataclass(frozen=True, slots=True)
class SubmapDescriptor:
    """Two-channel descriptor as defined in paper Eq 7.

    Attributes
    ----------
    h_class
        ``L¹``-normalised class histogram over confident voxels, shape ``(C+1,)``.
    h_vac
        Vacuity histogram with ``B`` bins, shape ``(B,)``; bins are uniform on
        ``(0, 1]``.
    submap_id
        Stable identifier, used by :mod:`matcher` for top-k retrieval.
    num_voxels
        Total voxel count in the submap (used by the consistency check).
    median_vacuity
        Per-submap median vacuity threshold used for the confident-voxel mask.
    """

    h_class: Tensor
    h_vac: Tensor
    submap_id: int
    num_voxels: int
    median_vacuity: float


def build_submap_descriptor(
    alpha: Tensor,
    *,
    submap_id: int,
    n_vac_bins: int = 10,
) -> SubmapDescriptor:
    """Build a descriptor from a submap's voxel concentrations.

    Parameters
    ----------
    alpha
        Per-voxel concentrations for one sealed submap, shape ``(V, C+1)``.
    submap_id
        Stable identifier for the submap.
    n_vac_bins
        Number ``B`` of vacuity histogram bins (paper §III.C: ``B = 10``).

    Returns
    -------
    SubmapDescriptor
    """
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be (V, C+1); got {tuple(alpha.shape)}")
    if alpha.shape[0] == 0:
        raise ValueError("submap has zero voxels; cannot build descriptor")
    num_classes_plus_one = alpha.shape[-1]

    vacuity = vacuity_from_alpha(alpha)  # (V,)
    median = float(vacuity.median().item())
    confident_mask = vacuity <= median

    # h_class — L¹-normalised class histogram over CONFIDENT voxels only.
    if int(confident_mask.sum().item()) == 0:
        h_class = torch.full(
            (num_classes_plus_one,), 1.0 / num_classes_plus_one,
            device=alpha.device, dtype=alpha.dtype,
        )
    else:
        confident_alpha = alpha[confident_mask]
        mean_probs = dirichlet_mean(confident_alpha).sum(dim=0)
        denom = mean_probs.sum().clamp_min(1e-12)
        h_class = mean_probs / denom

    # h_vac — uniform-bin histogram on (0, 1].
    bin_edges = torch.linspace(0.0, 1.0, n_vac_bins + 1, device=alpha.device, dtype=alpha.dtype)
    h_vac = torch.zeros(n_vac_bins, device=alpha.device, dtype=alpha.dtype)
    for b in range(n_vac_bins):
        lo, hi = float(bin_edges[b].item()), float(bin_edges[b + 1].item())
        if b < n_vac_bins - 1:
            mask = (vacuity > lo) & (vacuity <= hi) if b > 0 else (vacuity >= lo) & (vacuity <= hi)
        else:
            mask = (vacuity > lo) & (vacuity <= hi)
        h_vac[b] = mask.sum()
    h_vac = h_vac / h_vac.sum().clamp_min(1e-12)

    return SubmapDescriptor(
        h_class=h_class,
        h_vac=h_vac,
        submap_id=int(submap_id),
        num_voxels=int(alpha.shape[0]),
        median_vacuity=median,
    )
