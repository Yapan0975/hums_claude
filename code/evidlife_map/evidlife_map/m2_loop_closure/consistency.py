"""Post-registration consistency check (paper §III.C Eq 9).

After a candidate match is verified by geometric registration (semantic-ICP
restricted to confident voxels in both submaps), Eq 9 thresholds the median
per-class disagreement of the overlap region:

    Disagree(S₁, S₂) = median_{v ∈ overlap} [ 1 − ⟨E[p|α_v¹], E[p|α_v²]⟩ ]

A loop is accepted only if Disagree ≤ disagree_max (default 0.4, see
``configs/evidlife_kitti.yaml``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import dirichlet_mean


def disagreement_score(
    alpha_overlap_1: Tensor,
    alpha_overlap_2: Tensor,
) -> float:
    """Implement Eq 9: median over overlap voxels of ``1 − ⟨p̄₁, p̄₂⟩``.

    Parameters
    ----------
    alpha_overlap_1, alpha_overlap_2
        Concentration tensors for the overlap voxel set in each submap, both
        shape ``(N_overlap, C+1)`` and **in matching voxel order**.

    Returns
    -------
    float
        Disagreement score in ``[0, 1]``; lower is better.
    """
    if alpha_overlap_1.shape != alpha_overlap_2.shape:
        raise ValueError(
            f"shape mismatch: {tuple(alpha_overlap_1.shape)} vs "
            f"{tuple(alpha_overlap_2.shape)} (overlap voxels must be aligned)"
        )
    if alpha_overlap_1.dim() != 2:
        raise ValueError("alpha tensors must be (N_overlap, C+1)")
    if alpha_overlap_1.shape[0] == 0:
        # Empty overlap = degenerate match; return the worst score so the
        # caller's threshold check rejects it.
        return 1.0

    p1 = dirichlet_mean(alpha_overlap_1)
    p2 = dirichlet_mean(alpha_overlap_2)
    inner = (p1 * p2).sum(dim=-1)
    return float((1.0 - inner).median().item())


def accept_loop(
    alpha_overlap_1: Tensor,
    alpha_overlap_2: Tensor,
    *,
    disagree_max: float = 0.4,
) -> bool:
    """Convenience wrapper: return ``True`` iff Eq 9 ≤ ``disagree_max``."""
    return disagreement_score(alpha_overlap_1, alpha_overlap_2) <= disagree_max


def fuse_overlap(
    alpha_overlap_1: Tensor,
    alpha_overlap_2: Tensor,
) -> Tensor:
    """Conjugate inter-session fusion in the overlap (post-Eq 9).

    The paper (§III.C, last paragraph) prescribes parameter-free conjugate
    addition under the assumption of independent observations across sessions:

        α_v ← α_v^{(1)} + α_v^{(2)} − 1

    The ``− 1`` term subtracts the duplicated prior so that the final
    concentration still satisfies ``α ≥ 1``.

    Parameters
    ----------
    alpha_overlap_1, alpha_overlap_2
        Aligned concentration tensors of shape ``(N_overlap, C+1)``.

    Returns
    -------
    Tensor
        Fused concentrations of shape ``(N_overlap, C+1)``.
    """
    if alpha_overlap_1.shape != alpha_overlap_2.shape:
        raise ValueError("shape mismatch in fuse_overlap")
    fused = alpha_overlap_1 + alpha_overlap_2 - 1.0
    # Defensive clamp: rounding errors must not violate α ≥ 1.
    return torch.clamp(fused, min=1.0)
