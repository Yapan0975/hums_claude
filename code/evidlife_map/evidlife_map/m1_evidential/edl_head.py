"""Dirichlet evidential head (paper §III.B, Eq 1–3).

Implements the evidential-learning convention of [Sensoy 2018]:

* The semantic backbone produces a logits vector ``z ∈ R^{C+1}``.
* A non-negative activation (softplus by default; ReLU and exp also supported)
  maps logits to evidence ``e = phi(z) ≥ 0``.
* Concentrations are ``α = e + 1`` (Eq 1) so that the uninformed posterior is
  exactly the uniform Dirichlet ``Dir(1)``.
* Vacuity ``u_v = (C+1) / S_v`` (Eq 2) is the load-bearing scalar reused by
  M2 (loop closure entropy channel) and M3 (decay clock).
* Posterior class mean ``E[p_k|α] = α_k / S`` (Eq 3) gives the expected
  probability used by mIoU / ECE / argmax inference.

The head is intentionally last-layer-only: the Cylinder3D backbone is frozen in
the P2 fine-tune (research_plan v3 R-11 mitigation), so this module accepts an
input feature tensor of shape ``(B, F, N_points)`` and returns ``(α, u_v)``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

EvidenceActivation = Literal["softplus", "relu", "exp"]


def evidence_to_alpha(evidence: Tensor) -> Tensor:
    """Implement Eq 1: ``α_k = e_k + 1``.

    Parameters
    ----------
    evidence
        Non-negative evidence tensor of arbitrary shape; the last dimension is
        the class axis.

    Returns
    -------
    Tensor
        Concentration tensor of the same shape with values ``≥ 1``.
    """
    if torch.any(evidence < 0):
        raise ValueError("evidence must be non-negative for α = e + 1 (paper Eq 1)")
    return evidence + 1.0


def vacuity_from_alpha(alpha: Tensor) -> Tensor:
    """Implement Eq 2: ``u_v = (C+1) / S_v``.

    Parameters
    ----------
    alpha
        Dirichlet concentration tensor of shape ``(..., C+1)``.

    Returns
    -------
    Tensor
        Vacuity tensor of shape ``(...)`` with values in ``(0, 1]``.
    """
    num_classes_plus_one = alpha.shape[-1]
    strength = alpha.sum(dim=-1)
    return torch.as_tensor(num_classes_plus_one, dtype=alpha.dtype, device=alpha.device) / strength


def dirichlet_mean(alpha: Tensor) -> Tensor:
    """Implement Eq 3: ``E[p_k|α] = α_k / S``.

    Returns
    -------
    Tensor
        Expected class probabilities of the same shape as ``alpha``.
    """
    strength = alpha.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
    return alpha / strength


def _activate(z: Tensor, activation: EvidenceActivation) -> Tensor:
    if activation == "softplus":
        return torch.nn.functional.softplus(z)
    if activation == "relu":
        return torch.nn.functional.relu(z)
    if activation == "exp":
        return torch.exp(z.clamp(max=20.0))
    raise ValueError(f"unknown activation: {activation!r}")


class EDLHead(nn.Module):
    """Per-point evidential head wrapping a single linear layer.

    The output is the **tuple** ``(α, u_v)``; downstream code splits them
    appropriately so M2 and M3 can reuse the vacuity without recomputation
    (paper §III.E).

    Parameters
    ----------
    in_features
        Feature dimension produced by the upstream backbone (e.g. 128 for
        Cylinder3D's last hidden layer).
    num_classes
        Number of closed-set classes (``C``). The actual output dimension is
        ``C + 1`` because the explicit "unknown" channel is appended.
    activation
        Non-negative evidence activation. ``softplus`` is the EDL default
        recommended by Sensoy 2018; ``relu`` is the historical choice; ``exp``
        is provided for numerical comparison only.
    evidence_max
        Optional per-class evidence clip; bounds ``S_v ≤ S_max`` so that
        vacuity stays numerically stable across long sessions
        (paper §III.B implementation paragraph).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        activation: EvidenceActivation = "softplus",
        evidence_max: float | None = 50.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 (the C in C+1 must be ≥ 2)")
        self.num_classes = num_classes
        self.activation: EvidenceActivation = activation
        self.evidence_max = evidence_max
        # +1 = the explicit "unknown" channel (paper §III.A).
        self.linear = nn.Linear(in_features, num_classes + 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Compute concentration and vacuity from backbone features.

        Parameters
        ----------
        features
            Backbone feature tensor of shape ``(..., in_features)``.

        Returns
        -------
        alpha
            Concentration tensor of shape ``(..., C+1)`` with values ``≥ 1``.
        vacuity
            Vacuity tensor of shape ``(...)`` with values in ``(0, 1]``.
        """
        logits = self.linear(features)
        evidence = _activate(logits, self.activation)
        if self.evidence_max is not None:
            evidence = evidence.clamp(max=self.evidence_max)
        alpha = evidence_to_alpha(evidence)
        vacuity = vacuity_from_alpha(alpha)
        return alpha, vacuity

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, activation={self.activation!r}, "
            f"evidence_max={self.evidence_max}"
        )
