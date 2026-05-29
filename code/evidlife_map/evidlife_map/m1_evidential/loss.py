"""Evidential Deep Learning loss (paper §III.B Eq 5).

Implements the [Sensoy 2018] objective:

    L_EDL(α) =  Σ_k [ (y_k − p̂_k)² + p̂_k(1−p̂_k)/(S+1) ]
              + λ_t · KL( Dir(α̃) || Dir(1) )

where ``p̂_k = α_k / S``, ``α̃ = y + (1−y) ⊙ α`` masks the ground-truth class
out of the regulariser, ``λ_t`` is an annealing weight that grows over training
epochs, and ``Dir(1)`` is the uniform Dirichlet prior over ``C+1`` classes.

The KL term is the **closed-form** Dirichlet KL to the uniform Dirichlet (cf.
Sensoy 2018 Appendix A):

    KL( Dir(α) || Dir(1) ) =
        log Γ(S) − log Γ(C+1) − Σ_k log Γ(α_k)
        + Σ_k (α_k − 1) · ( ψ(α_k) − ψ(S) ) .

Numerically stable form using torch.lgamma + torch.digamma is used below.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _kl_dirichlet_to_uniform(alpha_tilde: Tensor, eps: float = 1e-8) -> Tensor:
    """Closed-form KL( Dir(α̃) || Dir(1) ).

    Parameters
    ----------
    alpha_tilde
        Masked concentration tensor of shape ``(..., C+1)``; the ground-truth
        class is masked to 1 so it does not contribute to the regulariser.
    eps
        Numerical floor to avoid ``digamma(0)`` and ``lgamma(0)`` blow-ups.

    Returns
    -------
    Tensor
        KL divergence per sample, shape ``(...)``.
    """
    alpha = alpha_tilde.clamp_min(1.0 + eps)
    k_plus_one = alpha.shape[-1]
    strength = alpha.sum(dim=-1)
    log_gamma_strength = torch.lgamma(strength)
    log_gamma_kp1 = torch.lgamma(
        torch.tensor(float(k_plus_one), dtype=alpha.dtype, device=alpha.device)
    )
    sum_log_gamma_alpha = torch.lgamma(alpha).sum(dim=-1)
    digamma_alpha = torch.digamma(alpha)
    digamma_strength = torch.digamma(strength).unsqueeze(-1)
    extra = ((alpha - 1.0) * (digamma_alpha - digamma_strength)).sum(dim=-1)
    return log_gamma_strength - log_gamma_kp1 - sum_log_gamma_alpha + extra


class EDLLoss(nn.Module):
    """Evidential MSE + Dirichlet KL prior regulariser (paper Eq 5).

    Parameters
    ----------
    num_classes_plus_one
        Output dimension of the EDL head (``C + 1``).
    kl_anneal_epochs
        Number of epochs over which ``λ_t`` ramps linearly from
        ``kl_lambda_start`` to ``kl_lambda_end``.
    kl_lambda_start, kl_lambda_end
        Start / end values of the annealing weight ``λ_t`` (paper Eq 5).
    ignore_index
        Per-point label value to skip (e.g. SemanticKITTI "outlier" = 0 after
        the standard remap). Set to ``-100`` to disable.
    """

    def __init__(
        self,
        num_classes_plus_one: int,
        *,
        kl_anneal_epochs: int = 10,
        kl_lambda_start: float = 0.0,
        kl_lambda_end: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if num_classes_plus_one < 2:
            raise ValueError("num_classes_plus_one must be ≥ 2")
        self.num_classes_plus_one = num_classes_plus_one
        self.kl_anneal_epochs = max(1, int(kl_anneal_epochs))
        self.kl_lambda_start = float(kl_lambda_start)
        self.kl_lambda_end = float(kl_lambda_end)
        self.ignore_index = int(ignore_index)

    def _current_lambda(self, epoch: int) -> float:
        if epoch >= self.kl_anneal_epochs:
            return self.kl_lambda_end
        frac = float(epoch) / float(self.kl_anneal_epochs)
        return self.kl_lambda_start + frac * (self.kl_lambda_end - self.kl_lambda_start)

    def forward(
        self,
        alpha: Tensor,
        target: Tensor,
        *,
        epoch: int = 0,
    ) -> dict[str, Tensor]:
        """Compute the EDL loss.

        Parameters
        ----------
        alpha
            Per-point Dirichlet concentrations from :class:`EDLHead`, shape
            ``(N, C+1)`` with values ``≥ 1``.
        target
            Per-point integer class labels, shape ``(N,)``. The ``ignore_index``
            value is masked out.
        epoch
            Current training epoch; controls the KL annealing weight ``λ_t``.

        Returns
        -------
        dict[str, Tensor]
            ``"loss"`` is the scalar total; ``"mse"``, ``"kl"``, ``"lambda"``
            are the components for logging.
        """
        if alpha.dim() != 2:
            raise ValueError(f"alpha must be (N, C+1); got {tuple(alpha.shape)}")
        if alpha.shape[-1] != self.num_classes_plus_one:
            raise ValueError(
                f"alpha last dim {alpha.shape[-1]} != num_classes_plus_one "
                f"{self.num_classes_plus_one}"
            )

        valid = target != self.ignore_index
        if not bool(valid.any()):
            zero = alpha.new_zeros(())
            return {"loss": zero, "mse": zero, "kl": zero, "lambda": zero}

        alpha_v = alpha[valid]
        target_v = target[valid]
        strength = alpha_v.sum(dim=-1, keepdim=True)
        p_hat = alpha_v / strength.clamp_min(1e-12)

        y = torch.nn.functional.one_hot(target_v, num_classes=self.num_classes_plus_one)
        y = y.to(p_hat.dtype)

        # MSE term and variance term (Sensoy 2018 Eq 5 expansion).
        mse_term = ((y - p_hat) ** 2).sum(dim=-1)
        var_term = ((p_hat * (1.0 - p_hat)) / (strength + 1.0)).sum(dim=-1)

        # Mask ground-truth class out of α to form α̃.
        alpha_tilde = y + (1.0 - y) * alpha_v
        kl_term = _kl_dirichlet_to_uniform(alpha_tilde)

        lam = self._current_lambda(epoch)
        mse_mean = (mse_term + var_term).mean()
        kl_mean = kl_term.mean()
        total = mse_mean + lam * kl_mean

        return {
            "loss": total,
            "mse": mse_mean.detach(),
            "kl": kl_mean.detach(),
            "lambda": alpha.new_tensor(lam),
        }
