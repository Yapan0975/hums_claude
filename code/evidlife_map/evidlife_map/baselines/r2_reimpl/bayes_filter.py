"""Argmax-Bayes per-voxel update — the R2 [jiao2024r2] baseline.

This is the simplest credible interpretation of the R2 "iterative per-voxel
Bayes filter" — Eq 4 of paper_v1.md notes that the EvidLife-Map evidential
update recovers this as a degenerate limit (one-hot evidence, uniform prior).

We make the implementation explicit here so the §IV RQ1 ablation A-1
(``± Dirichlet evidential head``) compares apples to apples.
"""

from __future__ import annotations

from threading import RLock

import torch
from torch import Tensor


def argmax_bayes_update(
    log_posterior: Tensor,
    per_point_log_likelihood: Tensor,
) -> Tensor:
    """Bayes' rule in log-space, normalised to a proper categorical.

    Parameters
    ----------
    log_posterior
        Per-voxel log posterior over ``C`` classes, shape ``(..., C)``.
    per_point_log_likelihood
        Per-point log likelihood, shape ``(..., C)``.

    Returns
    -------
    Tensor
        Updated log posterior, shape ``(..., C)``.
    """
    if log_posterior.shape != per_point_log_likelihood.shape:
        raise ValueError("posterior and likelihood shapes must match")
    unnorm = log_posterior + per_point_log_likelihood
    return unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)


class ArgmaxBayesAccumulator:
    """Per-voxel categorical-Bayes accumulator (R2-style).

    Parameters
    ----------
    num_classes
        Categorical support size (paper R2 uses C = 19 for SemKITTI).
    device, dtype
        Storage.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2")
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.dtype = dtype
        self._uniform = torch.full(
            (num_classes,), -float(torch.log(torch.tensor(float(num_classes)))),
            device=self.device, dtype=dtype,
        )
        self._log_post: dict[int, Tensor] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        return len(self._log_post)

    def get_posterior(self, voxel_key: int) -> Tensor:
        with self._lock:
            existing = self._log_post.get(voxel_key)
            return existing.clone() if existing is not None else self._uniform.clone()

    def update(self, voxel_key: int, log_likelihood: Tensor) -> Tensor:
        if log_likelihood.shape != (self.num_classes,):
            raise ValueError(
                f"log_likelihood must have shape ({self.num_classes},); "
                f"got {tuple(log_likelihood.shape)}"
            )
        with self._lock:
            prior = self._log_post.get(voxel_key)
            if prior is None:
                prior = self._uniform.clone()
            new = argmax_bayes_update(prior, log_likelihood.to(self.device, self.dtype))
            self._log_post[voxel_key] = new
        return new.clone()

    def update_batch(
        self,
        voxel_keys: Tensor,
        per_point_log_likelihoods: Tensor,
    ) -> None:
        """Vectorised batch update over many points sharing voxel keys.

        Equivalent to::

            for k, ll in zip(voxel_keys, per_point_log_likelihoods):
                self.update(k, ll)

        but ~100× faster on 100k-point frames by collapsing duplicate keys
        with ``torch.unique`` + ``scatter_add``.

        Parameters
        ----------
        voxel_keys
            Integer voxel hashes of shape ``(N,)``.
        per_point_log_likelihoods
            Per-point log-likelihoods of shape ``(N, C)``.
        """
        if voxel_keys.dim() != 1 or per_point_log_likelihoods.dim() != 2:
            raise ValueError(
                f"shapes must be (N,) + (N, C); got {tuple(voxel_keys.shape)} + "
                f"{tuple(per_point_log_likelihoods.shape)}"
            )
        if voxel_keys.shape[0] != per_point_log_likelihoods.shape[0]:
            raise ValueError("N mismatch between voxel_keys and log_likelihoods")
        if per_point_log_likelihoods.shape[1] != self.num_classes:
            raise ValueError(
                f"log_likelihoods last dim {per_point_log_likelihoods.shape[1]} "
                f"!= num_classes {self.num_classes}"
            )

        # Collapse duplicates: unique keys + inverse mapping (each point -> bin idx).
        log_likes = per_point_log_likelihoods.to(self.device, self.dtype)
        unique_keys, inverse = torch.unique(voxel_keys.to(self.device), return_inverse=True)
        n_unique = int(unique_keys.shape[0])

        # Aggregate: per-bin sum of log-likelihoods (Bayes prior += sum log_likes).
        agg = torch.zeros(n_unique, self.num_classes, device=self.device, dtype=self.dtype)
        agg.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, self.num_classes), log_likes)

        # Gather existing log-posteriors into a single (n_unique, C) tensor;
        # missing keys get the uniform prior. One dict scan + one batched
        # normalisation replaces n_unique × logsumexp + dict writes.
        keys_cpu = unique_keys.cpu().tolist()
        priors = torch.empty(n_unique, self.num_classes, device=self.device, dtype=self.dtype)
        with self._lock:
            for i, k in enumerate(keys_cpu):
                existing = self._log_post.get(k)
                priors[i] = existing if existing is not None else self._uniform
            # Batched normalisation across all unique voxels at once.
            unnorm = priors + agg
            new_post = unnorm - torch.logsumexp(unnorm, dim=-1, keepdim=True)
            # Write back. dict[int]=Tensor row is ~1 µs each.
            for i, k in enumerate(keys_cpu):
                self._log_post[k] = new_post[i]

    def argmax(self, voxel_key: int) -> int:
        return int(self.get_posterior(voxel_key).argmax().item())

    def argmax_batch(self, voxel_keys: Tensor) -> Tensor:
        """Vectorised per-voxel argmax over many keys.

        Returns
        -------
        Tensor
            Argmax-class labels, shape ``(N,)`` int64.
        """
        if voxel_keys.dim() != 1:
            raise ValueError("voxel_keys must be 1-D")
        out = torch.empty(voxel_keys.shape[0], dtype=torch.int64, device="cpu")
        keys_cpu = voxel_keys.cpu().tolist()
        with self._lock:
            for i, k in enumerate(keys_cpu):
                post = self._log_post.get(k)
                out[i] = int(post.argmax().item()) if post is not None else int(self._uniform.argmax().item())
        return out
