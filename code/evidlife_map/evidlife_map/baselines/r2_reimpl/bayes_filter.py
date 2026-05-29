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

    Two storage modes:

    * **Dict (legacy)** – sparse dict keyed by arbitrary int voxel hashes;
      kept for the W1 reference implementation and small unit tests.
    * **Flat tensors (W2)** – two sorted tensors ``_sorted_keys: (M,)`` and
      ``_log_post: (M, C)`` on ``device``. All hot-path operations
      (update_batch, argmax_batch) become O((M+N) log(M+N)) tensor ops with
      zero Python iteration over M. The dict mode is selected by default
      to preserve existing behaviour; pass ``backend="flat"`` to opt in.

    Parameters
    ----------
    num_classes
        Categorical support size (paper R2 uses C = 19 for SemKITTI).
    device, dtype
        Storage.
    backend
        ``"dict"`` (default) or ``"flat"``. The flat backend is ~10-100×
        faster on 100k-point frames at the cost of a single merge sort per
        batch.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: str = "dict",
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2")
        if backend not in ("dict", "flat"):
            raise ValueError(f"backend must be 'dict' or 'flat'; got {backend!r}")
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = backend
        self._uniform = torch.full(
            (num_classes,), -float(torch.log(torch.tensor(float(num_classes)))),
            device=self.device, dtype=dtype,
        )
        # Dict backend (legacy)
        self._log_post: dict[int, Tensor] = {}
        self._lock = RLock()
        # Flat backend
        self._sorted_keys: Tensor = torch.empty(0, dtype=torch.int64, device=self.device)
        self._flat_log_post: Tensor = torch.empty(0, num_classes, device=self.device, dtype=dtype)

    def __len__(self) -> int:
        if self.backend == "flat":
            return int(self._sorted_keys.shape[0])
        return len(self._log_post)

    def get_posterior(self, voxel_key: int) -> Tensor:
        if self.backend == "flat":
            return self._flat_get_posterior(int(voxel_key))
        with self._lock:
            existing = self._log_post.get(voxel_key)
            return existing.clone() if existing is not None else self._uniform.clone()

    def _flat_get_posterior(self, k: int) -> Tensor:
        if self._sorted_keys.numel() == 0:
            return self._uniform.clone()
        kt = torch.tensor([k], dtype=torch.int64, device=self.device)
        pos = torch.searchsorted(self._sorted_keys, kt)[0]
        n = int(self._sorted_keys.shape[0])
        if pos < n and int(self._sorted_keys[pos].item()) == k:
            return self._flat_log_post[pos].clone()
        return self._uniform.clone()

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

        log_likes = per_point_log_likelihoods.to(self.device, self.dtype)
        keys_dev = voxel_keys.to(self.device)
        if self.backend == "flat":
            self._flat_update_batch(keys_dev, log_likes)
            return

        # Collapse duplicates: unique keys + inverse mapping (each point -> bin idx).
        unique_keys, inverse = torch.unique(keys_dev, return_inverse=True)
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

    def _flat_update_batch(self, keys_dev: Tensor, log_likes: Tensor) -> None:
        """Pure-GPU batched update — no Python dict, no CPU sync.

        Algorithm
        ---------
        1. Compute per-batch aggregation: ``unique_new`` + ``agg``.
        2. ``full = torch.cat([sorted_keys, unique_new])``; ``full_unique``
           is the sorted union of old+new.
        3. Scatter old log-posteriors and new agg into ``full_unique``-aligned
           rows.
        4. Renormalise the rows touched by ``agg`` (cheap: all rows, batched).
        5. Replace state.
        """
        # 1. per-batch unique + agg
        unique_new, inv_new = torch.unique(keys_dev, return_inverse=True)
        n_new = int(unique_new.shape[0])
        agg = torch.zeros(n_new, self.num_classes, device=self.device, dtype=self.dtype)
        agg.scatter_add_(0, inv_new.unsqueeze(-1).expand(-1, self.num_classes), log_likes)

        # 2. merged unique sorted keys
        m_old = int(self._sorted_keys.shape[0])
        full = torch.cat([self._sorted_keys, unique_new])
        full_unique, full_inv = torch.unique(full, return_inverse=True, sorted=True)
        m_total = int(full_unique.shape[0])

        # 3. allocate new posterior table, broadcast uniform prior
        new_post = self._uniform.unsqueeze(0).expand(m_total, self.num_classes).clone()
        if m_old > 0:
            # full_inv[:m_old] maps each old key to its position in full_unique
            old_pos = full_inv[:m_old].unsqueeze(-1).expand(-1, self.num_classes)
            new_post.scatter_(0, old_pos, self._flat_log_post)
        # full_inv[m_old:] maps each unique_new to its position in full_unique
        new_pos = full_inv[m_old:]
        new_post.index_add_(0, new_pos, agg)

        # 4. normalise (logsumexp across class dim)
        new_post = new_post - torch.logsumexp(new_post, dim=-1, keepdim=True)

        # 5. replace state
        self._sorted_keys = full_unique
        self._flat_log_post = new_post

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
        if self.backend == "flat":
            return self._flat_argmax_batch(voxel_keys.to(self.device)).cpu()
        out = torch.empty(voxel_keys.shape[0], dtype=torch.int64, device="cpu")
        keys_cpu = voxel_keys.cpu().tolist()
        with self._lock:
            for i, k in enumerate(keys_cpu):
                post = self._log_post.get(k)
                out[i] = int(post.argmax().item()) if post is not None else int(self._uniform.argmax().item())
        return out

    def _flat_argmax_batch(self, query: Tensor) -> Tensor:
        """Pure-GPU per-key argmax via searchsorted (no Python iteration)."""
        n_sorted = int(self._sorted_keys.shape[0])
        u_argmax = int(self._uniform.argmax().item())
        if n_sorted == 0:
            return torch.full((query.shape[0],), u_argmax, dtype=torch.int64, device=self.device)
        pos = torch.searchsorted(self._sorted_keys, query)
        pos_clamped = pos.clamp(max=n_sorted - 1)
        # Match: pos < n_sorted AND sorted_keys[pos] == query
        match = (pos < n_sorted) & (self._sorted_keys[pos_clamped] == query)
        # Gather posterior rows at clamped positions
        rows = self._flat_log_post[pos_clamped]   # (N, C)
        # Argmax per row
        pred = rows.argmax(dim=-1)                # (N,)
        # For non-matches, use uniform argmax (broadcasts)
        return torch.where(match, pred,
                           torch.full_like(pred, u_argmax))
