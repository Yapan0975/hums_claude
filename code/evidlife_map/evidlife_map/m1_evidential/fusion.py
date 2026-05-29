"""Per-voxel evidence accumulator (paper §III.B Eq 4).

Implements

    α_v^{(t+1)} = α_v^{(t)} + e_v^{(t+1)}

as a dictionary-of-tensors keyed by integer voxel keys. The invariant
``α_v ≥ 1`` must hold after every update so that the vacuity ``u_v = (C+1) / S_v``
in Eq 2 stays well-defined.

The accumulator is intentionally lock-free for single-threaded use; for the
multi-stream Jetson deployment in :mod:`evidlife_map.jetson` we use a per-block
mutex around :meth:`accumulate_batch` instead of per-voxel locking, because the
nvblox layer cake already groups updates by ``8³`` blocks (paper §III.F).
"""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import vacuity_from_alpha
from evidlife_map.m1_evidential.prior import dirichlet_prior


class EvidenceAccumulator:
    """Per-voxel Dirichlet evidence accumulator.

    Parameters
    ----------
    num_classes_plus_one
        Dimension of ``α_v``.
    device
        Storage device. CPU is the safe default; M1 forward integration on
        Jetson uses ``cuda:0``.
    dtype
        Concentration tensor dtype. ``float32`` matches the nvblox-side
        ``EvidentialVoxel`` (44 kB / block at C=20).
    thread_safe
        If ``True``, a single :class:`threading.RLock` guards every accessor
        — sufficient for the multi-camera demo capture pipeline.
    """

    def __init__(
        self,
        num_classes_plus_one: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        thread_safe: bool = True,
        backend: str = "dict",
    ) -> None:
        if num_classes_plus_one < 2:
            raise ValueError("num_classes_plus_one must be ≥ 2")
        if backend not in ("dict", "flat"):
            raise ValueError(f"backend must be 'dict' or 'flat'; got {backend!r}")
        self.num_classes_plus_one = int(num_classes_plus_one)
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = backend
        self._lock: RLock | None = RLock() if thread_safe else None
        # Dict backend (legacy)
        self._alpha: dict[int, Tensor] = {}
        self._timestamps: dict[int, float] = {}
        self._prior = dirichlet_prior(self.num_classes_plus_one, device=self.device, dtype=dtype)
        # Flat backend (W2-3: pure-GPU, ~100x faster on 100k-point frames)
        self._sorted_keys: Tensor = torch.empty(0, dtype=torch.int64, device=self.device)
        self._flat_alpha: Tensor = torch.empty(0, num_classes_plus_one, device=self.device, dtype=dtype)
        self._flat_ts: Tensor = torch.empty(0, dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------ utils

    def __len__(self) -> int:
        if self.backend == "flat":
            return int(self._sorted_keys.shape[0])
        return len(self._alpha)

    def __contains__(self, voxel_key: int) -> bool:
        if self.backend == "flat":
            if self._sorted_keys.numel() == 0:
                return False
            kt = torch.tensor([int(voxel_key)], dtype=torch.int64, device=self.device)
            pos = torch.searchsorted(self._sorted_keys, kt)[0]
            n = int(self._sorted_keys.shape[0])
            return bool(pos < n and int(self._sorted_keys[pos].item()) == int(voxel_key))
        return voxel_key in self._alpha

    def _maybe_lock(self) -> RLock | _NullCtx:
        return self._lock if self._lock is not None else _NullCtx()

    # ----------------------------------------------------------------- access

    def get_alpha(self, voxel_key: int) -> Tensor:
        """Return ``α_v`` for the given key; uniform prior if unseen."""
        if self.backend == "flat":
            if self._sorted_keys.numel() == 0:
                return self._prior.clone()
            kt = torch.tensor([int(voxel_key)], dtype=torch.int64, device=self.device)
            pos = torch.searchsorted(self._sorted_keys, kt)[0]
            n = int(self._sorted_keys.shape[0])
            if pos < n and int(self._sorted_keys[pos].item()) == int(voxel_key):
                return self._flat_alpha[pos].clone()
            return self._prior.clone()
        with self._maybe_lock():
            existing = self._alpha.get(voxel_key)
            return existing.clone() if existing is not None else self._prior.clone()

    def get_vacuity(self, voxel_key: int) -> float:
        """Return ``u_v`` (Eq 2) for the given key; ``1.0`` if unseen."""
        alpha = self.get_alpha(voxel_key)
        return float(vacuity_from_alpha(alpha).item())

    # ------------------------------------------------------------- accumulate

    def accumulate(
        self,
        voxel_key: int,
        evidence: Tensor,
        *,
        timestamp: float,
    ) -> Tensor:
        """Accumulate a single evidence vector into ``α_v`` (Eq 4).

        Parameters
        ----------
        voxel_key
            Integer hash of the voxel coordinate.
        evidence
            Non-negative evidence vector of shape ``(C+1,)``.
        timestamp
            Wall-clock time of the observation; consumed by M3's age tracker.

        Returns
        -------
        Tensor
            The updated ``α_v``.
        """
        self._validate_evidence(evidence)
        with self._maybe_lock():
            current = self._alpha.get(voxel_key)
            if current is None:
                current = self._prior.clone()
            updated = current + evidence.to(device=self.device, dtype=self.dtype)
            # Invariant: α ≥ 1 ⇔ S ≥ C+1 ⇔ vacuity ≤ 1.
            # Adding non-negative evidence to a tensor ≥ 1 preserves the bound.
            self._alpha[voxel_key] = updated
            self._timestamps[voxel_key] = float(timestamp)
        return updated.clone()

    def accumulate_batch(
        self,
        voxel_keys: Iterable[int],
        evidence_batch: Tensor,
        *,
        timestamp: float,
    ) -> None:
        """Vectorised batch update.

        Parameters
        ----------
        voxel_keys
            Iterable of ``N`` integer voxel keys.
        evidence_batch
            Non-negative evidence tensor of shape ``(N, C+1)``.
        timestamp
            Common timestamp for the batch.
        """
        keys = list(voxel_keys)
        if evidence_batch.dim() != 2 or evidence_batch.shape[0] != len(keys):
            raise ValueError(
                f"evidence_batch must be (N={len(keys)}, C+1); "
                f"got {tuple(evidence_batch.shape)}"
            )
        self._validate_evidence(evidence_batch)
        evidence_batch = evidence_batch.to(device=self.device, dtype=self.dtype)
        if self.backend == "flat":
            keys_t = torch.as_tensor(keys, dtype=torch.int64, device=self.device)
            self._flat_accumulate_batch(keys_t, evidence_batch, timestamp)
            return
        with self._maybe_lock():
            for key, evid in zip(keys, evidence_batch, strict=True):
                current = self._alpha.get(key)
                if current is None:
                    current = self._prior.clone()
                self._alpha[key] = current + evid
                self._timestamps[key] = float(timestamp)

    def _flat_accumulate_batch(self, keys_dev: Tensor, evidence: Tensor, timestamp: float) -> None:
        """Pure-GPU evidence accumulation. α ≥ 1 invariant preserved.

        Mirrors :meth:`ArgmaxBayesAccumulator._flat_update_batch`: per-batch
        unique + scatter_add, then merge with stored sorted state.
        """
        # 1. per-batch unique keys + aggregated evidence
        unique_new, inv_new = torch.unique(keys_dev, return_inverse=True)
        n_new = int(unique_new.shape[0])
        agg = torch.zeros(n_new, self.num_classes_plus_one, device=self.device, dtype=self.dtype)
        agg.scatter_add_(0, inv_new.unsqueeze(-1).expand(-1, self.num_classes_plus_one), evidence)

        # 2. merge with stored
        m_old = int(self._sorted_keys.shape[0])
        full = torch.cat([self._sorted_keys, unique_new])
        full_unique, full_inv = torch.unique(full, return_inverse=True, sorted=True)
        m_total = int(full_unique.shape[0])

        # 3. build new α table seeded with prior (α=1 baseline)
        new_alpha = self._prior.unsqueeze(0).expand(m_total, self.num_classes_plus_one).clone()
        if m_old > 0:
            old_pos = full_inv[:m_old].unsqueeze(-1).expand(-1, self.num_classes_plus_one)
            new_alpha.scatter_(0, old_pos, self._flat_alpha)
        # add new evidence to positions of unique_new
        new_pos = full_inv[m_old:]
        new_alpha.index_add_(0, new_pos, agg)

        # 4. timestamps: keep old for unchanged, set new for updated positions
        new_ts = torch.zeros(m_total, dtype=torch.float64, device=self.device)
        if m_old > 0:
            new_ts.scatter_(0, full_inv[:m_old], self._flat_ts)
        new_ts.index_fill_(0, new_pos, float(timestamp))

        # 5. replace state
        self._sorted_keys = full_unique
        self._flat_alpha = new_alpha
        self._flat_ts = new_ts

    # --------------------------------------------------------------- helpers

    def _validate_evidence(self, evidence: Tensor) -> None:
        if evidence.shape[-1] != self.num_classes_plus_one:
            raise ValueError(
                f"evidence last dim {evidence.shape[-1]} != "
                f"num_classes_plus_one {self.num_classes_plus_one}"
            )
        if torch.any(evidence < 0):
            raise ValueError("evidence must be non-negative (Eq 4)")

    # ---------------------------------------------------- decay-callback hook

    def overwrite(self, voxel_key: int, alpha: Tensor, *, timestamp: float) -> None:
        """Replace ``α_v`` in-place. Used by M3's decay kernel.

        Raises
        ------
        ValueError
            If ``alpha`` violates the ``α ≥ 1`` invariant.
        """
        if alpha.shape[-1] != self.num_classes_plus_one:
            raise ValueError("alpha last dim mismatch in overwrite()")
        if torch.any(alpha < 1.0 - 1e-6):
            raise ValueError("overwrite() must preserve α ≥ 1 (paper Eq 12)")
        with self._maybe_lock():
            self._alpha[voxel_key] = alpha.to(device=self.device, dtype=self.dtype)
            self._timestamps[voxel_key] = float(timestamp)

    def last_update(self, voxel_key: int) -> float | None:
        """Return last-update timestamp (M3 ``t_{v,last-update}``); ``None`` if unseen."""
        with self._maybe_lock():
            return self._timestamps.get(voxel_key)

    def keys(self) -> list[int]:
        if self.backend == "flat":
            return self._sorted_keys.cpu().tolist()
        with self._maybe_lock():
            return list(self._alpha.keys())


class _NullCtx:
    """No-op context manager used when ``thread_safe=False``."""

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, *_args: object) -> None:
        return None
