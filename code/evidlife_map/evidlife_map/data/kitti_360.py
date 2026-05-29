"""KITTI-360 multi-session loader (RQ4 lifelong).

The W3 deliverable is the *revisit-pair overlap matrix*; this loader will
consume that JSON once the W3 pre-compute is done.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

KITTI360_NUM_CLASSES = 19


@dataclass(slots=True)
class RevisitPair:
    """One (session A, session B) revisit overlap."""

    sequence_a: str
    sequence_b: str
    frame_a: int
    frame_b: int
    overlap_m: float


@dataclass(slots=True)
class DatasetEntry:
    points_xyz_m: Tensor
    labels: Tensor
    sequence: str
    frame_idx: int


class KITTI360Dataset(Dataset[DatasetEntry]):
    """KITTI-360 wrapper (P1 skeleton).

    Parameters
    ----------
    root
        KITTI-360 dataset root.
    sequences
        Sequence directory names (paper uses 8 sequences).
    revisit_pair_list_path
        Optional JSON file describing the revisit pairs (W3 deliverable). When
        ``None``, the loader still works for single-session iteration.
    synthetic
        If ``True``, generate random clouds.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sequences: list[str],
        revisit_pair_list_path: str | Path | None = None,
        synthetic: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sequences = list(sequences)
        self.synthetic = bool(synthetic)
        self.revisit_pairs: list[RevisitPair] = self._load_revisit_pairs(revisit_pair_list_path)
        self._frame_index: list[tuple[str, int]] = [
            (seq, i) for seq in self.sequences for i in range(8 if synthetic else 0)
        ]

    def _load_revisit_pairs(self, path: str | Path | None) -> list[RevisitPair]:
        if path is None:
            return []
        with Path(path).open(encoding="utf-8") as fh:
            raw = json.load(fh)
        return [RevisitPair(**pair) for pair in raw]

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> DatasetEntry:
        sequence, frame_idx = self._frame_index[idx]
        if self.synthetic:
            generator = torch.Generator().manual_seed(hash((sequence, frame_idx)) & 0xFFFFFFFF)
            n = 2000
            return DatasetEntry(
                points_xyz_m=torch.randn(n, 3, generator=generator) * 20.0,
                labels=torch.randint(0, KITTI360_NUM_CLASSES, (n,), generator=generator),
                sequence=sequence,
                frame_idx=frame_idx,
            )
        raise NotImplementedError(
            "Real KITTI-360 loader lives in the P5 lifelong phase"
        )
