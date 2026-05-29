"""SemanticSpray loader (RQ3 traversability).

SemanticSpray ships per-point labels for wet-road / spray conditions; the W21
RQ3 passive replay uses them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

SEMANTIC_SPRAY_NUM_CLASSES = 4  # background / object / spray / ground


@dataclass(slots=True)
class DatasetEntry:
    points_xyz_m: Tensor
    labels: Tensor
    traversability_gt: Tensor   # (N,) int64 in TraversabilityLabel values
    sequence: str
    frame_idx: int


class SemanticSprayDataset(Dataset[DatasetEntry]):
    """SemanticSpray wrapper (P1 skeleton).

    Parameters
    ----------
    root
        Path to the SemanticSpray dataset root.
    synthetic
        If ``True``, generate random clouds.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        synthetic: bool = True,
        n_synthetic_frames: int = 8,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.synthetic = bool(synthetic)
        self._frame_index = list(range(n_synthetic_frames)) if synthetic else []

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> DatasetEntry:
        if not self.synthetic:
            raise NotImplementedError("Real SemanticSpray loader lives in P5 W21 RQ3 pass")
        generator = torch.Generator().manual_seed(idx)
        n = 1200
        labels = torch.randint(0, SEMANTIC_SPRAY_NUM_CLASSES, (n,), generator=generator)
        # crude traversability GT: ground=2 ⇒ traversable, spray=2 ⇒ defer.
        trav_gt = torch.where(
            labels == 3, torch.full_like(labels, 2),
            torch.where(labels == 2, torch.full_like(labels, 1), torch.zeros_like(labels)),
        )
        return DatasetEntry(
            points_xyz_m=torch.randn(n, 3, generator=generator) * 10.0,
            labels=labels,
            traversability_gt=trav_gt,
            sequence="synthetic",
            frame_idx=idx,
        )
