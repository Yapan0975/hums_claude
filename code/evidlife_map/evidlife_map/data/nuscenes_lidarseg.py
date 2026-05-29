"""nuScenes-LiDARSeg loader (RQ1 cross-domain + RQ2 reverse-OOD).

Bridges to the nuscenes-devkit; the actual ``.bin`` + ``lidarseg`` parsing
will be filled in W4 when cross-domain numbers are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

NUSCENES_NUM_CLASSES = 16

# OOD classes (research_plan v3 §4.1 RQ2 reverse cross-domain).
NUSCENES_OOD_CLASS_NAMES: tuple[str, ...] = (
    "construction_vehicle",
    "barrier",
    "traffic_cone",
    "pushable_pullable",
)


@dataclass(slots=True)
class DatasetEntry:
    points_xyz_m: Tensor
    intensities: Tensor
    labels: Tensor
    sample_token: str


class NuScenesLidarSegDataset(Dataset[DatasetEntry]):
    """nuScenes-LiDARSeg wrapper (P1 skeleton).

    Parameters
    ----------
    root
        nuScenes dataset root (containing ``samples/``, ``sweeps/``, ``v1.0-trainval/``).
    version
        nuScenes split version (e.g. ``"v1.0-trainval"``).
    sample_tokens
        Optional list of sample tokens to load; if ``None``, the test split is
        used.
    synthetic
        If ``True``, generate random clouds so the test suite can run without
        downloading 400 GB.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        version: str = "v1.0-trainval",
        sample_tokens: list[str] | None = None,
        synthetic: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.version = version
        self.synthetic = bool(synthetic)
        self.sample_tokens = sample_tokens or [f"synth_{i:04d}" for i in range(8)]
        self._nusc = None  # populated lazily when synthetic=False

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> DatasetEntry:
        token = self.sample_tokens[idx]
        if self.synthetic:
            generator = torch.Generator().manual_seed(hash(token) & 0xFFFFFFFF)
            n = 1500
            return DatasetEntry(
                points_xyz_m=torch.randn(n, 3, generator=generator) * 15.0,
                intensities=torch.rand(n, generator=generator),
                labels=torch.randint(0, NUSCENES_NUM_CLASSES, (n,), generator=generator),
                sample_token=token,
            )
        raise NotImplementedError(
            "Real nuScenes loader lives in the W4 cross-domain pass; "
            "instantiate with synthetic=True until then"
        )
