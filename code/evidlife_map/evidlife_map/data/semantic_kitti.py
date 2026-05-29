"""SemanticKITTI loader.

Closed-set RQ1 / RQ5 main dataset and substrate for the RQ2 14-known / 5-unknown
open-set split (research_plan v3 §4.1 D-a + §4.1 split protocol).

Real loader parses the binary ``.bin`` (xyzi float32, 4 floats / point) +
``.label`` (uint32, lower-16-bit = raw class) files in the SemanticKITTI
on-disk layout::

    root/sequences/<seq>/velodyne/<frame>.bin    # (N, 4) xyzi
    root/sequences/<seq>/labels/<frame>.label    # (N,)   uint32

The official 260-class raw label space is remapped to 19 learning classes via
``SEMANTIC_KITTI_LEARNING_MAP`` per the SemanticKITTI ``semantic-kitti.yaml``
config (vendored below — keeps the loader self-contained).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# SemanticKITTI 19-class set after the official remap (research_plan v3 §4.1).
SEMANTIC_KITTI_NUM_CLASSES = 19

# Provisional 5-class unknown split (research_plan v3 RQ2 primary):
# bicyclist, motorcyclist, truck, other-vehicle, other-ground.
SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY: tuple[int, ...] = (4, 5, 18, 17, 9)
SEMANTIC_KITTI_UNKNOWN_CLASSES_ROBUSTNESS: tuple[int, ...] = (4, 5, 18)

# Official SemanticKITTI raw -> learning class remap (semantic-kitti.yaml).
# 0 = ignore (mapped to -100 for torch CrossEntropyLoss ignore_index).
SEMANTIC_KITTI_LEARNING_MAP: dict[int, int] = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}
# After remap: 0 = unlabeled/ignore. 1..19 = the 19 learning classes.
# Convert to -100 ignore + zero-based [0..18] for torch.

_REMAP_LUT: np.ndarray | None = None


def _build_remap_lut() -> np.ndarray:
    """Build a 260-entry numpy LUT for raw-class -> learning-class remap.

    Output convention:
      - raw 0 / unmapped -> -100 (CrossEntropyLoss ignore_index)
      - raw mapped to 0 in YAML -> -100 (also ignore)
      - raw mapped to 1..19 -> 0..18 (zero-based learning class)
    """
    lut = np.full(260, -100, dtype=np.int64)
    for raw, learn in SEMANTIC_KITTI_LEARNING_MAP.items():
        if 0 <= raw < 260:
            lut[raw] = (learn - 1) if learn > 0 else -100
    return lut


def _get_remap_lut() -> np.ndarray:
    global _REMAP_LUT
    if _REMAP_LUT is None:
        _REMAP_LUT = _build_remap_lut()
    return _REMAP_LUT


@dataclass(slots=True)
class DatasetEntry:
    """One frame's payload as consumed by the M1 fine-tune loop."""

    points_xyz_m: Tensor             # (N, 3) world-frame metric points
    intensities: Tensor              # (N,) LiDAR intensities, normalised to [0, 1]
    labels: Tensor                   # (N,) int64 per-point class index (-100 = ignore)
    frame_idx: int
    sequence: str


class SemanticKITTIDataset(Dataset[DatasetEntry]):
    """SemanticKITTI dataset wrapper (P1 skeleton).

    Parameters
    ----------
    root
        Path to the SemanticKITTI dataset root.
    sequences
        Sequence directory names to include (e.g. ``["08"]`` for val).
    open_set_split
        If ``"primary"`` or ``"robustness"``, replace the labels of the
        configured "unknown" classes with the dedicated unknown index
        ``SEMANTIC_KITTI_NUM_CLASSES`` (i.e. the last channel of a 20-D head
        becomes the "unknown" channel).
    synthetic
        If ``True`` (the default for the P1 skeleton), generate random points
        on-the-fly so the loader is usable without an actual dataset on disk.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sequences: list[str],
        open_set_split: str | None = None,
        synthetic: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.sequences = list(sequences)
        if open_set_split not in (None, "primary", "robustness"):
            raise ValueError(
                f"open_set_split must be None | 'primary' | 'robustness'; got {open_set_split!r}"
            )
        self.open_set_split = open_set_split
        self.synthetic = bool(synthetic)
        self._frame_index: list[tuple[str, int]] = self._build_index()

    def _build_index(self) -> list[tuple[str, int]]:
        if self.synthetic:
            return [(seq, i) for seq in self.sequences for i in range(8)]
        index: list[tuple[str, int]] = []
        for seq in self.sequences:
            seq_dir = self.root / "sequences" / seq / "velodyne"
            if not seq_dir.exists():
                raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")
            frame_ids = sorted(int(p.stem) for p in seq_dir.glob("*.bin"))
            index.extend((seq, i) for i in frame_ids)
        return index

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> DatasetEntry:
        sequence, frame_idx = self._frame_index[idx]
        if self.synthetic:
            return self._synthetic_entry(sequence, frame_idx)
        return self._load_entry(sequence, frame_idx)

    # ---------------------------------------------------------- synthetic ---

    def _synthetic_entry(self, sequence: str, frame_idx: int) -> DatasetEntry:
        generator = torch.Generator().manual_seed(hash((sequence, frame_idx)) & 0xFFFFFFFF)
        n = 1000
        points = torch.randn(n, 3, generator=generator) * 10.0
        intens = torch.rand(n, generator=generator)
        labels = torch.randint(0, SEMANTIC_KITTI_NUM_CLASSES, (n,), generator=generator)
        labels = self._apply_open_set_remap(labels)
        return DatasetEntry(
            points_xyz_m=points,
            intensities=intens,
            labels=labels,
            frame_idx=frame_idx,
            sequence=sequence,
        )

    # --------------------------------------------------------- real loader ---

    def _load_entry(self, sequence: str, frame_idx: int) -> DatasetEntry:
        """Parse one frame's .bin (xyzi) + .label files from disk.

        SemanticKITTI on-disk layout::

            root/sequences/<seq>/velodyne/<frame:06d>.bin   # float32 (N, 4)
            root/sequences/<seq>/labels/<frame:06d>.label   # uint32  (N,)

        The lower 16 bits of each .label uint32 encode the semantic class id;
        the upper 16 bits encode the instance id (unused here).
        """
        seq_dir = self.root / "sequences" / sequence
        bin_path = seq_dir / "velodyne" / f"{frame_idx:06d}.bin"
        label_path = seq_dir / "labels" / f"{frame_idx:06d}.label"

        if not bin_path.exists():
            raise FileNotFoundError(f"Missing point cloud: {bin_path}")
        # Test sequences (11..21) have no labels -- return zeros as ignore.
        has_labels = label_path.exists()

        # .bin = float32 [x y z intensity x y z intensity ...]
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        n = pts.shape[0]
        if n == 0:
            raise ValueError(f"Empty point cloud at {bin_path}")
        xyz = pts[:, :3]
        intens = pts[:, 3]

        if has_labels:
            raw_labels = np.fromfile(label_path, dtype=np.uint32)
            if raw_labels.shape[0] != n:
                raise ValueError(
                    f"Point/label count mismatch in {sequence}/{frame_idx}: "
                    f"{n} points vs {raw_labels.shape[0]} labels"
                )
            sem_raw = (raw_labels & 0xFFFF).astype(np.int64)  # lower-16 = sem class
            lut = _get_remap_lut()
            # Guard against out-of-range raw labels (shouldn't happen on official data).
            sem_raw_safe = np.where(sem_raw < 260, sem_raw, 0)
            learn_labels = lut[sem_raw_safe]  # int64, with -100 for ignore
        else:
            learn_labels = np.full(n, -100, dtype=np.int64)

        labels_t = torch.from_numpy(learn_labels)
        labels_t = self._apply_open_set_remap(labels_t)
        return DatasetEntry(
            points_xyz_m=torch.from_numpy(xyz),
            intensities=torch.from_numpy(intens.clip(0.0, 1.0)),
            labels=labels_t,
            frame_idx=frame_idx,
            sequence=sequence,
        )

    # -------------------------------------------------------- open-set remap

    def _apply_open_set_remap(self, labels: Tensor) -> Tensor:
        if self.open_set_split is None:
            return labels
        unknowns = (
            SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY
            if self.open_set_split == "primary"
            else SEMANTIC_KITTI_UNKNOWN_CLASSES_ROBUSTNESS
        )
        unknown_idx = SEMANTIC_KITTI_NUM_CLASSES  # last channel = explicit unknown
        for c in unknowns:
            labels = torch.where(labels == c, torch.full_like(labels, unknown_idx), labels)
        return labels
