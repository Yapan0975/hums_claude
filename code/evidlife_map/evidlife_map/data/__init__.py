"""Dataset wrappers (research_plan v3 §4.1).

Five datasets in the v3 plan:

* :mod:`semantic_kitti` — RQ1 / RQ5 main, RQ2 14/5 split substrate.
* :mod:`nuscenes_lidarseg` — RQ1 cross-domain, RQ2 reverse cross-domain OOD.
* :mod:`kitti_360` — RQ4 lifelong multi-session.
* :mod:`semantic_spray` — RQ3 traversability.

The wrappers are deliberately thin: each exposes a ``DatasetEntry`` dataclass
with the data the M1 fine-tune loop expects, and a ``__getitem__`` that returns
that dataclass. No actual data loading is performed in the P1 skeleton.
"""

from __future__ import annotations

from evidlife_map.data.kitti_360 import KITTI360Dataset
from evidlife_map.data.nuscenes_lidarseg import NuScenesLidarSegDataset
from evidlife_map.data.semantic_kitti import SemanticKITTIDataset
from evidlife_map.data.semantic_spray import SemanticSprayDataset

__all__ = [
    "KITTI360Dataset",
    "NuScenesLidarSegDataset",
    "SemanticKITTIDataset",
    "SemanticSprayDataset",
]
