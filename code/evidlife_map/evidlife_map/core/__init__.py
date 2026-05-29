"""Core abstractions shared by M1 / M2 / M3.

* :mod:`voxel_map` — Python-side voxel grid API; ground-truth implementation
  lives in the C++ ``nvblox_evidential`` plug-in.
* :mod:`nvblox_bridge` — thin wrapper around the C++ plug-in once it is built.
* :mod:`traversability` — vacuity-to-traversability conversion (RQ3).
* :mod:`viewer` — open3d-based visualisation (debug + demo video overlay).
"""

from __future__ import annotations

from evidlife_map.core.nvblox_bridge import NvbloxBridge, NvbloxBridgeError
from evidlife_map.core.traversability import vacuity_to_traversability
from evidlife_map.core.viewer import VoxelMapViewer
from evidlife_map.core.voxel_map import VoxelKey, VoxelMap, key_to_xyz, xyz_to_key

__all__ = [
    "NvbloxBridge",
    "NvbloxBridgeError",
    "VoxelKey",
    "VoxelMap",
    "VoxelMapViewer",
    "key_to_xyz",
    "vacuity_to_traversability",
    "xyz_to_key",
]
