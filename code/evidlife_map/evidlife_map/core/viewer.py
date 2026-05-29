"""open3d-based visualiser for the voxel map (debug + demo overlay).

The viewer is intentionally minimal — it draws each occupied voxel as a coloured
cube whose colour either encodes (a) the argmax class or (b) the vacuity heat
value. Used by:

* The interactive debug loop during W7 M2 bring-up.
* The Jetson Orin NX 30 s demo capture script (``evidlife_map.jetson.record_demo``).

We import ``open3d`` lazily so that headless CI runs without the GPU can still
collect this file's tests.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

from evidlife_map.m1_evidential.edl_head import dirichlet_mean, vacuity_from_alpha


def _load_open3d() -> ModuleType:
    try:
        return import_module("open3d")
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for visualisation; install with `pip install open3d`"
        ) from exc


def _vacuity_colormap(vacuity: Tensor) -> Tensor:
    """Simple blue-to-red colour map: low vacuity = cool, high = hot."""
    v = vacuity.clamp(0.0, 1.0)
    r = v
    g = 1.0 - (v - 0.5).abs() * 2.0
    b = 1.0 - v
    return torch.stack([r, g.clamp(0.0, 1.0), b], dim=-1)


class VoxelMapViewer:
    """Wrapper around an open3d Visualizer that knows about ``α`` and ``u_v``.

    Parameters
    ----------
    voxel_size_m
        Edge length of one voxel cube in world metres.
    palette
        Optional ``(C+1, 3)`` RGB palette for the class colour mode.
        If ``None``, a deterministic Hue cycling palette is generated.
    """

    def __init__(
        self,
        *,
        voxel_size_m: float = 0.25,
        palette: Tensor | None = None,
    ) -> None:
        self.voxel_size_m = float(voxel_size_m)
        self.palette = palette
        self._vis: Any | None = None

    def _ensure_window(self) -> Any:
        if self._vis is None:
            o3d = _load_open3d()
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(window_name="EvidLife-Map viewer")
        return self._vis

    def add_voxel_cloud(
        self,
        voxel_centres_m: Tensor,
        alpha: Tensor,
        *,
        colour_mode: str = "vacuity",
    ) -> None:
        """Add coloured voxels to the open3d window.

        Parameters
        ----------
        voxel_centres_m
            ``(V, 3)`` world-frame voxel centres.
        alpha
            ``(V, C+1)`` per-voxel concentrations.
        colour_mode
            ``"vacuity"`` (default) or ``"class"``.
        """
        if colour_mode not in {"vacuity", "class"}:
            raise ValueError(f"unknown colour_mode {colour_mode!r}")
        if voxel_centres_m.shape[0] != alpha.shape[0]:
            raise ValueError("voxel_centres_m and alpha row count must match")

        o3d = _load_open3d()
        vis = self._ensure_window()

        if colour_mode == "vacuity":
            colours = _vacuity_colormap(vacuity_from_alpha(alpha)).cpu().numpy()
        else:
            pred = dirichlet_mean(alpha).argmax(dim=-1)
            palette = (
                self.palette
                if self.palette is not None
                else _default_palette(alpha.shape[-1], device=alpha.device)
            )
            colours = palette[pred].cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_centres_m.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colours)
        vis.add_geometry(pcd)

    def spin_once(self) -> None:
        vis = self._ensure_window()
        vis.poll_events()
        vis.update_renderer()

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None


def _default_palette(n: int, *, device: torch.device | str = "cpu") -> Tensor:
    """Hue-cycling deterministic palette."""
    hues = torch.linspace(0.0, 1.0, n + 1, device=device)[:-1]
    # Simple HSV-to-RGB with S=1, V=1.
    h6 = hues * 6.0
    i = h6.floor()
    f = h6 - i
    q = 1.0 - f
    out = torch.zeros(n, 3, device=device)
    for k in range(n):
        ii = int(i[k].item()) % 6
        if ii == 0:
            out[k] = torch.tensor([1.0, f[k], 0.0], device=device)
        elif ii == 1:
            out[k] = torch.tensor([q[k], 1.0, 0.0], device=device)
        elif ii == 2:
            out[k] = torch.tensor([0.0, 1.0, f[k]], device=device)
        elif ii == 3:
            out[k] = torch.tensor([0.0, q[k], 1.0], device=device)
        elif ii == 4:
            out[k] = torch.tensor([f[k], 0.0, 1.0], device=device)
        else:
            out[k] = torch.tensor([1.0, 0.0, q[k]], device=device)
    return out
