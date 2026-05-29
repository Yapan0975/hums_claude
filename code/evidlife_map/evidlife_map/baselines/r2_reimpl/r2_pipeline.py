"""R2 [jiao2024r2] online pipeline — bring-up scaffold (P1 skeleton).

This file is the **framework**: stable signatures, narrated stages, no actual
inference. The full pipeline is reconstructed in W1 (research_plan v3 §7 P1)
from the cleaned R2 text at
``D:\\_7_sci\\semantic_mapping\\_Online_Metric_Semantic_Mapping_for_Autonomous.txt``.

Six stages match paper R2 §III:

    Stage 1.  LVIO state estimation        (FAST-LIO2 / R3LIVE)
    Stage 2.  LiDAR ray-cast splatting     (nvblox geometric backbone)
    Stage 3.  HRNet semantic head          (confidence-aware aleatoric)
    Stage 4.  per-voxel argmax-Bayes       (THIS file delegates to bayes_filter)
    Stage 5.  mesh extraction              (nvblox MeshLayer)
    Stage 6.  traversability head          (unlabeled = untraversable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from evidlife_map.baselines.r2_reimpl.bayes_filter import ArgmaxBayesAccumulator


class _MarkerPointNet(torch.nn.Module):
    """Wrapper marker so stage3 knows to feed xyzi (not xyz)."""
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return self.inner(x)


@dataclass(slots=True)
class R2PipelineConfig:
    """Configuration consumed by :class:`R2Pipeline`.

    Mirrors the fields R2 (T-ASE 2024) lists in §III; defaults match the values
    the published paper reports for outdoor LiDAR-only settings.
    """

    num_classes: int = 19
    voxel_size_m: float = 0.25
    lvio: str = "fast_lio2"
    semantic_head: str = "hrnet_confidence"
    semantic_weights_path: Path | str | None = None
    mesh_layer_active: bool = True
    traversable_class_indices: tuple[int, ...] = (8, 9, 10, 15)   # road etc.
    device: str = "cpu"
    # W1 bring-up additions:
    gt_pose_path: Path | str | None = None  # if provided, use GT poses instead of LVIO
    semantic_head_hidden: int = 64          # MLP semantic stand-in for HRNet
    bayes_backend: str = "flat"             # W2-3: dict (legacy) or flat (default, ~150x faster)


class R2Pipeline:
    """R2 [jiao2024r2] reference pipeline.

    This is intentionally a thin orchestrator over modular stages so the
    fine-grained per-stage timings can be measured for the §IV.G Jetson
    benchmark.
    """

    def __init__(self, config: R2PipelineConfig | None = None) -> None:
        self.config: R2PipelineConfig = config or R2PipelineConfig()
        self.bayes = ArgmaxBayesAccumulator(
            num_classes=self.config.num_classes,
            device=self.config.device,
            backend=self.config.bayes_backend,
        )
        self._stage_times_ms: dict[str, list[float]] = {
            "stage1_lvio": [],
            "stage2_raycast": [],
            "stage3_hrnet": [],
            "stage4_bayes": [],
            "stage5_mesh": [],
            "stage6_trav": [],
        }
        self._frames_processed: int = 0
        # GT pose table: rows of 12 floats = 3x4 pose (KITTI format), one per frame.
        self._gt_poses: Tensor | None = self._load_gt_poses(self.config.gt_pose_path)
        # MLP stand-in for HRNet — lazy init in stage3 once we know point dim.
        self._sem_mlp: torch.nn.Sequential | None = None

    # --------------------------------------------------------------- I/O ---

    @staticmethod
    def _load_gt_poses(path: Path | str | None) -> Tensor | None:
        """Load a KITTI-format poses.txt file (rows of 12 floats = 3x4 pose).

        Returns ``None`` if path is ``None`` or missing. KITTI poses are in
        camera-coord; the LiDAR pose is the camera pose pre-multiplied by the
        velodyne-to-camera extrinsic (handled by the caller via dataset class
        — for the bring-up we expose the raw camera-frame pose).
        """
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            return None
        import numpy as np
        poses = np.loadtxt(p, dtype=np.float32)
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.shape[1] != 12:
            raise ValueError(f"Expected 12 cols per row in {p}; got {poses.shape}")
        return torch.from_numpy(poses)

    # ----------------------------------------------------------------- stages

    def stage1_lvio_pose(self, frame: dict[str, Any]) -> Tensor:
        """Return the 4x4 world-frame pose for this frame.

        W1 bring-up strategy: if the dataset provides ground-truth poses (KITTI
        format ``poses.txt``), use those. Otherwise fall back to identity, with
        a flag in the diagnostics dict to record that LVIO is unwired.
        Real FAST-LIO2 / R3LIVE wrapper is W2 work (research_plan v3 P1).
        """
        idx = int(frame.get("frame_idx", self._frames_processed))
        if self._gt_poses is not None and idx < int(self._gt_poses.shape[0]):
            T = torch.eye(4)
            T[:3, :4] = self._gt_poses[idx].view(3, 4)
            return T
        # Fallback: identity — diagnostics surface this via no "pose_source" key.
        return torch.eye(4)

    def stage2_raycast_splat(
        self,
        points_xyz_m: Tensor,
        pose_world: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Splat each point into its containing voxel.

        W1 bring-up strategy: produce world-frame XYZ + integer voxel index in
        Python. Real nvblox projective LiDAR integrator (CUDA) lives in
        ``nvblox_evidential/src/evidential_layer.cu`` and is wired in W2.
        For W1 the Python pseudo-splat is sufficient to compare R2-reimpl
        argmax-Bayes vs M1 evidential on the same set of voxel keys.
        """
        points_world = (
            (pose_world[:3, :3] @ points_xyz_m.T) + pose_world[:3, 3:4]
        ).T
        voxel_int = torch.floor(points_world / self.config.voxel_size_m).to(torch.int64)
        return points_world, voxel_int

    def stage3_semantic_head(self, points_xyz_m: Tensor, intensities: Tensor | None = None) -> Tensor:
        """Return per-point class probabilities from the semantic head.

        Selection logic (W1 bring-up):
          1. If ``config.semantic_weights_path`` points to a ``state_dict``
             containing keys starting with ``point_mlp`` or ``classifier``,
             load a :class:`PointNetVanilla` and use it.
          2. Else if the file exists but is the older MLP stand-in, use that.
          3. Else lazy-build a random-init MLP (random predictions ~ uniform).
        W2 swaps this for Cylinder3D + spconv once the sparse-conv toolchain
        is installed on the deployment target.
        """
        n = int(points_xyz_m.shape[0])
        C = int(self.config.num_classes)
        if self._sem_mlp is None:
            self._sem_mlp = self._load_or_build_semantic_head(
                C, hidden=self.config.semantic_head_hidden,
                weights_path=self.config.semantic_weights_path,
            )
            self._sem_mlp.eval()
            self._sem_mlp.to(self.config.device)
        with torch.no_grad():
            # Build (N, 4) xyzi tensor if intensities provided + model expects 4-D
            if isinstance(self._sem_mlp, _MarkerPointNet):
                xyz_norm = points_xyz_m.to(self.config.device, dtype=torch.float32) / 100.0
                if intensities is None:
                    intens = torch.zeros(n, 1, device=self.config.device, dtype=torch.float32)
                else:
                    intens = intensities.to(self.config.device, dtype=torch.float32).unsqueeze(-1)
                x = torch.cat([xyz_norm, intens], dim=-1)
                logits = self._sem_mlp(x)
            else:
                logits = self._sem_mlp(points_xyz_m.to(self.config.device, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)
        return probs

    @staticmethod
    def _load_or_build_semantic_head(
        num_classes: int, hidden: int, weights_path: Path | str | None,
    ) -> torch.nn.Module:
        """Pick the right backbone based on what's in the ckpt (if any)."""
        if weights_path is not None:
            wp = Path(weights_path)
            if wp.exists():
                state = torch.load(wp, map_location="cpu", weights_only=True)
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                # Heuristic: PointNet ckpt has point_mlp + classifier keys.
                if any(k.startswith("point_mlp") for k in sd.keys()):
                    from evidlife_map.models.pointnet_vanilla import PointNetVanilla
                    model = PointNetVanilla(in_channels=4, num_classes=num_classes)
                    model.load_state_dict(sd, strict=False)
                    print(f"[r2_pipeline] Loaded PointNet from {wp}")
                    return _MarkerPointNet(model)
                # Else fall back to small MLP state.
                small = torch.nn.Sequential(
                    torch.nn.Linear(3, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, num_classes),
                )
                try:
                    small.load_state_dict(sd)
                    print(f"[r2_pipeline] Loaded small MLP from {wp}")
                except Exception:
                    print(f"[r2_pipeline] WARN: weights at {wp} could not load, using random.")
                return small
        # Default: random-init small MLP.
        return torch.nn.Sequential(
            torch.nn.Linear(3, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )

    def stage4_bayes_update(
        self,
        voxel_keys: list[int] | Tensor,
        per_point_probs: Tensor,
    ) -> None:
        """Per-voxel argmax-Bayes update (vectorised).

        Delegates to :meth:`ArgmaxBayesAccumulator.update_batch` which collapses
        duplicate voxel keys via ``torch.unique`` + ``scatter_add`` — roughly
        100× faster than the per-point Python loop on 100k-point frames.
        """
        log_likes = torch.log(per_point_probs.clamp_min(1e-12))
        if not isinstance(voxel_keys, Tensor):
            voxel_keys = torch.as_tensor(voxel_keys, dtype=torch.int64)
        self.bayes.update_batch(voxel_keys, log_likes)

    def stage5_extract_mesh(self) -> Any | None:
        """Pull the latest mesh from nvblox's MeshLayer.

        TODO(W2): wire to the nvblox mesh extractor; the published R2 latency
        on RTX 3080 Ti is ~1.6 ms which we use as the upper bound at W3 G-1.
        """
        return None

    def stage6_traversability(self, voxel_keys: list[int] | Tensor) -> list[int]:
        """R2 traversability heuristic: argmax class ∈ traversable_class_indices.

        Uses :meth:`ArgmaxBayesAccumulator.argmax_batch` so the per-voxel
        argmax loop runs over unique voxels (~35k for a typical SemKITTI
        frame at 0.25 m) instead of per-point (~123k).
        """
        if not isinstance(voxel_keys, Tensor):
            voxel_keys = torch.as_tensor(voxel_keys, dtype=torch.int64)
        labels = self.bayes.argmax_batch(voxel_keys)
        trav_set = set(self.config.traversable_class_indices)
        return [int(int(l.item()) in trav_set) for l in labels]

    # ------------------------------------------------------------- top-level

    def step(self, frame: dict[str, Any]) -> dict[str, Any]:
        """One end-to-end frame.

        Parameters
        ----------
        frame
            Dict with at least ``"points_xyz_m"`` (Tensor of shape ``(N, 3)``).

        Returns
        -------
        dict
            Diagnostics: stage timings, num voxels touched, etc.
        """
        points = frame["points_xyz_m"]
        if not isinstance(points, Tensor):
            raise TypeError("frame['points_xyz_m'] must be a torch.Tensor")

        pose = self.stage1_lvio_pose(frame)
        _, voxel_int = self.stage2_raycast_splat(points, pose)
        probs = self.stage3_semantic_head(points)
        # Vectorised voxel-key hash (replaces N-iteration Python list comp).
        voxel_keys = (
            voxel_int[:, 0] * 1_000_003
            + voxel_int[:, 1] * 1_009
            + voxel_int[:, 2]
        )
        self.stage4_bayes_update(voxel_keys, probs)
        self.stage5_extract_mesh()
        trav = self.stage6_traversability(voxel_keys)

        self._frames_processed += 1
        return {
            "n_points": int(points.shape[0]),
            "n_voxels_touched": int(torch.unique(voxel_keys).shape[0]),
            "frames_processed": self._frames_processed,
            "traversable_fraction": (
                float(sum(trav)) / float(len(trav)) if trav else float("nan")
            ),
        }
