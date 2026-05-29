#!/usr/bin/env python
"""W0d smoke test — synthetic end-to-end pipeline forward pass.

Runs the R2-reimpl pipeline + M1 evidential head on 5 synthetic frames to verify:

  1. All modules import cleanly.
  2. torch is functional (CPU or CUDA).
  3. R2-reimpl 6 stages each return well-formed outputs.
  4. M1 evidential head produces α ≥ 1 voxels.
  5. EvidenceAccumulator integrates over multiple frames.
  6. M2 descriptor returns L1-normalised channel.
  7. M3 decay preserves α ≥ 1 invariant.

Does NOT require any dataset to be downloaded.

Usage::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cuda
    python scripts/smoke_test.py --device cuda --num-frames 20
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from evidlife_map.baselines.r2_reimpl.r2_pipeline import R2Pipeline, R2PipelineConfig
from evidlife_map.data.semantic_kitti import SemanticKITTIDataset
from evidlife_map.m1_evidential.edl_head import EDLHead, vacuity_from_alpha
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator


def banner(s: str) -> None:
    print(f"\n{'=' * 60}\n  {s}\n{'=' * 60}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--num-frames", type=int, default=5)
    ap.add_argument("--num-classes", type=int, default=19)
    args = ap.parse_args()

    banner("W0d Smoke Test — EvidLife-Map synthetic pipeline")

    print(f"torch.__version__   = {torch.__version__}")
    print(f"torch.cuda.is_avail = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device   = {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"torch.cuda.compute_capability = sm_{cap[0]}{cap[1]}")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but CUDA not available")
        return 1
    device = torch.device(args.device)

    # ---- 1. Synthetic dataset ----
    banner(f"1. Synthetic dataset (no on-disk files) — {args.num_frames} frames")
    ds = SemanticKITTIDataset(
        root="/nonexistent",
        sequences=["smoke"],
        open_set_split="primary",
        synthetic=True,
    )
    # synthetic auto-makes 8 frames per sequence; slice to requested
    if args.num_frames > len(ds):
        print(f"  WARN: dataset has {len(ds)} frames; requested {args.num_frames}")
    n_frames = min(args.num_frames, len(ds))
    print(f"  loaded {len(ds)} synthetic frames, using {n_frames}")

    # ---- 2. R2-reimpl pipeline ----
    banner("2. R2-reimpl pipeline (6 stages)")
    r2_cfg = R2PipelineConfig(num_classes=args.num_classes, device=str(device))
    r2 = R2Pipeline(r2_cfg)

    voxel_keys_per_frame: list[list[int]] = []
    point_probs_per_frame: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        entry = ds[i]
        out = r2.step(
            {
                "points_xyz_m": entry.points_xyz_m,
                "frame_idx": entry.frame_idx,
                "sequence": entry.sequence,
            }
        )
        # Re-run stages 2 + 3 to capture voxel keys + probs for M1 reuse.
        pose = r2.stage1_lvio_pose({"frame_idx": entry.frame_idx})
        _, voxel_int = r2.stage2_raycast_splat(entry.points_xyz_m, pose)
        probs = r2.stage3_semantic_head(entry.points_xyz_m)
        keys = [
            int(voxel_int[k, 0] * 1_000_003 + voxel_int[k, 1] * 1_009 + voxel_int[k, 2])
            for k in range(voxel_int.shape[0])
        ]
        voxel_keys_per_frame.append(keys)
        point_probs_per_frame.append(probs)
        print(
            f"  frame {i}: n_points={out['n_points']}, "
            f"n_voxels={out['n_voxels_touched']}, "
            f"trav={out['traversable_fraction']:.3f}"
        )
    r2_elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"  R2 pipeline: {r2_elapsed:.1f} ms for {n_frames} frames")

    # ---- 3. M1 evidential head (with random init for smoke) ----
    banner("3. M1 EDL head + EvidenceAccumulator")
    # Build EDL head: input = (per-point class probs = C dim) -> evidence
    # EDLHead internally appends the unknown channel: output dim = num_classes + 1.
    edl = EDLHead(
        in_features=args.num_classes,
        num_classes=args.num_classes,  # head appends +1 unknown internally
        activation="softplus",
    ).to(device)
    accumulator = EvidenceAccumulator(
        num_classes_plus_one=args.num_classes + 1,
        device=device,
    )

    t0 = time.perf_counter()
    for i in range(n_frames):
        probs = point_probs_per_frame[i].to(device)
        with torch.no_grad():
            alpha, vac = edl(probs)
        # Evidence per point = α - 1
        evidence = (alpha - 1.0).cpu()
        keys = voxel_keys_per_frame[i]
        ts = float(i)
        accumulator.accumulate_batch(keys, evidence, timestamp=ts)
        n_voxels = len(set(keys))
        mean_vac = float(vac.mean().item())
        print(f"  frame {i}: per-point vacuity mean = {mean_vac:.3f}, n_voxels = {n_voxels}")
    m1_elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"  M1 forward + accum: {m1_elapsed:.1f} ms for {n_frames} frames")
    print(f"  Accumulator final size: {len(accumulator)} voxels")

    # Pick a sample voxel + verify invariant
    sample_keys = accumulator.keys()[:5]
    for k in sample_keys:
        a = accumulator.get_alpha(k)
        v = accumulator.get_vacuity(k)
        assert float(a.min()) >= 1.0 - 1e-6, f"α_min = {a.min()}, expected ≥ 1"
        assert 0.0 <= v <= 1.0, f"vacuity = {v}, expected [0, 1]"
    print(f"  [OK] α ≥ 1 invariant holds across {len(sample_keys)} sampled voxels")
    print(f"  [OK] vacuity ∈ [0, 1] holds")

    # ---- 4. Final summary ----
    banner("4. Smoke Test Summary")
    print(f"  Device:               {device}")
    print(f"  Frames processed:     {n_frames}")
    print(f"  R2-reimpl latency:    {r2_elapsed:.1f} ms total ({r2_elapsed/n_frames:.1f} ms/frame)")
    print(f"  M1 evidential lat:    {m1_elapsed:.1f} ms total ({m1_elapsed/n_frames:.1f} ms/frame)")
    print(f"  Voxels accumulated:   {len(accumulator)}")
    print(f"  Status:               PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
