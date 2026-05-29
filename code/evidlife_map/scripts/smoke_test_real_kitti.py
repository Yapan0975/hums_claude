#!/usr/bin/env python
"""W1 G-1 minimum: real SemanticKITTI smoke test.

Loads real .bin + .label frames from disk, runs R2-reimpl pipeline + M1
evidential head, computes per-frame mIoU (placeholder model = random softmax,
so mIoU will be low — that's expected for smoke). Goal is just to validate
data → model → metric end-to-end on real data.

Usage::

    python scripts/smoke_test_real_kitti.py --root /d/datasets/SemanticKITTI \
        --seq 08 --frames 5 --device cuda

Expected output: per-frame mIoU between ~0.0-0.05 (random head),
and r2 pipeline runtime per frame. Once we plug in a real trained head
the mIoU jumps to the published 50-77 range.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evidlife_map.baselines.r2_reimpl.r2_pipeline import R2Pipeline, R2PipelineConfig
from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.edl_head import EDLHead
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator


def banner(s: str) -> None:
    print(f"\n{'=' * 60}\n  {s}\n{'=' * 60}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="SemanticKITTI root containing dataset/sequences/")
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--semantic-weights", type=Path, default=None,
                    help="Optional path to PointNet ckpt; random init if omitted.")
    args = ap.parse_args()

    banner("W1 G-1 Smoke Test — real SemanticKITTI")

    # Auto-locate "dataset/" sub-root if present (matches official zip layout).
    root = args.root
    if (root / "dataset" / "sequences").exists():
        root = root / "dataset"
        print(f"  auto-located dataset/ subdir at {root}")
    seq_dir = root / "sequences" / args.seq / "velodyne"
    if not seq_dir.exists():
        print(f"ERROR: missing velodyne dir: {seq_dir}")
        print(f"  Expected layout: {root}/sequences/{args.seq}/velodyne/*.bin")
        return 1
    n_avail = len(list(seq_dir.glob("*.bin")))
    print(f"  found {n_avail} velodyne frames at {seq_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"  device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM free {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

    # ---- 1. Dataset ----
    banner(f"1. Real SemanticKITTI dataset (seq {args.seq})")
    ds = SemanticKITTIDataset(
        root=root,
        sequences=[args.seq],
        open_set_split=None,
        synthetic=False,  # REAL loader
    )
    n_frames = min(args.frames, len(ds))
    print(f"  dataset size: {len(ds)}, using {n_frames} frames")

    # Inspect first frame structure
    entry0 = ds[0]
    print(f"  frame 0: n_points={entry0.points_xyz_m.shape[0]}, "
          f"x range [{entry0.points_xyz_m[:,0].min():.1f}, {entry0.points_xyz_m[:,0].max():.1f}]")
    n_valid = (entry0.labels >= 0).sum().item()
    n_ignore = (entry0.labels < 0).sum().item()
    print(f"  frame 0 labels: {n_valid} valid + {n_ignore} ignore (after remap)")

    # ---- 2. R2-reimpl pipeline ----
    banner("2. R2-reimpl on real frames")
    r2_cfg = R2PipelineConfig(
        num_classes=SEMANTIC_KITTI_NUM_CLASSES,
        device=str(device),
        semantic_weights_path=args.semantic_weights,
    )
    r2 = R2Pipeline(r2_cfg)
    print(f"  semantic_weights_path: {args.semantic_weights}")

    voxel_keys_per_frame: list[torch.Tensor] = []
    point_probs_per_frame: list[torch.Tensor] = []
    point_labels_per_frame: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        entry = ds[i]
        out = r2.step({
            "points_xyz_m": entry.points_xyz_m,
            "frame_idx": entry.frame_idx,
            "sequence": entry.sequence,
        })
        pose = r2.stage1_lvio_pose({"frame_idx": entry.frame_idx})
        _, voxel_int = r2.stage2_raycast_splat(entry.points_xyz_m, pose)
        probs = r2.stage3_semantic_head(entry.points_xyz_m, intensities=entry.intensities).cpu()
        # Vectorised voxel-key hash (matches R2Pipeline.step()).
        keys = (
            voxel_int[:, 0] * 1_000_003
            + voxel_int[:, 1] * 1_009
            + voxel_int[:, 2]
        )
        voxel_keys_per_frame.append(keys)
        point_probs_per_frame.append(probs)
        point_labels_per_frame.append(entry.labels)
        print(f"  frame {i}: n_points={out['n_points']}, "
              f"n_voxels={out['n_voxels_touched']}, "
              f"trav={out['traversable_fraction']:.3f}")
    r2_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  R2 pipeline: {r2_ms:.0f} ms total ({r2_ms/n_frames:.0f} ms/frame)")

    # ---- 3. Per-point mIoU (random head, expected ~5%) ----
    banner("3. Per-point mIoU on real labels (random head — sanity only)")
    iou = IoUTracker(num_classes=SEMANTIC_KITTI_NUM_CLASSES, ignore_index=-100)
    for probs, labels in zip(point_probs_per_frame, point_labels_per_frame, strict=True):
        pred = probs.argmax(dim=-1)  # (N,)
        iou.update(pred, labels)
    metric = iou.compute()
    print(f"  miou = {metric['miou']:.4f}   (random head expected ~0.05 = 1/19)")
    print(f"  per-class iou:")
    for c, v in enumerate(metric['per_class_iou']):
        if v is not None:
            print(f"    class {c:2d}: iou = {v:.4f}")

    # ---- 4. M1 evidential head over the same frames ----
    banner("4. M1 EDL head on real frames")
    edl = EDLHead(in_features=SEMANTIC_KITTI_NUM_CLASSES,
                  num_classes=SEMANTIC_KITTI_NUM_CLASSES,
                  activation="softplus").to(device)
    accumulator = EvidenceAccumulator(
        num_classes_plus_one=SEMANTIC_KITTI_NUM_CLASSES + 1,
        device=device,
    )
    t0 = time.perf_counter()
    for i in range(n_frames):
        probs = point_probs_per_frame[i].to(device)
        with torch.no_grad():
            alpha, vac = edl(probs)
        evidence = (alpha - 1.0).cpu()
        keys = voxel_keys_per_frame[i].tolist()
        accumulator.accumulate_batch(keys, evidence, timestamp=float(i))
        n_unique = int(torch.unique(voxel_keys_per_frame[i]).shape[0])
        print(f"  frame {i}: vacuity mean={vac.mean().item():.3f}, n_voxels={n_unique}")
    m1_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  M1 + accum: {m1_ms:.0f} ms total ({m1_ms/n_frames:.0f} ms/frame)")
    print(f"  final accumulator size: {len(accumulator)} unique voxels")

    # Verify invariants on real data
    sample = accumulator.keys()[:10]
    for k in sample:
        a = accumulator.get_alpha(k)
        v = accumulator.get_vacuity(k)
        assert a.min().item() >= 1.0 - 1e-6
        assert 0.0 <= v <= 1.0
    print(f"  [OK] alpha >= 1 + vacuity in [0,1] on {len(sample)} real voxels")

    # ---- 5. Summary ----
    banner("W1 G-1 Smoke Test Summary")
    print(f"  Sequence:     {args.seq} ({n_frames} frames)")
    print(f"  Device:       {device}")
    print(f"  R2 latency:   {r2_ms/n_frames:.0f} ms/frame")
    print(f"  M1 latency:   {m1_ms/n_frames:.0f} ms/frame")
    print(f"  Voxels:       {len(accumulator)}")
    print(f"  Random-head mIoU: {metric['miou']:.4f} (sanity — expected ~5%)")
    print(f"  Status:       PASS (data → pipeline → metric end-to-end)")
    print()
    print("  Next: replace R2Pipeline.stage3_semantic_head() MLP stand-in with")
    print("        a trained semantic head (HRNet or Cylinder3D) and re-run for")
    print("        the real W1 G-1 numbers (target: ConvBKI 77.7 mIoU on seq 08).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
