#!/usr/bin/env python
"""W2-4 first cut: side-by-side RQ1 comparison.

Runs the R2 pipeline (argmax-Bayes per-voxel) AND the M1 evidential head
(Dirichlet) on the same SemKITTI seq 08 frames, computing mIoU + ECE for
each. This is the first RQ1 data point — even with PointNet semantic
backbone the relative comparison (Bayes vs evidential) is meaningful.

Usage::

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/rq1_compare_seq08.py \
        --root /data/shared/SemanticKITTI \
        --frames 10 \
        --semantic-weights weights/pointnet_seq08_quick.pt

Outputs a JSON results file with the per-system metrics.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

from evidlife_map.baselines.r2_reimpl.r2_pipeline import R2Pipeline, R2PipelineConfig
from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.ece import ECECalculator
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.edl_head import EDLHead, dirichlet_mean, vacuity_from_alpha
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--semantic-weights", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/rq1_compare_seq08.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free)")

    # ---- Build R2 with shared semantic backbone ----
    r2_cfg = R2PipelineConfig(
        num_classes=SEMANTIC_KITTI_NUM_CLASSES,
        device=str(device),
        semantic_weights_path=args.semantic_weights,
        bayes_backend="flat",
    )
    r2 = R2Pipeline(r2_cfg)

    # ---- Build M1 evidential head sharing same backbone output ----
    # The PointNet semantic head outputs (N, C=19) per-point logits. The EDL
    # head reads those as features → produces α of shape (N, C+1). The +1 is
    # the explicit "unknown" channel that participates in evidential math
    # but is dropped before the closed-set IoU comparison (so both R2 and
    # M1 score the same 19 closed-set classes).
    C = SEMANTIC_KITTI_NUM_CLASSES
    edl = EDLHead(in_features=C, num_classes=C, activation="softplus").to(device)
    m1_acc = EvidenceAccumulator(num_classes_plus_one=C + 1,
                                  device=device, backend="flat")

    # ---- Dataset ----
    root = args.root
    if (root / "dataset" / "sequences").exists():
        root = root / "dataset"
    ds = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    n_frames = min(args.frames, len(ds))
    print(f"loaded {len(ds)} frames from seq {args.seq}, processing first {n_frames}")

    # ---- Per-system metrics ----
    iou_r2 = IoUTracker(num_classes=C, ignore_index=-100)
    ece_r2 = ECECalculator()
    iou_m1 = IoUTracker(num_classes=C, ignore_index=-100)
    ece_m1 = ECECalculator()

    # Track latency
    lat_r2 = 0.0
    lat_m1 = 0.0
    lat_total = time.perf_counter()
    voxel_keys_per_frame: list[torch.Tensor] = []
    labels_per_frame: list[torch.Tensor] = []
    pointcounts: list[int] = []

    for i in range(n_frames):
        entry = ds[i]
        # R2 step (also updates internal Bayes accumulator)
        t0 = time.perf_counter()
        r2.step({
            "points_xyz_m": entry.points_xyz_m,
            "frame_idx": entry.frame_idx,
            "sequence": entry.sequence,
        })
        # Re-run stages 1-3 to capture stable voxel_keys + probs for M1 + per-point R2 pred
        pose = r2.stage1_lvio_pose({"frame_idx": entry.frame_idx})
        _, voxel_int = r2.stage2_raycast_splat(entry.points_xyz_m, pose)
        probs = r2.stage3_semantic_head(entry.points_xyz_m, intensities=entry.intensities)
        voxel_keys = (
            voxel_int[:, 0] * 1_000_003 + voxel_int[:, 1] * 1_009 + voxel_int[:, 2]
        )
        torch.cuda.synchronize()
        lat_r2 += time.perf_counter() - t0

        voxel_keys_per_frame.append(voxel_keys)
        labels_per_frame.append(entry.labels)
        pointcounts.append(int(entry.points_xyz_m.shape[0]))

        # M1 evidential update + per-point posterior
        t0 = time.perf_counter()
        probs_dev = probs.to(device)
        with torch.no_grad():
            alpha, vac = edl(probs_dev)  # alpha shape (N, C+1)
        evidence = (alpha - 1.0).cpu()
        m1_acc.accumulate_batch(voxel_keys.cpu().tolist(), evidence, timestamp=float(i))
        torch.cuda.synchronize()
        lat_m1 += time.perf_counter() - t0

        # Per-point predictions:
        # R2: argmax of bayes posterior per voxel, mapped back to points
        r2_pred_voxel = r2.bayes.argmax_batch(voxel_keys)        # (N,) int64 on cpu
        # ECE for R2: convert log-posterior at each point to softmax-ish prob.
        # Use the per-point per-voxel posterior (cached in bayes._flat_log_post).
        # Cheap path: compute per-point probs from frame's R2 per-point likelihood.
        r2_probs_per_point = probs.cpu()                         # (N, C) softmax probs from stage3
        iou_r2.update(r2_pred_voxel, entry.labels)
        ece_r2.update(r2_probs_per_point, entry.labels)

        # M1 evidential per-point: use direct per-point alpha (frame-local)
        alpha_cpu = alpha.cpu()
        # Closed-set IoU: drop the unknown channel before argmax (C+1 -> C)
        m1_pred_closed = alpha_cpu[:, :C].argmax(dim=-1)
        m1_probs_closed = dirichlet_mean(alpha_cpu[:, :C])
        iou_m1.update(m1_pred_closed, entry.labels)
        ece_m1.update(m1_probs_closed, entry.labels)

    total_dt = time.perf_counter() - lat_total
    metric_r2 = iou_r2.compute()
    metric_m1 = iou_m1.compute()
    ece_r2_val = ece_r2.compute()
    ece_m1_val = ece_m1.compute()

    summary = {
        "config": {
            "seq": args.seq,
            "n_frames": n_frames,
            "total_points": sum(pointcounts),
            "semantic_weights": str(args.semantic_weights) if args.semantic_weights else None,
        },
        "systems": {
            "r2_reimpl_argmax_bayes": {
                "mIoU": metric_r2["miou"],
                "ECE": ece_r2_val,
                "latency_s": lat_r2,
                "latency_ms_per_frame": lat_r2 * 1000 / n_frames,
            },
            "m1_evidential_dirichlet": {
                "mIoU": metric_m1["miou"],
                "ECE": ece_m1_val,
                "latency_s": lat_m1,
                "latency_ms_per_frame": lat_m1 * 1000 / n_frames,
            },
        },
        "total_time_s": total_dt,
    }

    print("=" * 60)
    print(f"RQ1 first cut — SemKITTI seq {args.seq} ({n_frames} frames, "
          f"~{sum(pointcounts)//1000}k points total)")
    print("=" * 60)
    for sys_name, m in summary["systems"].items():
        print(f"  {sys_name}")
        print(f"    mIoU = {m['mIoU']:.4f}")
        print(f"    ECE  = {m['ECE']:.4f}")
        print(f"    latency = {m['latency_ms_per_frame']:.1f} ms/frame")
    print(f"\nTotal time: {total_dt:.2f} s ({total_dt*1000/n_frames:.0f} ms/frame both systems)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
