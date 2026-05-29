#!/usr/bin/env python
"""W3-D RQ4 lifelong multi-session simulation (paper §IV.F preview).

Without KITTI-360 yet, simulate a single-vehicle revisit by splitting
SemanticKITTI seq 08 first-100 into Session A (frames 0–49) and Session B
(frames 50–99). The two sessions cover the same physical corridor, so M2's
descriptor matcher should report at least one revisit, and M3 decay applied
during the inter-session gap should age stale voxels appropriately.

This is NOT the final RQ4 evaluation — it's a *sanity check* of the
M2 + M3 + M1 inter-session pipeline on real data before the full
KITTI-360 multi-session arena lands.

Outputs:
  • Per-session voxel counts before / after merge
  • Inter-session ECE drift (paper §IV.F target ≤ 1.5×)
  • Decay effect: # voxels whose vacuity crossed a threshold
  • Stale-voxel removal precision @ recall (paper §IV.F core metric)

Usage on 5090::

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/validate_rq4_lifelong_sim.py \
        --root /data/shared/SemanticKITTI \
        --edl-ckpt weights/pointnet_edl_seq08_30ep.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.edl_head import dirichlet_mean, vacuity_from_alpha
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator
from evidlife_map.m3_decay.conjugate_decay import decay_concentration
from scripts.train_pointnet_edl import PointNetEDL


def _voxel_hash(xyz_m: torch.Tensor, voxel_size: float = 0.25) -> torch.Tensor:
    v = (xyz_m / voxel_size).floor().to(torch.int64)
    return v[:, 0] * 1_000_003 + v[:, 1] * 1_009 + v[:, 2]


def build_session(model, ds, frame_range, device, voxel_size=0.25) -> EvidenceAccumulator:
    """Build an evidence accumulator over a contiguous frame range."""
    C_plus_1 = SEMANTIC_KITTI_NUM_CLASSES + 1
    acc = EvidenceAccumulator(num_classes_plus_one=C_plus_1, device=device, backend="flat")
    for i in frame_range:
        entry = ds[i]
        xyz = entry.points_xyz_m.to(device, dtype=torch.float32)
        inten = entry.intensities.to(device, dtype=torch.float32).unsqueeze(-1)
        x = torch.cat([xyz / 100.0, inten], dim=-1)
        with torch.no_grad():
            alpha, _ = model(x)
        keys = _voxel_hash(xyz, voxel_size).cpu().tolist()
        acc.accumulate_batch(keys, (alpha - 1.0).cpu(), timestamp=float(i))
    return acc


def merge_sessions(a: EvidenceAccumulator, b: EvidenceAccumulator,
                   *, inter_session_gap_s: float = 600.0) -> EvidenceAccumulator:
    """Apply M3 decay over inter-session gap, then merge B's evidence into A."""
    # Decay A's α by inter_session_gap_s
    a_alpha_decayed = decay_concentration(
        a._flat_alpha, inter_session_gap_s,
        tau_min_s=60.0, tau_max_s=3600.0,
    )
    a_keys = a._sorted_keys.clone()
    # Merge by re-accumulating decayed evidence into a fresh accumulator,
    # then re-accumulating B's evidence on top.
    C_plus_1 = a.num_classes_plus_one
    merged = EvidenceAccumulator(num_classes_plus_one=C_plus_1, device=a.device, backend="flat")
    if a_keys.shape[0] > 0:
        # The decayed α values represent "carried-over" evidence (α - 1) post-decay.
        merged.accumulate_batch(a_keys.cpu().tolist(),
                                (a_alpha_decayed - 1.0).cpu(),
                                timestamp=0.0)
    if b._sorted_keys.shape[0] > 0:
        merged.accumulate_batch(b._sorted_keys.cpu().tolist(),
                                (b._flat_alpha - 1.0).cpu(),
                                timestamp=float(inter_session_gap_s))
    return merged


def stale_voxel_precision_recall(
    a: EvidenceAccumulator, b: EvidenceAccumulator,
    *, vacuity_threshold: float = 0.5,
) -> dict:
    """A "stale" voxel = one in Session A whose vacuity passed the threshold
    after M3 decay (i.e., M3 says "evidence is now stale enough").
    Ground truth = "voxel NOT updated by Session B" (no new evidence arrived).
    """
    # Decay A's α as if it sat idle for 600 s
    a_alpha = decay_concentration(a._flat_alpha, 600.0, tau_min_s=60.0, tau_max_s=3600.0)
    a_vac = vacuity_from_alpha(a_alpha)
    flagged_stale = a_vac > vacuity_threshold     # M3 says stale

    # Which A voxels appear in B?
    n_a = int(a._sorted_keys.shape[0])
    if n_a == 0 or b._sorted_keys.numel() == 0:
        return {"precision": float("nan"), "recall": float("nan"),
                "n_flagged_stale": 0, "n_in_a_only": 0, "n_a": n_a}
    pos = torch.searchsorted(b._sorted_keys, a._sorted_keys)
    pos_clamped = pos.clamp(max=int(b._sorted_keys.shape[0]) - 1)
    overlap = (pos < b._sorted_keys.shape[0]) & (b._sorted_keys[pos_clamped] == a._sorted_keys)
    a_only = ~overlap

    # Stale-voxel precision = P(actually-not-revisited | flagged_stale)
    n_flagged = int(flagged_stale.sum())
    n_correctly_flagged = int((flagged_stale & a_only).sum())
    n_a_only = int(a_only.sum())
    precision = (n_correctly_flagged / n_flagged) if n_flagged > 0 else float("nan")
    recall = (n_correctly_flagged / n_a_only) if n_a_only > 0 else float("nan")
    return {
        "vacuity_threshold": vacuity_threshold,
        "precision": precision,
        "recall": recall,
        "n_flagged_stale": n_flagged,
        "n_correctly_flagged": n_correctly_flagged,
        "n_a_only": n_a_only,
        "n_a": n_a,
        "n_b": int(b._sorted_keys.shape[0]),
        "overlap_count": int(overlap.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames-a", type=int, default=50)
    ap.add_argument("--frames-b", type=int, default=50)
    ap.add_argument("--gap-seconds", type=float, default=600.0)
    ap.add_argument("--edl-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/rq4_lifelong_sim.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    C = SEMANTIC_KITTI_NUM_CLASSES

    model = PointNetEDL(in_channels=4, num_classes=C).to(device)
    ckpt = torch.load(args.edl_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"M1 backbone: val_miou={ckpt['val_miou']:.4f} @ ep{ckpt['epoch']}")
    print()

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    ds = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    n = len(ds)

    range_a = range(0, min(args.frames_a, n))
    range_b = range(args.frames_a, min(args.frames_a + args.frames_b, n))
    print(f"Session A: frames {range_a.start}..{range_a.stop-1} ({len(range_a)} frames)")
    print(f"Session B: frames {range_b.start}..{range_b.stop-1} ({len(range_b)} frames)")
    print(f"Inter-session gap simulated: {args.gap_seconds:.0f} s ({args.gap_seconds/60:.1f} min)")
    print()

    t0 = time.perf_counter()
    sess_a = build_session(model, ds, range_a, device)
    t1 = time.perf_counter()
    sess_b = build_session(model, ds, range_b, device)
    t2 = time.perf_counter()
    n_a = len(sess_a)
    n_b = len(sess_b)
    print(f"Session A built in {t1-t0:.1f}s — {n_a} unique voxels")
    print(f"Session B built in {t2-t1:.1f}s — {n_b} unique voxels")

    # Overlap measure: how many voxels visited by both sessions
    if n_a > 0 and n_b > 0:
        pos = torch.searchsorted(sess_b._sorted_keys, sess_a._sorted_keys)
        pos_clamped = pos.clamp(max=n_b - 1)
        overlap = ((pos < n_b) & (sess_b._sorted_keys[pos_clamped] == sess_a._sorted_keys)).sum().item()
        overlap_frac = overlap / n_a
        print(f"Spatial overlap: {overlap} voxels in both sessions ({overlap_frac:.1%} of A)")
    else:
        overlap = 0
        overlap_frac = float("nan")

    print()
    print("=" * 60)
    print("Lifelong merge (M3 decay + accumulation)")
    print("=" * 60)
    merged = merge_sessions(sess_a, sess_b, inter_session_gap_s=args.gap_seconds)
    print(f"Merged accumulator: {len(merged)} unique voxels")
    print(f"  vs A + B union (expected): up to {n_a + n_b}")
    print(f"  dedup saving: {1 - len(merged) / max(n_a + n_b, 1):.1%}")

    # Stale-voxel P/R at threshold 0.5
    pr = stale_voxel_precision_recall(sess_a, sess_b, vacuity_threshold=0.5)
    print()
    print("Stale-voxel removal P/R @ vacuity threshold 0.5:")
    print(f"  Precision = {pr['precision']:.4f}")
    print(f"  Recall    = {pr['recall']:.4f}")
    print(f"  ({pr['n_correctly_flagged']} correctly flagged / {pr['n_flagged_stale']} total flagged)")
    print(f"  ({pr['n_a_only']} voxels actually 'A-only' = ground truth stale)")

    # ECE drift: not directly measurable without per-frame labels in B, so we
    # report the vacuity histograms before/after merge as a proxy.
    pre_vac = vacuity_from_alpha(sess_a._flat_alpha) if n_a > 0 else torch.empty(0)
    post_vac = vacuity_from_alpha(merged._flat_alpha) if len(merged) > 0 else torch.empty(0)
    print()
    print("Vacuity drift between Session A (alone) and merged map:")
    if pre_vac.numel() > 0 and post_vac.numel() > 0:
        print(f"  Session A:        mean {pre_vac.mean():.4f}, max {pre_vac.max():.4f}")
        print(f"  After merge:      mean {post_vac.mean():.4f}, max {post_vac.max():.4f}")

    out = {
        "config": {
            "seq": args.seq, "frames_a": len(range_a), "frames_b": len(range_b),
            "gap_seconds": args.gap_seconds,
            "edl_ckpt": str(args.edl_ckpt),
        },
        "voxel_counts": {"session_a": n_a, "session_b": n_b,
                          "spatial_overlap": overlap, "overlap_frac": overlap_frac,
                          "merged": len(merged)},
        "stale_voxel_pr_at_thresh_0p5": pr,
        "vacuity": {
            "pre_mean": float(pre_vac.mean().item()) if pre_vac.numel() > 0 else None,
            "post_mean": float(post_vac.mean().item()) if post_vac.numel() > 0 else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
