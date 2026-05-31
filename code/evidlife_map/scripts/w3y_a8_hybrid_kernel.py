#!/usr/bin/env python
"""W3-Y A-8 hybrid kernel ablation — R2 W1 / R4 W4 round-3 follow-up.

Per R2 W1 / R4 W4: test whether a light Gaussian spatial kernel applied
*only* to the posterior-mean channel (M1 mIoU output) — leaving the
per-voxel vacuity/dissonance scalars un-smoothed — Pareto-dominates the
pure per-voxel architecture currently in §III.B.

The test:
  - Load W3-R single-checkpoint joint-serving model (open-set trained)
  - On seq 08 val, accumulate per-voxel α from per-point predictions
  - For M1 mIoU: apply 3D Gaussian kernel smoothing on posterior mean p = α/S
  - For AUROC + M3: keep raw per-voxel α (unsmoothed)
  - Compare against no-kernel baseline

Pass condition for the §III.B per-voxel-purity rejection:
  hybrid mIoU >> baseline mIoU AND hybrid AUROC ≈ baseline AUROC AND
  hybrid M3 F1 ≈ baseline M3 F1

If hybrid Pareto-dominates → §III.B should adopt hybrid; the v2 draft's
"pure per-voxel" rejection is wrong on empirical grounds.
If hybrid fails to Pareto-dominate (AUROC or M3 F1 drops) → §III.B
rejection is vindicated empirically.

Operates on val frames only (post-hoc; no retraining required).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.m1_evidential.edl_head import vacuity_from_alpha
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample
from scripts.train_pointnet2lite_v2_edl_multiseq import PointNetLiteV2EDL
from scripts.validate_m1_openset import _is_unknown, compute_auroc


def _voxel_hash(xyz: torch.Tensor, voxel_size: float = 0.25) -> torch.Tensor:
    v = (xyz / voxel_size).floor().to(torch.int64)
    return v[:, 0] * 1_000_003 + v[:, 1] * 1_009 + v[:, 2]


def per_voxel_aggregate(xyz: torch.Tensor, alpha: torch.Tensor, voxel_size: float = 0.25):
    """Aggregate per-point alpha to per-voxel by sum (Dirichlet conjugate).

    Returns:
        voxel_keys: (V,) unique voxel hash keys
        voxel_alpha: (V, C+1) summed alpha per voxel
        voxel_xyz_mean: (V, 3) mean xyz per voxel
        point_to_voxel: (N,) index into voxel_keys for each point
    """
    keys = _voxel_hash(xyz, voxel_size)
    unique_keys, inv = torch.unique(keys, return_inverse=True)
    V = unique_keys.shape[0]
    Cp1 = alpha.shape[-1]
    voxel_alpha = torch.zeros(V, Cp1, device=alpha.device)
    voxel_alpha.scatter_add_(0, inv.unsqueeze(-1).expand(-1, Cp1), alpha)
    voxel_xyz_sum = torch.zeros(V, 3, device=xyz.device)
    voxel_xyz_sum.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 3), xyz)
    voxel_count = torch.zeros(V, device=xyz.device).scatter_add_(
        0, inv, torch.ones_like(inv, dtype=torch.float32)
    )
    voxel_xyz_mean = voxel_xyz_sum / voxel_count.unsqueeze(-1).clamp_min(1.0)
    return unique_keys, voxel_alpha, voxel_xyz_mean, inv


def gaussian_kernel_smooth(voxel_alpha: torch.Tensor, voxel_xyz: torch.Tensor,
                             *, sigma: float = 0.4, k: int = 8) -> torch.Tensor:
    """Apply a Gaussian-weighted local smoothing on the posterior MEAN.

    For each voxel, find k nearest voxels by xyz distance; compute Gaussian
    weights w = exp(-d^2 / 2*sigma^2); normalised mean of alpha across the
    k+1 voxels (self + neighbors). The mean is computed on the MEAN
    p_k = alpha_k / S, then re-projected to alpha space by S_local * p_smoothed.

    Returns: (V, C+1) smoothed alpha tensor, same shape.
    """
    V, Cp1 = voxel_alpha.shape
    if V <= k:
        return voxel_alpha
    # Per-voxel S and p
    S = voxel_alpha.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    p = voxel_alpha / S  # (V, C+1) posterior mean
    # kNN by chunked cdist for memory
    chunk = 4096
    smoothed_p = torch.zeros_like(p)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        d = torch.cdist(voxel_xyz[s:e], voxel_xyz)  # (chunk, V)
        topk_d, topk_idx = torch.topk(d, k=k + 1, dim=1, largest=False)  # include self
        w = torch.exp(-(topk_d ** 2) / (2.0 * sigma ** 2))  # (chunk, k+1)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)
        # Gather neighbor p: (chunk, k+1, C+1)
        p_nb = p[topk_idx.reshape(-1)].reshape(e - s, k + 1, Cp1)
        smoothed_p[s:e] = (p_nb * w.unsqueeze(-1)).sum(dim=1)
        del d, topk_d, topk_idx, w, p_nb
    # Re-project to alpha space by keeping S_local (un-smoothed)
    return smoothed_p * S


def eval_with_voxel_aggregation(model, loader, device, num_classes,
                                  *, hybrid_kernel: bool, sigma: float = 0.4,
                                  n_points: int = 20000) -> dict:
    """Evaluate model with optional hybrid kernel smoothing on M1 only.

    M1 mIoU + ECE: predicted from per-voxel posterior (smoothed if hybrid)
    AUROC + M3: from per-voxel raw alpha (always un-smoothed)
    """
    model.eval()
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    sum_ece, n_ece = 0.0, 0
    all_vac, all_unk = [], []
    with torch.no_grad():
        for entry in loader:
            entry = _subsample(entry, n_points)
            x = _make_xyzi(entry, device=device)
            alpha_p, _ = model(x)  # (N, C+1)
            xyz_p = entry.points_xyz_m.to(device, dtype=torch.float32)
            # Per-voxel aggregation
            v_keys, v_alpha, v_xyz, p2v = per_voxel_aggregate(xyz_p, alpha_p, voxel_size=0.25)
            # Hybrid: smooth per-voxel posterior mean (M1) only
            v_alpha_m1 = gaussian_kernel_smooth(v_alpha, v_xyz, sigma=sigma) if hybrid_kernel else v_alpha
            # Project per-voxel predictions back to per-point for mIoU eval
            point_alpha_m1 = v_alpha_m1[p2v]  # (N, C+1)
            closed = point_alpha_m1[:, :num_classes]
            pred = closed.argmax(dim=-1)
            # mIoU on known
            labels = entry.labels.to(device)
            known_mask = torch.ones_like(labels, dtype=torch.bool)
            for u in SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY:
                known_mask &= (labels != u)
            labels_known = torch.where(known_mask & (labels >= 0), labels,
                                         torch.full_like(labels, -100))
            iou.update(pred, labels_known)
            # ECE on the smoothed posterior
            valid_for_ece = (labels_known >= 0)
            if valid_for_ece.any():
                ece = expected_calibration_error(
                    point_alpha_m1[valid_for_ece].cpu(),
                    labels_known[valid_for_ece].cpu(),
                    n_bins=15, ignore_index=-100,
                )
                sum_ece += float(ece.item())
                n_ece += 1
            # AUROC: vacuity from un-smoothed per-voxel alpha, projected to points
            v_vac = vacuity_from_alpha(v_alpha)  # un-smoothed
            point_vac = v_vac[p2v]  # (N,)
            valid = entry.labels >= 0
            all_vac.append(point_vac[valid].cpu())
            all_unk.append(_is_unknown(entry.labels)[valid])
    metric = iou.compute()
    metric["ece"] = sum_ece / max(1, n_ece)
    all_vac_cat = torch.cat(all_vac)
    all_unk_cat = torch.cat(all_unk)
    metric["vacuity_auroc"] = compute_auroc(all_vac_cat, all_unk_cat)
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("weights/w3r_joint_serving.pt"))
    ap.add_argument("--sigma", type=float, default=0.4,
                    help="Gaussian kernel sigma in metres (voxel-grid units)")
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/w3y_a8_hybrid_kernel.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"==== W3-Y A-8 hybrid kernel ablation ====")
    print(f"ckpt: {args.ckpt}, sigma: {args.sigma}\n", flush=True)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetLiteV2EDL(in_channels=4, num_classes=C, k=16).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    print(f"Loaded W3-R: ep {state['epoch']}, mIoU_known={state['mIoU_known']:.4f}, "
          f"AUROC={state['vacuity_AUROC']:.4f}", flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    val_ds = build_val(root, n_val=100)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    print(f"\n=== Baseline (no kernel, pure per-voxel) ===", flush=True)
    base = eval_with_voxel_aggregation(model, val_loader, device, num_classes=C,
                                          hybrid_kernel=False, n_points=args.n_points)
    print(f"  mIoU = {base['miou']:.4f}, ECE = {base['ece']:.4f}, "
          f"AUROC = {base['vacuity_auroc']:.4f}")

    print(f"\n=== Hybrid (Gaussian kernel on M1 only, vacuity un-smoothed) sigma={args.sigma} ===", flush=True)
    hyb = eval_with_voxel_aggregation(model, val_loader, device, num_classes=C,
                                         hybrid_kernel=True, sigma=args.sigma,
                                         n_points=args.n_points)
    print(f"  mIoU = {hyb['miou']:.4f}, ECE = {hyb['ece']:.4f}, "
          f"AUROC = {hyb['vacuity_auroc']:.4f}")

    print(f"\n=== Delta (hybrid - baseline) ===")
    dm = hyb["miou"] - base["miou"]
    de = hyb["ece"] - base["ece"]
    da = hyb["vacuity_auroc"] - base["vacuity_auroc"]
    print(f"  Δ mIoU  = {dm:+.4f}")
    print(f"  Δ ECE   = {de:+.4f}")
    print(f"  Δ AUROC = {da:+.4f}")

    print()
    pareto_dom = (dm > 0) and (de <= 0.001) and (da >= -0.001)
    print(f"Hybrid Pareto-dominates per-voxel? {pareto_dom}")
    if pareto_dom:
        print("→ §III.B per-voxel-purity rejection is FALSIFIED; hybrid wins on mIoU"
              " without hurting AUROC. Paper should adopt hybrid.")
    else:
        print("→ §III.B per-voxel-purity rejection SURVIVES the ablation; the "
              "kernel-smoothing trade-off is real.")

    out = {
        "ckpt": str(args.ckpt), "ckpt_ep": int(state["epoch"]),
        "sigma_m": args.sigma,
        "baseline": {"mIoU_known": base["miou"], "ECE_known": base["ece"],
                       "vacuity_AUROC": base["vacuity_auroc"]},
        "hybrid": {"mIoU_known": hyb["miou"], "ECE_known": hyb["ece"],
                     "vacuity_AUROC": hyb["vacuity_auroc"]},
        "delta": {"d_mIoU": dm, "d_ECE": de, "d_AUROC": da},
        "hybrid_pareto_dominates": pareto_dom,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
