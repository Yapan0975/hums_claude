#!/usr/bin/env python
"""P1-B7 R2-reimpl temperature scaling — close R1 W3 (Major).

R1 W3 demands post-hoc temperature scaling (Guo et al. 2017 "On Calibration
of Modern Neural Networks") on the R2-reimpl softmax-Bayes head before
comparing ECE to the EDL Dirichlet head. The v2-draft 4-7× ECE win
compared an EDL-tuned head (W3-M, W3-S) against a softmax-Bayes head
with no calibration adjustment — confounded comparison.

This script:
  1. Loads W3-S CE-lite_v2 ckpt (R2 baseline at W3-M tier)
  2. Fits a single temperature T on val logits by minimising NLL
  3. Reports ECE pre-TS vs post-TS
  4. Optional: also fits Platt-style 2-parameter scaling for sharper baseline

Output: artifacts/p1b7_temperature_scaling.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.models.pointnet2_lite_v2 import PointNet2Lite_v2
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample


@torch.no_grad()
def collect_logits_labels(model, loader, device, n_points=20000):
    """Single pass: gather all logits and labels for temperature fitting."""
    model.eval()
    all_logits, all_labels = [], []
    for entry in loader:
        entry = _subsample(entry, n_points)
        x = _make_xyzi(entry, device=device)
        logits = model(x)  # (N, C)
        valid = entry.labels >= 0
        all_logits.append(logits[valid].cpu())
        all_labels.append(entry.labels[valid].cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                     *, max_iter: int = 100, lr: float = 0.01) -> float:
    """Fit single temperature T to minimise NLL on (logits, labels)."""
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        scaled = logits / T.clamp(min=1e-3)
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.item())


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """ECE on the same harness as the EDL eval (pad with 0 unknown channel)."""
    probs = torch.softmax(logits, dim=-1)
    probs_padded = torch.cat([probs, torch.zeros(probs.shape[0], 1)], dim=-1)
    ece = expected_calibration_error(probs_padded, labels, n_bins=15, ignore_index=-100)
    return float(ece.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=Path("weights/w3s_ce_lite_v2.pt"))
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("artifacts/p1b7_temperature_scaling.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"==== P1-B7 R2-reimpl temperature scaling ====")
    print(f"ckpt: {args.ckpt}\n", flush=True)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNet2Lite_v2(in_channels=4, num_classes=C, k=16).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    print(f"Loaded W3-S (CE-lite_v2): ep={state['epoch']}, "
          f"val_mIoU={state['val_miou']:.4f}, val_ECE={state['val_ece']:.4f}",
          flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    val_ds = build_val(root, n_val=100)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)
    print(f"val: {len(val_ds)} frames\n", flush=True)

    print("Collecting logits...", flush=True)
    logits, labels = collect_logits_labels(model, val_loader, device, args.n_points)
    print(f"  {logits.shape[0]} valid points\n", flush=True)

    # ECE pre-TS
    print("Computing ECE pre-TS (baseline)...", flush=True)
    ece_pre = compute_ece(logits, labels, C)
    print(f"  ECE pre-TS: {ece_pre:.4f}")

    # Fit temperature on the same val set (held-out TS would be cleaner; this is a
    # standard upper-bound test — if even oracle TS doesn't close the gap, EDL wins)
    print("\nFitting temperature on val set (LBFGS)...", flush=True)
    T = fit_temperature(logits, labels)
    print(f"  T* = {T:.4f}")

    # ECE post-TS
    ece_post = compute_ece(logits / T, labels, C)
    print(f"  ECE post-TS: {ece_post:.4f}")
    print()

    rel_improvement = (ece_pre - ece_post) / max(1e-9, ece_pre) * 100
    print(f"Temperature scaling: {ece_pre:.4f} -> {ece_post:.4f} ({rel_improvement:+.1f}% relative)")

    # Compare against EDL baselines
    print()
    print("=== Comparison to EDL baselines (from W3-M and W3-R) ===")
    print(f"  W3-S (CE-lite_v2) pre-TS:   ECE = {ece_pre:.4f}")
    print(f"  W3-S (CE-lite_v2) post-TS:  ECE = {ece_post:.4f}  <-- fair comparison")
    print(f"  W3-M (EDL-lite_v2):         ECE = 0.1252  (R2 W3 unsolved confound)")
    print(f"  W3-R (EDL-lite_v2 W3-R):    ECE = 0.3197  (joint-serving, high AUROC floor)")
    print()
    if ece_post < 0.1252:
        print(f"VERDICT: Post-TS CE (ECE {ece_post:.4f}) BEATS EDL at lite_v2 tier "
              f"(ECE 0.1252). The §III.B 'EDL trades small mIoU for big ECE win' "
              f"thesis is FALSIFIED at the lite_v2 tier under fair-baseline comparison.")
    else:
        print(f"VERDICT: Post-TS CE (ECE {ece_post:.4f}) does not beat EDL at lite_v2 tier "
              f"(ECE 0.1252). The §III.B ECE-win thesis SURVIVES the temperature-scaling check.")

    out = {
        "ckpt": str(args.ckpt),
        "ckpt_epoch": int(state.get("epoch", -1)),
        "ckpt_val_mIoU": float(state.get("val_miou", 0)),
        "n_val_points": int(logits.shape[0]),
        "ECE_pre_TS": ece_pre,
        "T_star": T,
        "ECE_post_TS": ece_post,
        "rel_improvement_pct": rel_improvement,
        "EDL_baseline_W3M_ECE": 0.1252,
        "EDL_baseline_W3R_ECE": 0.3197,
        "post_TS_beats_W3M": ece_post < 0.1252,
        "post_TS_beats_W3R": ece_post < 0.3197,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
