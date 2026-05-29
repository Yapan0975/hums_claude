#!/usr/bin/env python
"""W3-I aggregate: print one comparison row per checkpoint.

Loads up to three multi-seq checkpoints:

  - weights/pointnet_ce_multiseq.pt         (CE PointNet-Vanilla)
  - weights/pointnet_edl_multiseq.pt        (EDL PointNet-Vanilla)
  - weights/pointnet_edl_multiseq_lite.pt   (EDL PointNet2Lite)

For each it loads the model, runs evaluation on the same val_ds (seq 08 first
100 frames, same as training scripts use), reports val mIoU + ECE in a single
table. Use this as the source for the paper §IV.0 multi-seq update.

Output goes to stdout in plain text and to ``artifacts/multiseq_compare.json``
in a machine-readable form.

Usage on the 5090 server::

    cd ~/Documents/yping/mapping/code/evidlife_map
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/compare_multiseq.py \\
        --root /data/shared/SemanticKITTI \\
        --out artifacts/multiseq_compare.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.models.pointnet_vanilla import PointNetVanilla
from scripts.train_pointnet_edl import PointNetEDL, _collate, _eval, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_val


def _load_edl(path: Path, *, backbone: str, num_classes: int, device) -> PointNetEDL:
    model = PointNetEDL(in_channels=4, num_classes=num_classes, backbone=backbone)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state


def _load_ce(path: Path, *, backbone: str, num_classes: int, device):
    if backbone == "vanilla":
        model = PointNetVanilla(in_channels=4, num_classes=num_classes)
    else:
        from evidlife_map.models.pointnet2_lite import PointNet2Lite
        model = PointNet2Lite(in_channels=4, num_classes=num_classes, k=16, sigma=1.0)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state


@torch.no_grad()
def _eval_ce(model, loader, device, num_classes) -> dict:
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    sum_ece, n = 0.0, 0
    for entry in loader:
        x = _make_xyzi(entry, device=device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        iou.update(pred, entry.labels.to(device))
        # Treat logits-derived softmax as a C-class "Dirichlet mean" surrogate
        # by appending a 0-mass unknown channel so ECE is comparable to the
        # EDL number (the unknown bin will simply contribute 0 calibration
        # mass for CE; this matches the §IV.0 §"closed-set ECE" definition).
        probs_padded = torch.cat([probs, torch.zeros(probs.shape[0], 1, device=device)], dim=-1)
        ece = expected_calibration_error(
            probs_padded.cpu(), entry.labels.cpu(), n_bins=15, ignore_index=-100
        )
        sum_ece += float(ece.item())
        n += 1
    out = iou.compute()
    out["ece"] = sum_ece / max(1, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("artifacts/multiseq_compare.json"))
    ap.add_argument("--ce-vanilla", type=Path,
                    default=Path("weights/pointnet_ce_multiseq.pt"))
    ap.add_argument("--edl-vanilla", type=Path,
                    default=Path("weights/pointnet_edl_multiseq.pt"))
    ap.add_argument("--edl-lite", type=Path,
                    default=Path("weights/pointnet_edl_multiseq_lite.pt"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    C = SEMANTIC_KITTI_NUM_CLASSES

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    val_ds = build_val(root, n_val=args.n_val)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)
    print(f"val set: {len(val_ds)} frames from seq 08\n")

    rows: list[dict] = []
    configs = [
        ("CE-vanilla",     args.ce_vanilla,  "ce",  "vanilla"),
        ("EDL-vanilla",    args.edl_vanilla, "edl", "vanilla"),
        ("EDL-lite",       args.edl_lite,    "edl", "lite"),
    ]
    for name, ckpt_path, loss_kind, backbone in configs:
        if not ckpt_path.exists():
            print(f"[skip] {name:14s} (no ckpt at {ckpt_path})")
            continue
        t0 = time.perf_counter()
        if loss_kind == "edl":
            model, state = _load_edl(ckpt_path, backbone=backbone, num_classes=C, device=device)
            metric = _eval(model, val_loader, device, num_classes=C)
        else:
            model, state = _load_ce(ckpt_path, backbone=backbone, num_classes=C, device=device)
            metric = _eval_ce(model, val_loader, device, num_classes=C)
        dt = time.perf_counter() - t0
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        row = {
            "name": name,
            "loss": loss_kind,
            "backbone": backbone,
            "params_M": round(params_m, 3),
            "best_ep_val_miou": round(float(state.get("val_miou", -1.0)), 4),
            "best_ep_val_ece": round(float(state.get("val_ece", -1.0)), 4) if "val_ece" in state else None,
            "fresh_eval_miou": round(metric["miou"], 4),
            "fresh_eval_ece": round(metric["ece"], 4),
            "epoch": int(state.get("epoch", -1)),
            "eval_s": round(dt, 1),
            "ckpt": str(ckpt_path),
        }
        rows.append(row)
        print(f"{name:14s} params={params_m:.3f}M  "
              f"fresh: mIoU={metric['miou']:.4f} ECE={metric['ece']:.4f}  "
              f"(best-ep mIoU in ckpt = {state.get('val_miou','?')})")

    # Pretty table
    print("\n=== Multi-seq comparison (official SemKITTI splits, val seq 08 100 frames) ===\n")
    print(f"{'system':16s} {'params':>8s}  {'mIoU':>8s}  {'ECE':>8s}  {'ep':>4s}")
    for r in rows:
        print(f"{r['name']:16s} {r['params_M']:>7.3f}M  "
              f"{r['fresh_eval_miou']:>8.4f}  {r['fresh_eval_ece']:>8.4f}  {r['epoch']:>4d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
