#!/usr/bin/env python
"""W3-I add-on: per-class IoU dump for the best lite checkpoint.

Loads ``weights/pointnet2lite_edl_multiseq.pt`` (or alternative via --ckpt),
runs the closed-set argmax on seq 08 val 100 frames, and dumps per-class IoU
into a JSON for §IV.C commentary.

Output goes to ``artifacts/per_class_iou_lite.json`` with structure:

    {
      "ckpt": "...", "params_M": 0.276,
      "miou": 0.179, "ece": 0.096,
      "per_class": [{"id": 0, "name": "car", "iou": 0.42, "support": 1234}, ...]
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from scripts.train_pointnet_edl import PointNetEDL, _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample


# SemanticKITTI 19 learning-class names (zero-based, post-remap).
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist", "road", "parking",
    "sidewalk", "other-ground", "building", "fence", "vegetation",
    "trunk", "terrain", "pole", "traffic-sign",
]
assert len(CLASS_NAMES) == 19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("weights/pointnet2lite_edl_multiseq.pt"))
    ap.add_argument("--backbone", default="lite", choices=["vanilla", "lite"])
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/per_class_iou_lite.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetEDL(in_channels=4, num_classes=C, backbone=args.backbone)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    print(f"loaded {args.ckpt} (ep {state.get('epoch')}, mIoU {state.get('val_miou'):.4f})",
          flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    val_ds = build_val(root, n_val=100)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    iou_tracker = IoUTracker(num_classes=C, ignore_index=-100)
    label_support = torch.zeros(C, dtype=torch.int64)
    sum_ece, n_ece = 0.0, 0

    with torch.no_grad():
        for entry in loader:
            entry = _subsample(entry, args.n_points)
            x = _make_xyzi(entry, device=device)
            alpha, _ = model(x)
            closed = alpha[:, :C]
            pred = closed.argmax(dim=-1)
            iou_tracker.update(pred, entry.labels.to(device))
            # tally support per class
            valid = entry.labels >= 0
            for c in range(C):
                label_support[c] += int((entry.labels[valid] == c).sum().item())
            # ECE
            ece = expected_calibration_error(
                alpha.cpu(), entry.labels.cpu(), n_bins=15, ignore_index=-100
            )
            sum_ece += float(ece.item())
            n_ece += 1

    metric = iou_tracker.compute()
    per_class_iou = metric["per_class_iou"]
    miou = metric["miou"]
    ece_v = sum_ece / max(1, n_ece)

    rows = []
    for c in range(C):
        rows.append({
            "id": c,
            "name": CLASS_NAMES[c],
            "iou": float(per_class_iou[c]) if per_class_iou[c] is not None else None,
            "support": int(label_support[c].item()),
        })

    out = {
        "ckpt": str(args.ckpt),
        "backbone": args.backbone,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "ckpt_epoch": int(state.get("epoch", -1)),
        "miou": round(miou, 4),
        "ece": round(ece_v, 4),
        "per_class": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Per-class IoU ({args.backbone}, ep {state.get('epoch')}) ===")
    print(f"{'cid':>3s}  {'name':16s}  {'IoU':>8s}  {'support':>10s}")
    print("-" * 50)
    for r in rows:
        iou_s = f"{r['iou']:.4f}" if r['iou'] is not None else " (absent)"
        print(f"{r['id']:>3d}  {r['name']:16s}  {iou_s:>8s}  {r['support']:>10d}")
    print(f"\n  mIoU: {miou:.4f}   ECE: {ece_v:.4f}")
    print(f"  wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
