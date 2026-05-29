#!/usr/bin/env python
"""W3-G CE PointNet train on OFFICIAL SemanticKITTI splits.

Companion to ``train_pointnet_edl_multiseq.py`` — same protocol, same
backbone, but plain cross-entropy loss instead of EDL. Used as a control
for the §IV.0 RQ1 comparison so the M1 vs R2 row is apples-to-apples.

Default: 300 frames per train sequence (≈ 2 700 train + 100 val);
20 epochs ≈ 16 min on a 5090.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.models.pointnet_vanilla import PointNetVanilla
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val


def _eval_ce(model, loader, device, num_classes) -> dict:
    model.eval()
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    with torch.no_grad():
        for entry in loader:
            x = _make_xyzi(entry, device=device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            iou.update(pred, entry.labels.to(device))
    return iou.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vanilla", choices=["vanilla", "lite"])
    ap.add_argument("--out", type=Path,
                    default=Path("weights/pointnet_ce_multiseq.pt"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    if args.backbone == "vanilla":
        model = PointNetVanilla(in_channels=4, num_classes=C).to(device)
    else:
        from evidlife_map.models.pointnet2_lite import PointNet2Lite
        model = PointNet2Lite(in_channels=4, num_classes=C, k=16).to(device)
    print(f"backbone: {args.backbone}, params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_ds)
    )
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t_total = time.perf_counter()
    print(f"\n=== Train CE on official SemKITTI splits ===\n")
    for ep in range(args.epochs):
        model.train()
        ep_loss, n_step = 0.0, 0
        t_ep = time.perf_counter()
        for entry in train_loader:
            y_cpu = entry.labels
            if not bool((y_cpu >= 0).any()):
                continue
            x = _make_xyzi(entry, device=device)
            y = y_cpu.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
            n_step += 1
        ep_loss /= max(1, n_step)
        metric = _eval_ce(model, val_loader, device, num_classes=C)
        miou_v = metric["miou"]
        dt = time.perf_counter() - t_ep
        msg = (f"ep {ep+1}/{args.epochs}: loss={ep_loss:.4f} "
               f"val_miou={miou_v:.4f} time={dt:.1f}s")
        if miou_v > best_miou:
            best_miou = miou_v
            torch.save(
                {"state_dict": model.state_dict(), "num_classes": C,
                 "epoch": ep + 1, "val_miou": miou_v,
                 "backbone": args.backbone, "train_per_seq": args.train_per_seq},
                args.out,
            )
            msg += "  [SAVED]"
        print(f"  {msg}", flush=True)
    total = time.perf_counter() - t_total
    print(f"\n=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")


if __name__ == "__main__":
    main()
