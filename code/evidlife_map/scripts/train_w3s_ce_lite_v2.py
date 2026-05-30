#!/usr/bin/env python
"""W3-S CE-lite_v2 baseline — matched-protocol R2 companion for W3-M.

Per round-1 R1 W1: the §VI 'matched-capacity 10× method win' is false
because W3-M (M1+EDL+lite_v2+WR+600fr+25ep) was compared against
W3-G (R2+CE+vanilla+linear+300fr+20ep) — 5 asymmetric knobs.

This script runs the missing R2 baseline at the W3-M tier:
  - Backbone: PointNet2Lite_v2 (0.49 M, same as W3-M)
  - Loss: CrossEntropyLoss (replaces EDL)
  - Train data: 600 fr/seq (same as W3-M)
  - Epochs: 25 (same as W3-M)
  - Schedule: cosine LR with warm-restart on the LR (mirroring W3-M's
    KL warm-restart in spirit — same restart period, different signal)
  - Subsample: 20 000 points/frame

For the joint comparison, both M1+W3-M and CE+W3-S will be evaluated
on the same val set with the same harness, producing the symmetric
4-row {CE, EDL} × {vanilla, lite_v2-WR-600-25} matrix that round-1
R1 W1 demands.

After this run finishes, the {CE-vanilla, EDL-vanilla, CE-lite_v2-bigger,
EDL-lite_v2-bigger} 4-row matrix is reportable. Then the §IV.C / §VI
'method effect at matched backbone tier' rewrites can use:
  - small backbone (vanilla):  CE = 2.28% vs EDL = 1.66%  (R2 slightly wins)
  - large backbone (lite_v2):  CE = W3-S% vs EDL = 23.32% (W3-M)
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
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.models.pointnet2_lite_v2 import PointNet2Lite_v2
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample


def _eval_ce(model, loader, device, num_classes, n_points=20000) -> dict:
    model.eval()
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    sum_ece, n_ece = 0.0, 0
    with torch.no_grad():
        for entry in loader:
            entry = _subsample(entry, n_points)
            x = _make_xyzi(entry, device=device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            iou.update(pred, entry.labels.to(device))
            # Pad with 0 unknown channel so ECE is comparable to EDL's (C+1)-class ECE
            probs_padded = torch.cat([probs, torch.zeros(probs.shape[0], 1, device=device)], dim=-1)
            ece = expected_calibration_error(
                probs_padded.cpu(), entry.labels.cpu(), n_bins=15, ignore_index=-100
            )
            sum_ece += float(ece.item())
            n_ece += 1
    out = iou.compute()
    out["ece"] = sum_ece / max(1, n_ece)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--restart-period", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("weights/w3s_ce_lite_v2.pt"))
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"==== W3-S CE-lite_v2 matched-protocol R2 baseline ====")
    print(f"Backbone: PointNet2Lite_v2 (matches W3-M)")
    print(f"Loss: CrossEntropyLoss (replaces EDL for the R2 baseline)")
    print(f"Schedule: cosine LR warm-restart (period={args.restart_period}, "
          f"mirroring W3-M KL warm-restart)")
    print(f"Train: {args.train_per_seq} fr/seq, {args.epochs} epochs, "
          f"{args.n_points} pts/frame subsample\n", flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNet2Lite_v2(in_channels=4, num_classes=C, k=args.k).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.3f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # CosineAnnealingWarmRestarts mirrors W3-M's KL warm-restart for the LR
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.restart_period * len(train_ds), T_mult=1,
    )
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t_total = time.perf_counter()
    print(f"\n=== Train CE-lite_v2 {args.epochs} epochs (matched W3-M tier) ===\n",
          flush=True)
    for ep in range(args.epochs):
        model.train()
        ep_loss, n_step = 0.0, 0
        t_ep = time.perf_counter()
        for entry in train_loader:
            y_cpu = entry.labels
            if not bool((y_cpu >= 0).any()):
                continue
            entry = _subsample(entry, args.n_points, generator=gen)
            x = _make_xyzi(entry, device=device)
            y = entry.labels.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
            n_step += 1
        ep_loss /= max(1, n_step)
        metric = _eval_ce(model, val_loader, device, num_classes=C, n_points=args.n_points)
        miou_v, ece_v = metric["miou"], metric["ece"]
        dt = time.perf_counter() - t_ep
        msg = (f"ep {ep+1}/{args.epochs}: loss={ep_loss:.4f} "
               f"val_miou={miou_v:.4f} val_ece={ece_v:.4f} time={dt:.1f}s")
        if miou_v > best_miou:
            best_miou = miou_v
            torch.save({
                "state_dict": model.state_dict(), "num_classes": C,
                "epoch": ep + 1, "val_miou": miou_v, "val_ece": ece_v,
                "backbone": "lite_v2", "loss": "CE",
                "train_per_seq": args.train_per_seq, "epochs": args.epochs,
                "schedule": "cosine warm-restart",
            }, args.out)
            msg += "  [SAVED]"
        print(f"  {msg}", flush=True)
    total = time.perf_counter() - t_total
    print(f"\n=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")


if __name__ == "__main__":
    main()
