#!/usr/bin/env python
"""W3-M Path 4 backbone + warm-restart + bigger train (best-of-all).

Combines the three top knobs from the W3-J/K/L ablation campaign:
- backbone: PointNet2Lite_v2 (0.49 M, 3 LSE+AP blocks) — W3-L winner
- KL schedule: warm-restart, period=5, kl_end=0.3 — W3-J winner
- train data: 600 frames per sequence (≈ 5 670 train) — W3-K config
- epochs: 25 (longer for the larger backbone)

Target: push the §IV.0 mIoU floor past 22 % (current best 18.61 % from
W3-J at PointNet2Lite vanilla).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample
from scripts.train_pointnet2lite_v2_edl_multiseq import (
    PointNetLiteV2EDL, _eval,
)
from scripts.train_pointnet2lite_edl_warmrestart import EDLLossWarmRestart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--restart-period", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("weights/pointnet2lite_v2_warmrestart_bigger.pt"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"n_points: {args.n_points}, k: {args.k}, "
          f"kl_end: {args.kl_end}, restart_period: {args.restart_period}\n",
          flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetLiteV2EDL(in_channels=4, num_classes=C, k=args.k).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"backbone: lite_v2, params: {n_params:.3f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_ds)
    )
    loss_fn = EDLLossWarmRestart(
        num_classes_plus_one=C + 1,
        kl_anneal_epochs=args.epochs,
        kl_lambda_start=0.0,
        kl_lambda_end=args.kl_end,
        ignore_index=-100,
        restart_period=args.restart_period,
    )
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t_total = time.perf_counter()
    print(f"\n=== Train lite_v2 + warm-restart + bigger, "
          f"{args.epochs} epochs ===\n", flush=True)
    for ep in range(args.epochs):
        model.train()
        ep_loss, n_step = 0.0, 0
        t_ep = time.perf_counter()
        lam_print = loss_fn._current_lambda(ep)
        for entry in train_loader:
            y_cpu = entry.labels
            if not bool((y_cpu >= 0).any()):
                continue
            entry = _subsample(entry, args.n_points, generator=gen)
            x = _make_xyzi(entry, device=device)
            y = entry.labels.to(device)
            alpha, _ = model(x)
            loss_dict = loss_fn(alpha, y, epoch=ep)
            loss = loss_dict["loss"]
            if not loss.requires_grad:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
            n_step += 1
        ep_loss /= max(1, n_step)
        metric = _eval(model, val_loader, device, num_classes=C, n_points=args.n_points)
        miou_v, ece_v = metric["miou"], metric["ece"]
        dt = time.perf_counter() - t_ep
        msg = (f"ep {ep+1}/{args.epochs}: lam={lam_print:.3f} "
               f"loss={ep_loss:.4f} val_miou={miou_v:.4f} "
               f"val_ece={ece_v:.4f} time={dt:.1f}s")
        if miou_v > best_miou:
            best_miou = miou_v
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": C,
                    "epoch": ep + 1,
                    "val_miou": miou_v,
                    "val_ece": ece_v,
                    "kl_end": args.kl_end,
                    "restart_period": args.restart_period,
                    "n_points": args.n_points,
                    "k": args.k,
                    "train_per_seq": args.train_per_seq,
                    "backbone": "lite_v2",
                },
                args.out,
            )
            msg += "  [SAVED]"
        print(f"  {msg}", flush=True)
    total = time.perf_counter() - t_total
    print(f"\n=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")


if __name__ == "__main__":
    main()
