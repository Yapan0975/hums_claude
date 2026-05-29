#!/usr/bin/env python
"""W3-E EDL PointNet train with gentler KL annealing (max λ = 0.3).

The default schedule (λ → 1.0 over 5 epochs) caused vacuity AUROC to collapse
after epoch 1 in the §IV.0 preliminary RQ2 run. This script uses λ → 0.3 over
20 epochs as a hand-tuned mitigation; the proper warm-restart schedule
referenced in §V.B is pre-registered for the next revision.

Usage::

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/train_pointnet_edl_kl03.py \
        --root /data/shared/SemanticKITTI \
        --seq 00 --n-train 4000 --n-val 100 --epochs 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import (
    PointNetEDL,
    _collate,
    _eval,
    _make_xyzi,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--kl-end", type=float, default=0.3,
                    help="Max λ at end of anneal schedule (paper default 1.0; 0.3 = gentle)")
    ap.add_argument("--out", type=Path, default=Path("weights/pointnet_edl_kl03.pt"))
    args = ap.parse_args()

    device = torch.device("cuda")
    C = SEMANTIC_KITTI_NUM_CLASSES

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    full = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    n_train = min(args.n_train, len(full) - args.n_val)
    n_val = min(args.n_val, len(full) - n_train)
    train_loader = DataLoader(Subset(full, list(range(n_train))),
                              batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(full, list(range(len(full) - n_val, len(full)))),
                            batch_size=1, shuffle=False, collate_fn=_collate)
    print(f"seq {args.seq} | train={n_train} val={n_val} | epochs={args.epochs} | kl_end={args.kl_end}")

    model = PointNetEDL(in_channels=4, num_classes=C).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_train)
    loss_fn = EDLLoss(num_classes_plus_one=C + 1,
                      kl_anneal_epochs=args.epochs,
                      kl_lambda_start=0.0,
                      kl_lambda_end=args.kl_end,
                      ignore_index=-100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        ep_loss, n_step = 0.0, 0
        for entry in train_loader:
            y_cpu = entry.labels
            if not bool((y_cpu >= 0).any()):
                continue
            x = _make_xyzi(entry, device=device)
            y = y_cpu.to(device)
            alpha, _ = model(x)
            loss_dict = loss_fn(alpha, y, epoch=ep)
            loss = loss_dict["loss"]
            if not loss.requires_grad:
                continue
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss += float(loss.item()); n_step += 1
        ep_loss /= max(1, n_step)
        metric = _eval(model, val_loader, device, num_classes=C)
        miou_v = metric["miou"]
        ece_v = metric["ece"]
        msg = f"ep {ep+1}/{args.epochs}: loss={ep_loss:.4f} val_miou={miou_v:.4f} val_ece={ece_v:.4f}"
        if miou_v > best_miou:
            best_miou = miou_v
            torch.save({"state_dict": model.state_dict(),
                        "num_classes": C, "epoch": ep + 1,
                        "val_miou": miou_v, "val_ece": ece_v,
                        "kl_end": args.kl_end},
                       args.out)
            msg += "  [SAVED]"
        print(f"  {msg}")
    total = time.perf_counter() - t0
    print(f"\n=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")
    print(f"ckpt: {args.out.resolve()}")


if __name__ == "__main__":
    main()
