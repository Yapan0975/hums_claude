#!/usr/bin/env python
"""W3-F EDL PointNet train on OFFICIAL SemanticKITTI splits.

Train sequences: 00, 01, 02, 03, 04, 05, 06, 07, 09, 10 (sampled).
Val sequence:    08 (first 100 frames — only subset transferred to server).

This is the official SemanticKITTI evaluation protocol; results here are
directly comparable to all published baselines (Cylinder3D 67.8, ConvBKI
65.6, etc.) for the PointNet-Vanilla capacity tier. The single-sequence
W3-A/E numbers were spatially-correlated train/val splits; this script
runs the proper protocol.

Default: sample 300 frames per train sequence (2 700 train + 100 val) to
keep wall-clock under 20 min on a 5090. Use --train-per-seq 0 for full
training (15+ k frames, ~5 hour epoch).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import PointNetEDL, _collate, _eval, _make_xyzi


TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQ = "08"


def build_train(root: Path, per_seq: int, seed: int = 0) -> torch.utils.data.Dataset:
    """Build a concat dataset of sampled frames across train sequences."""
    rng = random.Random(seed)
    parts: list[torch.utils.data.Dataset] = []
    counts: list[tuple[str, int]] = []
    for seq in TRAIN_SEQS:
        ds_seq = SemanticKITTIDataset(root=root, sequences=[seq], synthetic=False)
        n = len(ds_seq)
        if per_seq <= 0 or per_seq >= n:
            chosen = list(range(n))
        else:
            chosen = rng.sample(range(n), per_seq)
        parts.append(Subset(ds_seq, chosen))
        counts.append((seq, len(chosen)))
    full = ConcatDataset(parts)
    print(f"train: {len(full)} frames total")
    for seq, c in counts:
        print(f"  seq {seq}: {c}")
    return full


def build_val(root: Path, n_val: int = 100) -> torch.utils.data.Dataset:
    ds = SemanticKITTIDataset(root=root, sequences=[VAL_SEQ], synthetic=False)
    n_avail = len(ds)
    n_take = min(n_val, n_avail)
    print(f"val:   {n_take} / {n_avail} frames from seq {VAL_SEQ}")
    return Subset(ds, list(range(n_take)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=300,
                    help="frames per training seq (0 = full = ~5 hr/epoch on 5090)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("weights/pointnet_edl_multiseq.pt"))
    ap.add_argument("--backbone", default="vanilla",
                    choices=["vanilla", "lite"],
                    help="vanilla = PointNetVanilla 0.21M, lite = PointNet2Lite 0.28M (with kNN context)")
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free)")
    print()

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetEDL(in_channels=4, num_classes=C, backbone=args.backbone).to(device)
    print(f"\nbackbone: {args.backbone}, params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_ds)
    )
    loss_fn = EDLLoss(
        num_classes_plus_one=C + 1,
        kl_anneal_epochs=args.epochs,
        kl_lambda_start=0.0,
        kl_lambda_end=args.kl_end,
        ignore_index=-100,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t_total = time.perf_counter()
    print(f"\n=== Train {args.epochs} epochs on official SemKITTI splits ===")
    print(f"kl_lambda_end = {args.kl_end}, lr = {args.lr}\n")

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
        metric = _eval(model, val_loader, device, num_classes=C)
        miou_v = metric["miou"]
        ece_v = metric["ece"]
        dt = time.perf_counter() - t_ep
        msg = (f"ep {ep+1}/{args.epochs}: loss={ep_loss:.4f} "
               f"val_miou={miou_v:.4f} val_ece={ece_v:.4f} time={dt:.1f}s")
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
                    "train_per_seq": args.train_per_seq,
                    "train_seqs": TRAIN_SEQS,
                    "val_seq": VAL_SEQ,
                },
                args.out,
            )
            msg += "  [SAVED]"
        print(f"  {msg}", flush=True)

    total = time.perf_counter() - t_total
    print(f"\n=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")
    print(f"ckpt: {args.out.resolve()}")


if __name__ == "__main__":
    main()
