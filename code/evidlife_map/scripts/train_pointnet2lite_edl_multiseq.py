#!/usr/bin/env python
"""W3-H EDL PointNet2Lite train with point subsampling.

The previous attempt at training the lite backbone (k-NN context) on full
~120k-point frames was hitting the chunked-cdist wall: each epoch took
~20 min and 20 epochs was a full day. Since the kNN computation is O(N²)
in points-per-frame and the rest of the network is O(N), random sub-
sampling each frame to ``--n-points`` points reduces the wall-clock
quadratically. Empirically 20 000 points is sufficient to keep
geometric structure (SemKITTI frames have ~120k points; 20k is ~1 of
every 6 points which is roughly KITTI's published "voxelised" working
size).

Default: 20 000 points/frame, 20 epochs, 300 frames per train seq.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import PointNetEDL, _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val


def _subsample(entry, n_points: int, *, generator: torch.Generator | None = None):
    """Return a copy of ``entry`` with ``points_xyz_m`` / ``intensities`` /
    ``labels`` randomly subsampled to at most ``n_points`` rows."""
    n = entry.points_xyz_m.shape[0]
    if n <= n_points:
        return entry
    if generator is None:
        idx = torch.randperm(n)[:n_points]
    else:
        idx = torch.randperm(n, generator=generator)[:n_points]
    out = type(entry)(
        points_xyz_m=entry.points_xyz_m[idx],
        intensities=entry.intensities[idx],
        labels=entry.labels[idx],
        frame_idx=entry.frame_idx,
        sequence=entry.sequence,
    )
    return out


def _eval(model, loader, device, num_classes, n_points=20000) -> dict:
    model.eval()
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    sum_ece, n_ece = 0.0, 0
    with torch.no_grad():
        for entry in loader:
            entry = _subsample(entry, n_points)
            x = _make_xyzi(entry, device=device)
            alpha, _ = model(x)
            closed = alpha[:, :num_classes]
            pred = closed.argmax(dim=-1)
            iou.update(pred, entry.labels.to(device))
            ece = expected_calibration_error(
                alpha.cpu(), entry.labels.cpu(), n_bins=15, ignore_index=-100
            )
            sum_ece += float(ece.item())
            n_ece += 1
    out = iou.compute()
    out["ece"] = sum_ece / max(1, n_ece)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("weights/pointnet2lite_edl_multiseq.pt"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free)")
    print(f"n_points: {args.n_points} (subsample per frame)\n", flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetEDL(in_channels=4, num_classes=C, backbone="lite").to(device)
    print(f"backbone: lite, params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M",
          flush=True)

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
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t_total = time.perf_counter()
    print(f"\n=== Train lite {args.epochs} epochs on official SemKITTI splits ===")
    print(f"kl_lambda_end = {args.kl_end}, lr = {args.lr}\n", flush=True)
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
                    "n_points": args.n_points,
                    "backbone": "lite",
                },
                args.out,
            )
            msg += "  [SAVED]"
        print(f"  {msg}", flush=True)
    total = time.perf_counter() - t_total
    print(f"\n=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")


if __name__ == "__main__":
    main()
