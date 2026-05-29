#!/usr/bin/env python
"""W3-J PointNet2Lite EDL with KL warm-restart schedule.

Addresses §V.B(d) failure mode: under the default linear KL ramp,
both EDL-vanilla and EDL-lite peak in the first ~3 epochs and then
plateau slightly below the peak as KL pulls the head toward uniform
Dirichlet.

Warm-restart schedule (Loshchilov 2017 in spirit, adapted to KL):

    λ(epoch) = kl_end * ((epoch mod restart_period) / restart_period)

So λ ramps from 0 → kl_end over each restart_period epochs, then
snaps back to 0 and ramps again. This periodically gives the head a
"breathing" interval to recover discriminative signal before the
next KL push.

Default: restart_period=5, kl_end=0.3, 20 epochs (so 4 cycles),
20 000 point subsample, 300 frames per train seq.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import PointNetEDL, _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val
from scripts.train_pointnet2lite_edl_multiseq import _eval, _subsample


class EDLLossWarmRestart(EDLLoss):
    """EDLLoss with periodic warm-restart of the KL annealing weight."""

    def __init__(self, *args, restart_period: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.restart_period = max(1, int(restart_period))

    def _current_lambda(self, epoch: int) -> float:
        # Within each restart cycle, ramp from start to end.
        cycle_pos = epoch % self.restart_period
        frac = float(cycle_pos) / float(self.restart_period)
        return self.kl_lambda_start + frac * (self.kl_lambda_end - self.kl_lambda_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--restart-period", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("weights/pointnet2lite_edl_warmrestart.pt"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free)")
    print(f"n_points: {args.n_points}, kl_end: {args.kl_end}, "
          f"restart_period: {args.restart_period}\n", flush=True)

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
    print(f"\n=== Train lite warm-restart, period={args.restart_period}, "
          f"kl_end={args.kl_end}, lr={args.lr}, {args.epochs} epochs ===\n", flush=True)
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
        miou_v = metric["miou"]
        ece_v = metric["ece"]
        dt = time.perf_counter() - t_ep
        msg = (f"ep {ep+1}/{args.epochs}: lam={lam_print:.3f} "
               f"loss={ep_loss:.4f} val_miou={miou_v:.4f} val_ece={ece_v:.4f} "
               f"time={dt:.1f}s")
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
