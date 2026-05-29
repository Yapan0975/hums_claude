#!/usr/bin/env python
"""W3-L EDL PointNet2Lite-v2 (Path 4 backbone) multi-seq train.

Stacks 3 LSE + Attentive-Pooling blocks at fixed N points (no random
downsampling) for a small RandLA-Net-class backbone (~0.6 M params).
Target: lift the §IV.0 lite mIoU floor from 18.65 % to ~25-35 %.

Trains EDL (Dirichlet head) on the same official SemKITTI multi-seq
split as W3-F/G/H, with the same 20 000-point subsample as W3-H/J/K.
Default: 600 frames per training seq, 20 epochs, kl_end 0.3.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import SEMANTIC_KITTI_NUM_CLASSES
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.m1_evidential.edl_head import EDLHead
from evidlife_map.m1_evidential.loss import EDLLoss
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.models.pointnet2_lite_v2 import PointNet2Lite_v2
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample


class PointNetLiteV2EDL(nn.Module):
    """PointNet2Lite_v2 + EDL head."""

    def __init__(self, in_channels: int, num_classes: int, *,
                 k: int = 16, channels=(64, 128, 256, 256)) -> None:
        super().__init__()
        self.backbone = PointNet2Lite_v2(
            in_channels=in_channels, num_classes=num_classes,
            k=k, channels=channels,
        )
        self.edl_head = EDLHead(
            in_features=num_classes, num_classes=num_classes,
            activation="softplus", evidence_max=50.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_logits = self.backbone(x)
        alpha, vac = self.edl_head(feat_logits)
        return alpha, vac


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
    ap.add_argument("--train-per-seq", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("weights/pointnet2lite_v2_edl_multiseq.pt"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free)")
    print(f"n_points: {args.n_points}, k: {args.k}, kl_end: {args.kl_end}\n",
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
    print(f"\n=== Train lite_v2 {args.epochs} epochs, kl_end={args.kl_end} ===\n",
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
                    "k": args.k,
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
