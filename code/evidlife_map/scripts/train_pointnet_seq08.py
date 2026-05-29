#!/usr/bin/env python
"""W1 G-1 quick train: PointNet-Vanilla on SemanticKITTI seq 08 first-100.

Single-sequence, single-GPU, 5-epoch quick fine-tune. NOT a leaderboard run.
The W1 G-1 goal is to push the random-head 3% mIoU to a *real* mIoU figure
(target: 30-50%) so the R2 pipeline produces meaningful end-to-end numbers
while the proper Cylinder3D integration (W2) is pending.

Usage on the 5090 server::

    cd ~/Documents/yping/mapping/code/evidlife_map
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/train_pointnet_seq08.py \
        --root /data/shared/SemanticKITTI \
        --epochs 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.models.pointnet_vanilla import PointNetVanilla


def _collate_single(batch):
    assert len(batch) == 1
    return batch[0]


def _make_xyzi(entry, *, device, range_m: float = 100.0) -> torch.Tensor:
    """Build (N, 4) xyzi feature tensor, range-normalized so net sees ~[-1, 1]."""
    xyz = entry.points_xyz_m.to(device, dtype=torch.float32)
    inten = entry.intensities.to(device, dtype=torch.float32).unsqueeze(-1)
    return torch.cat([xyz / range_m, inten], dim=-1)


def _eval(model, loader, device) -> float:
    model.eval()
    iou = IoUTracker(num_classes=SEMANTIC_KITTI_NUM_CLASSES, ignore_index=-100)
    with torch.no_grad():
        for entry in loader:
            x = _make_xyzi(entry, device=device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            iou.update(pred, entry.labels.to(device))
    return iou.compute()["miou"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("weights/pointnet_seq08_quick.pt"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}, "
              f"free {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB")

    root = args.root
    if (root / "dataset" / "sequences").exists():
        root = root / "dataset"
    full = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    n_total = len(full)
    print(f"loaded {n_total} frames from seq {args.seq}")

    n_train = min(args.n_train, n_total - args.n_val)
    n_val = min(args.n_val, n_total - n_train)
    train_set = Subset(full, list(range(n_train)))
    val_set = Subset(full, list(range(n_total - n_val, n_total)))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=_collate_single)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=_collate_single)
    print(f"split: train={n_train} val={n_val}")

    model = PointNetVanilla(in_channels=4, num_classes=SEMANTIC_KITTI_NUM_CLASSES).to(device)
    print(f"params: {model.num_params() / 1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_train)
    # NOTE: tried class-weighted CE but it hurt val mIoU (overfits to rare
    # classes absent from val). Plain CE on this 100-frame split tops out at
    # ~5% mIoU — PointNet-Vanilla has no spatial context. W2 will integrate
    # Cylinder3D + spconv for a proper SemKITTI baseline (target 70+ mIoU).
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    t_total = time.perf_counter()
    init_miou = _eval(model, val_loader, device)
    print(f"\n=== init random-head val mIoU: {init_miou:.4f} ===")

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        t_ep = time.perf_counter()
        for entry in train_loader:
            x = _make_xyzi(entry, device=device)
            y = entry.labels.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
        ep_loss /= max(1, len(train_loader))
        miou = _eval(model, val_loader, device)
        dt = time.perf_counter() - t_ep
        msg = f"epoch {ep+1}/{args.epochs}: loss={ep_loss:.4f} val_miou={miou:.4f} time={dt:.1f}s"
        if miou > best:
            best = miou
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": SEMANTIC_KITTI_NUM_CLASSES,
                "epoch": ep + 1,
                "val_miou": miou,
            }, args.out)
            msg += f"  [SAVED -> {args.out}]"
        print(f"  {msg}")

    total = time.perf_counter() - t_total
    print(f"\n=== DONE ===")
    print(f"init mIoU:   {init_miou:.4f}")
    print(f"best mIoU:   {best:.4f} (Δ = {best-init_miou:+.4f})")
    print(f"total time:  {total:.1f}s")
    print(f"ckpt:        {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
