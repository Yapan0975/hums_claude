#!/usr/bin/env python
"""W3-A: Train PointNet with EDL (evidential) loss on SemKITTI seq 08.

Mirrors :mod:`scripts.train_pointnet_seq08` but the model outputs evidence
(softplus) → α and trains with the Sensoy 2018 EDL loss instead of vanilla
CE. The resulting checkpoint feeds the M1 evidential head in the RQ1
comparison, giving a *fair* CE-vs-EDL backbone comparison on the same data.

Usage on the 5090 server::

    cd ~/Documents/yping/mapping/code/evidlife_map
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/train_pointnet_edl.py \
        --root /data/shared/SemanticKITTI \
        --epochs 10 \
        --out weights/pointnet_edl_seq08.pt
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
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.m1_evidential.edl_head import (
    EDLHead,
    dirichlet_mean,
    vacuity_from_alpha,
)
from evidlife_map.m1_evidential.loss import EDLLoss
from evidlife_map.models.pointnet_vanilla import PointNetVanilla


def _collate(batch):
    assert len(batch) == 1
    return batch[0]


def _make_xyzi(entry, *, device, range_m: float = 100.0) -> torch.Tensor:
    xyz = entry.points_xyz_m.to(device, dtype=torch.float32)
    inten = entry.intensities.to(device, dtype=torch.float32).unsqueeze(-1)
    return torch.cat([xyz / range_m, inten], dim=-1)


class PointNetEDL(nn.Module):
    """PointNet feature extractor + Dirichlet evidential head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        backbone: str = "vanilla",
    ) -> None:
        super().__init__()
        if backbone == "vanilla":
            self.backbone = PointNetVanilla(
                in_channels=in_channels, num_classes=num_classes
            )
        elif backbone == "lite":
            from evidlife_map.models.pointnet2_lite import PointNet2Lite
            self.backbone = PointNet2Lite(
                in_channels=in_channels, num_classes=num_classes,
                k=16, sigma=1.0,
            )
        else:
            raise ValueError(f"unknown backbone {backbone!r}; "
                             f"must be 'vanilla' or 'lite'")
        self.backbone_name = backbone
        # The backbone's classifier outputs (N, num_classes) logits.
        # We feed those into the EDL head which internally adds +1 for unknown.
        self.edl_head = EDLHead(
            in_features=num_classes, num_classes=num_classes,
            activation="softplus", evidence_max=50.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(alpha, vacuity)`` with alpha shape ``(N, C+1)``."""
        feat_logits = self.backbone(x)  # (N, C)
        alpha, vac = self.edl_head(feat_logits)  # (N, C+1), (N,)
        return alpha, vac


def _eval(model: PointNetEDL, loader, device, *, num_classes: int) -> dict:
    model.eval()
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    sum_ece = 0.0
    n_ece = 0
    with torch.no_grad():
        for entry in loader:
            x = _make_xyzi(entry, device=device)
            alpha, _ = model(x)
            # Closed-set IoU: drop unknown channel before argmax
            closed = alpha[:, :num_classes]
            pred = closed.argmax(dim=-1)
            iou.update(pred, entry.labels.to(device))
            # ECE on the (C+1)-class evidential distribution
            ece = expected_calibration_error(alpha.cpu(), entry.labels.cpu(),
                                              n_bins=15, ignore_index=-100)
            sum_ece += float(ece.item())
            n_ece += 1
    metric = iou.compute()
    metric["ece"] = sum_ece / max(1, n_ece)
    return metric


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("weights/pointnet_edl_seq08.pt"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[0]/1024**3:.1f} GB free)")

    root = args.root
    if (root / "dataset" / "sequences").exists():
        root = root / "dataset"
    full = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    n_total = len(full)
    n_train = min(args.n_train, n_total - args.n_val)
    n_val = min(args.n_val, n_total - n_train)
    train_loader = DataLoader(Subset(full, list(range(n_train))),
                              batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(full, list(range(n_total - n_val, n_total))),
                            batch_size=1, shuffle=False, collate_fn=_collate)
    print(f"split: train={n_train} val={n_val}")

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetEDL(in_channels=4, num_classes=C).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_train)
    loss_fn = EDLLoss(num_classes_plus_one=C + 1,
                       kl_anneal_epochs=args.epochs,
                       kl_lambda_start=0.0,
                       kl_lambda_end=1.0,
                       ignore_index=-100)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    t_total = time.perf_counter()
    print()
    print("=== Train PointNet+EDL ===")
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        t_ep = time.perf_counter()
        for entry in train_loader:
            y_cpu = entry.labels
            # Skip frames with no valid labels (avoids no-grad zero from EDLLoss)
            if not bool((y_cpu >= 0).any()):
                continue
            x = _make_xyzi(entry, device=device)
            y = y_cpu.to(device)
            alpha, _ = model(x)
            loss_dict = loss_fn(alpha, y, epoch=ep)
            loss = loss_dict["loss"]
            if not loss.requires_grad:
                # Defensive: if for any reason loss has no grad, skip this frame
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
        ep_loss /= max(1, len(train_loader))
        metric = _eval(model, val_loader, device, num_classes=C)
        dt = time.perf_counter() - t_ep
        msg = (f"epoch {ep+1}/{args.epochs}: loss={ep_loss:.4f} "
               f"val_miou={metric['miou']:.4f} val_ece={metric['ece']:.4f} time={dt:.1f}s")
        if metric["miou"] > best_miou:
            best_miou = metric["miou"]
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": C,
                "epoch": ep + 1,
                "val_miou": metric["miou"],
                "val_ece": metric["ece"],
            }, args.out)
            msg += f"  [SAVED -> {args.out}]"
        print(f"  {msg}")

    total = time.perf_counter() - t_total
    print()
    print(f"=== DONE — best val mIoU = {best_miou:.4f}, total {total:.1f}s ===")
    print(f"ckpt: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
