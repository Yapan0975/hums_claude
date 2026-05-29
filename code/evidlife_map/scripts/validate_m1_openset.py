#!/usr/bin/env python
"""W3-C M1 open-set vacuity AUROC — paper RQ2 thesis.

The thesis: the SAME vacuity scalar that drives M1's calibration (validated
in RQ1) ALSO functions as a natural OOD detector — without a separate
unknown-class head.

Protocol:
  1. Treat SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY (5 rare classes:
     bicyclist, motorcyclist, truck, other-vehicle, other-ground) as
     "unknown". The remaining 14 classes are "known".
  2. Train a PointNet+EDL backbone on the 14 known classes (label 14
     unknown points as -100 ignore).
  3. At eval, compute per-point vacuity. For each point, ask: "did the
     model output high vacuity (uncertain) on points whose true class is
     unknown, and low vacuity on known points?"
  4. Report AUROC: 1.0 = perfect open-set discrimination.

Usage on 5090::

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/validate_m1_openset.py \
        --root /data/shared/SemanticKITTI \
        --epochs 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY,
    SemanticKITTIDataset,
)
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.edl_head import vacuity_from_alpha
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import PointNetEDL


def _collate(batch):
    assert len(batch) == 1
    return batch[0]


def _make_xyzi(entry, *, device, range_m: float = 100.0) -> torch.Tensor:
    xyz = entry.points_xyz_m.to(device, dtype=torch.float32)
    inten = entry.intensities.to(device, dtype=torch.float32).unsqueeze(-1)
    return torch.cat([xyz / range_m, inten], dim=-1)


def _mask_unknown_for_training(labels: torch.Tensor) -> torch.Tensor:
    """Set unknown-class labels to -100 (ignore) so we train on knowns only."""
    out = labels.clone()
    for u in SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY:
        out[out == u] = -100
    return out


def _is_unknown(labels: torch.Tensor) -> torch.Tensor:
    """Binary mask: True if point's GT label is in the unknown set."""
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for u in SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY:
        mask |= (labels == u)
    return mask


def compute_auroc(scores: torch.Tensor, labels_binary: torch.Tensor) -> float:
    """Compute AUROC for "high score = positive" binary classification.

    Uses the rank-based formulation that avoids scikit-learn dependency.
    """
    pos = scores[labels_binary]
    neg = scores[~labels_binary]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Mann-Whitney U statistic
    # Concatenate and rank
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    all_scores = torch.cat([pos, neg])
    sorted_idx = torch.argsort(all_scores)
    ranks = torch.empty_like(all_scores, dtype=torch.float64)
    ranks[sorted_idx] = torch.arange(1, all_scores.numel() + 1, dtype=torch.float64,
                                     device=all_scores.device)
    rank_sum_pos = ranks[:n_pos].sum().item()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs/m1_openset.json"))
    ap.add_argument("--ckpt-out", type=Path, default=Path("weights/pointnet_edl_openset.pt"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"unknown classes (held out): {list(SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY)}")
    print()

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    full = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    n = len(full)
    n_train = min(args.n_train, n - args.n_val)
    n_val = min(args.n_val, n - n_train)
    train_loader = DataLoader(Subset(full, list(range(n_train))),
                              batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(full, list(range(n - n_val, n))),
                            batch_size=1, shuffle=False, collate_fn=_collate)
    print(f"train={n_train} val={n_val}")

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetEDL(in_channels=4, num_classes=C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_train)
    loss_fn = EDLLoss(num_classes_plus_one=C + 1, kl_anneal_epochs=args.epochs,
                      kl_lambda_start=0.0, kl_lambda_end=1.0, ignore_index=-100)

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best_auroc = -1.0
    print(f"\n=== Train PointNet+EDL on 14 known classes for {args.epochs} epochs ===")
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        t0 = time.perf_counter()
        for entry in train_loader:
            x = _make_xyzi(entry, device=device)
            y = _mask_unknown_for_training(entry.labels).to(device)
            alpha, _ = model(x)
            loss_dict = loss_fn(alpha, y, epoch=ep)
            loss = loss_dict["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
        ep_loss /= max(1, len(train_loader))

        # Eval: vacuity AUROC on val frames
        model.eval()
        all_vac = []
        all_unk = []
        with torch.no_grad():
            for entry in val_loader:
                x = _make_xyzi(entry, device=device)
                alpha, vac = model(x)
                valid = entry.labels >= 0  # has GT label (drops bg)
                all_vac.append(vac[valid].cpu())
                all_unk.append(_is_unknown(entry.labels)[valid])
        all_vac_cat = torch.cat(all_vac)
        all_unk_cat = torch.cat(all_unk)
        auroc = compute_auroc(all_vac_cat, all_unk_cat)
        n_unk = int(all_unk_cat.sum())
        n_kn = int((~all_unk_cat).sum())

        msg = (f"ep {ep+1}/{args.epochs}: loss={ep_loss:.4f} "
               f"vacuity_AUROC={auroc:.4f} "
               f"(n_unk={n_unk}, n_known={n_kn}) "
               f"time={time.perf_counter()-t0:.1f}s")
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({"state_dict": model.state_dict(),
                        "auroc": auroc, "epoch": ep + 1,
                        "unknown_classes": list(SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY)},
                       args.ckpt_out)
            msg += f"  [SAVED]"
        print(f"  {msg}")

    print()
    print(f"=== DONE — best vacuity AUROC = {best_auroc:.4f} ===")
    if best_auroc > 0.7:
        print(f"  [PASS] Paper RQ2 thesis verified: vacuity acts as a usable OOD score")
    elif best_auroc > 0.55:
        print(f"  [MARGINAL] AUROC > random but not strong (PointNet capacity-limited)")
    else:
        print(f"  [WEAK]  AUROC near chance — backbone too weak to learn known boundary")

    out = {
        "config": {
            "seq": args.seq, "n_train": n_train, "n_val": n_val, "epochs": args.epochs,
            "unknown_classes": list(SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY),
        },
        "best_auroc": best_auroc,
        "ckpt": str(args.ckpt_out),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
