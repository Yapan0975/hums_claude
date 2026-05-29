#!/usr/bin/env python
"""W3-O M1 open-set AUROC — multi-seq + PointNet-Vanilla backbone.

Same protocol as W3-N (multi-seq train, 300 fr/seq) but at the
PointNet-Vanilla backbone (0.21 M params). Tests whether the
W3-N AUROC drop (0.808 → 0.738) is due to (a) protocol change or
(b) backbone change.

If W3-O AUROC ≈ 0.738: backbone-driven (better fit reduces vacuity
discrimination).
If W3-O AUROC ≈ 0.808: protocol-driven (single-seq W3-C was
spatially inflated, multi-seq is the honest value).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY,
)
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import PointNetEDL, _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample
from scripts.validate_m1_openset import (
    _mask_unknown_for_training, _is_unknown, compute_auroc,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", type=Path,
                    default=Path("artifacts/m1_openset_multiseq_vanilla.json"))
    ap.add_argument("--out-ckpt", type=Path,
                    default=Path("weights/m1_openset_multiseq_vanilla.pt"))
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"unknown classes: {list(SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY)}\n",
          flush=True)

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetEDL(in_channels=4, num_classes=C, backbone="vanilla").to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"backbone: vanilla, params: {n_params:.3f} M", flush=True)

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

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_auroc = -1.0
    history = []
    t_total = time.perf_counter()
    print(f"\n=== Open-set train (vanilla) {args.epochs} epochs ===\n", flush=True)
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
            y = _mask_unknown_for_training(entry.labels).to(device)
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

        model.eval()
        all_vac, all_unk = [], []
        with torch.no_grad():
            for entry in val_loader:
                entry = _subsample(entry, args.n_points)
                x = _make_xyzi(entry, device=device)
                alpha, vac = model(x)
                valid = entry.labels >= 0
                all_vac.append(vac[valid].cpu())
                all_unk.append(_is_unknown(entry.labels)[valid])
        all_vac_cat = torch.cat(all_vac)
        all_unk_cat = torch.cat(all_unk)
        auroc = compute_auroc(all_vac_cat, all_unk_cat)

        dt = time.perf_counter() - t_ep
        msg = (f"ep {ep+1}/{args.epochs}: loss={ep_loss:.4f} "
               f"vacuity_AUROC={auroc:.4f} time={dt:.1f}s")
        history.append({"epoch": ep + 1, "loss": ep_loss, "auroc": float(auroc)})
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({"state_dict": model.state_dict(),
                        "vacuity_auroc": float(auroc), "epoch": ep + 1,
                        "backbone": "vanilla"}, args.out_ckpt)
            msg += "  [SAVED]"
        print(f"  {msg}", flush=True)

    total = time.perf_counter() - t_total
    print(f"\n=== DONE — best AUROC = {best_auroc:.4f}, total {total:.1f}s ===")

    args.out_json.write_text(json.dumps({
        "best_auroc": float(best_auroc), "backbone": "vanilla",
        "params_M": round(n_params, 3),
        "split": "PRIMARY 14/5", "history": history,
    }, indent=2))
    print(f"json: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
