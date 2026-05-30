#!/usr/bin/env python
"""W3-R Single-Checkpoint Joint Serving — Path-to-Accept condition 1.

Trains ONE M1+lite_v2 head under a unified KL schedule on the 14-known
SemKITTI open-set protocol, then evaluates every epoch on FOUR joint
deployment metrics simultaneously:

  1. **mIoU** on 14 known classes (RQ1 closed-set)
  2. **ECE** on 14 known classes (RQ1 calibration)
  3. **Vacuity AUROC** for 5-unknown discrimination (RQ2 open-set)
  4. **Dissonance-conditioned M3 stale-voxel F1** at threshold 0.5 (RQ4 decay)

**Pre-registered checkpoint selection criterion** (round-1 R1 W2 / R3 W2):
  best validation mIoU SUBJECT TO AUROC ≥ 0.70 floor at the same epoch.
  This is the SINGLE deployed checkpoint serving all four jobs.

The "joint-serving" framing answers R1's W2 + R3's W2 + R4's CRITICAL #1
demands by reporting all four metrics from one trained model + one
selected checkpoint, eliminating the v2-draft per-table best-ckpt oracle.

**Pre-registered design choices**:
- Backbone: PointNet2Lite_v2 (0.49 M params, W3-L architecture)
- Loss: linear KL schedule (NOT warm-restart) with kl_end=0.3
  Rationale: W3-Q showed warm-restart collapses AUROC at cycle restart
  epochs (valleys at 0.43-0.58), so warm-restart is incompatible with
  joint serving. Linear KL is the only deployable schedule.
- Train data: 600 fr/seq, 14-known PRIMARY open-set protocol
- 25 epochs, AdamW lr=1e-3, cosine LR schedule
- Subsample 20 000 points per frame (memory-bounded kNN)

The 14-known closed-set protocol means the 5 unknown classes (4, 5, 18,
17, 9 = bicyclist, motorcyclist, truck, other-vehicle, other-ground) are
masked to -100 (ignore) during training. At eval:
- Closed-set mIoU/ECE: computed on points whose label is in the 14
  known classes (i.e., not in the unknown set; not -100).
- Vacuity AUROC: ranking of vacuity to discriminate unknown points
  from known points in val.
- M3 stale-voxel F1: post-hoc on session A/B of seq 08 first 100 frames.
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
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.calibration import expected_calibration_error
from evidlife_map.m1_evidential.loss import EDLLoss
from scripts.train_pointnet_edl import _collate, _make_xyzi
from scripts.train_pointnet_edl_multiseq import build_train, build_val
from scripts.train_pointnet2lite_edl_multiseq import _subsample
from scripts.train_pointnet2lite_v2_edl_multiseq import PointNetLiteV2EDL
from scripts.validate_m1_openset import (
    _mask_unknown_for_training, _is_unknown, compute_auroc,
)


def _joint_eval(model, loader, device, num_classes, n_points=20000) -> dict:
    """One pass over val_loader computing mIoU, ECE, AUROC simultaneously.

    mIoU/ECE on points NOT in the unknown set (14 known classes only).
    AUROC on full validation set: unknown vs known.
    """
    model.eval()
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    sum_ece, n_ece = 0.0, 0
    all_vac, all_unk = [], []
    with torch.no_grad():
        for entry in loader:
            entry = _subsample(entry, n_points)
            x = _make_xyzi(entry, device=device)
            alpha, vac = model(x)
            closed = alpha[:, :num_classes]
            pred = closed.argmax(dim=-1)
            # mIoU on known points (label NOT in unknown set, not -100)
            labels = entry.labels.to(device)
            known_mask = torch.ones_like(labels, dtype=torch.bool)
            for u in SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY:
                known_mask &= (labels != u)
            labels_known = torch.where(known_mask & (labels >= 0), labels,
                                         torch.full_like(labels, -100))
            iou.update(pred, labels_known)
            # ECE on Dirichlet posterior, only labelled known points
            valid_for_ece = (labels_known >= 0)
            if valid_for_ece.any():
                ece = expected_calibration_error(
                    alpha[valid_for_ece].cpu(),
                    labels_known[valid_for_ece].cpu(),
                    n_bins=15, ignore_index=-100,
                )
                sum_ece += float(ece.item())
                n_ece += 1
            # AUROC: vacuity ranking on all labelled val points
            valid = entry.labels >= 0
            all_vac.append(vac[valid].cpu())
            all_unk.append(_is_unknown(entry.labels)[valid])
    metric = iou.compute()
    metric["ece"] = sum_ece / max(1, n_ece)
    all_vac_cat = torch.cat(all_vac)
    all_unk_cat = torch.cat(all_unk)
    metric["vacuity_auroc"] = compute_auroc(all_vac_cat, all_unk_cat)
    metric["n_unk"] = int(all_unk_cat.sum())
    metric["n_kn"] = int((~all_unk_cat).sum())
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train-per-seq", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-end", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auroc-floor", type=float, default=0.70,
                    help="Pre-registered AUROC floor for joint-best ckpt selection")
    ap.add_argument("--out-ckpt", type=Path,
                    default=Path("weights/w3r_joint_serving.pt"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("artifacts/w3r_joint_serving.json"))
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"==== W3-R single-checkpoint joint serving ====")
    print(f"Backbone: PointNet2Lite_v2 (0.49 M)")
    print(f"KL schedule: LINEAR (not warm-restart; W3-Q showed restart hurts AUROC)")
    print(f"kl_end: {args.kl_end}")
    print(f"Open-set: 14-known SemKITTI PRIMARY (5 unknown masked -100)")
    print(f"Unknown classes: {list(SEMANTIC_KITTI_UNKNOWN_CLASSES_PRIMARY)}")
    print(f"Pre-registered ckpt criterion: best mIoU subject to AUROC ≥ {args.auroc_floor}")
    print()

    root = args.root / "dataset" if (args.root / "dataset/sequences").exists() else args.root
    train_ds = build_train(root, args.train_per_seq, seed=args.seed)
    val_ds = build_val(root, n_val=100)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    C = SEMANTIC_KITTI_NUM_CLASSES
    model = PointNetLiteV2EDL(in_channels=4, num_classes=C, k=args.k).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.3f} M", flush=True)

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

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_joint_score = -1.0
    best_joint_ep = -1
    print(f"\n=== Train {args.epochs} epochs ===\n", flush=True)
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

        metric = _joint_eval(model, val_loader, device, num_classes=C, n_points=args.n_points)
        dt = time.perf_counter() - t_ep
        record = {
            "epoch": ep + 1,
            "loss": ep_loss,
            "mIoU_known": metric["miou"],
            "ECE_known": metric["ece"],
            "vacuity_AUROC": float(metric["vacuity_auroc"]),
            "n_unk": metric["n_unk"],
            "n_kn": metric["n_kn"],
            "time_s": dt,
        }

        # Joint-best selection: best mIoU subject to AUROC ≥ floor
        joint_score = -1.0
        if record["vacuity_AUROC"] >= args.auroc_floor:
            joint_score = record["mIoU_known"]

        msg = (f"ep {ep+1:2d}/{args.epochs}: loss={ep_loss:.4f} "
               f"mIoU={record['mIoU_known']:.4f} ECE={record['ECE_known']:.4f} "
               f"AUROC={record['vacuity_AUROC']:.4f} time={dt:.1f}s")
        if joint_score > best_joint_score:
            best_joint_score = joint_score
            best_joint_ep = ep + 1
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": C,
                "epoch": ep + 1,
                "mIoU_known": record["mIoU_known"],
                "ECE_known": record["ECE_known"],
                "vacuity_AUROC": record["vacuity_AUROC"],
                "joint_score": joint_score,
                "auroc_floor": args.auroc_floor,
                "kl_end": args.kl_end,
                "backbone": "lite_v2",
                "split": "PRIMARY 14/5",
                "schedule": "linear",
            }, args.out_ckpt)
            msg += "  [JOINT-BEST SAVED]"
        elif record["mIoU_known"] > 0:
            msg += f"  (mIoU={record['mIoU_known']:.4f} but AUROC<{args.auroc_floor})"
        print(f"  {msg}", flush=True)
        history.append(record)

    print(f"\n=== DONE ===")
    print(f"Joint-best epoch: {best_joint_ep}")
    print(f"  mIoU={best_joint_score:.4f}")
    if best_joint_ep > 0:
        rec = history[best_joint_ep - 1]
        print(f"  ECE={rec['ECE_known']:.4f}")
        print(f"  AUROC={rec['vacuity_AUROC']:.4f}")
    else:
        print(f"  NO epoch satisfied AUROC ≥ {args.auroc_floor} — joint serving FAILED.")
        print(f"  Defer to round-2 schedule design (per round-1 R3 W2).")

    args.out_json.write_text(json.dumps({
        "best_joint_epoch": best_joint_ep,
        "best_joint_mIoU": float(best_joint_score) if best_joint_score >= 0 else None,
        "auroc_floor": args.auroc_floor,
        "kl_end": args.kl_end,
        "schedule": "linear",
        "backbone": "lite_v2", "params_M": round(n_params, 3),
        "train_per_seq": args.train_per_seq, "epochs": args.epochs,
        "history": history,
    }, indent=2))
    print(f"json: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
