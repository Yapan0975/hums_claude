# §3.9 Backbone Path Findings (2026-05-29)

## TL;DR

**PVKD inherits the same spconv 1.x dependency as Cylinder3D.** The
research_plan v4 §3.9 Path 2 ("Adopt PVKD as Cylinder3D successor") does
not actually unblock the spconv 1.x ↔ 2.x value-semantic gap. The
unblock options collapse to:

- Path 1 — spconv 1.x source build against torch 2.11+cu130 (4-8 hours,
  60 % succeed probability)
- Path 3 — alternative backbone (MinkUNet / RandLA-Net / KPConv /
  WaffleIron) — pure-PyTorch, 2-3 days engineering, mIoU loss ~ 5-10 vs
  Cylinder3D 67 %
- Path 4 (NEW) — extend our existing PointNet2Lite into a deeper
  kNN-context backbone (effectively a small RandLA-Net), no spconv
  dependency at all

## Inspection of PVKD repo (via WebFetch 2026-05-29)

URL: https://github.com/cardwing/Codes-for-PVKD

Findings:

- `requirements.txt`: `spconv==1.2.1` — explicit 1.x dependency
- `README.md`: PyTorch >= 1.2 (old; same era as Cylinder3D)
- Architecture: Cylinder3D backbone unchanged (PVKD's contribution is
  knowledge-distillation of Cylinder3D into smaller versions
  `1.0x_71.8 mIoU`, `1.5x_72.4 mIoU`)
- Pretrained weights: distributed as `.pt` checkpoints via Google Drive

This means PVKD pretrained weights would face the *same* kernel-index
ordering value-semantic gap we observed for the original Cylinder3D
checkpoint. Adopting PVKD does not avoid Path 1's spconv 1.x source build
or Path 3's backbone swap.

## Revised path order (v5 / 2026-05-29 update)

| # | Path | Effort | mIoU floor | Risk | Decision-ready |
|---|------|--------|-----------|------|----------------|
| 1 | spconv 1.x source build vs cu130 | 4-8 hr | ~ 67 % (full fidelity) | M-H | yes |
| 3 | Alt sparse-conv backbone (MinkUNet via MinkowskiEngine) | 2-3 d | ~ 60-65 % | M | yes |
| 4 | Extend PointNet2Lite to small RandLA-Net | 1-2 d | ~ 50-60 % (estimate) | L | yes |
| ~~2~~ | ~~PVKD adoption~~ | ~~1 d~~ | ~~67 %~~ | RULED OUT | inherits spconv 1.x |

## Recommended sequence

Given the (a) 9-month runway to IROS 2027 and (b) the fact that
PointNet2Lite is already producing 17.92 % mIoU at 0.28 M params, the
**pragmatic path** is:

1. **Now (W4)**: extend PointNet2Lite to a 2-layer kNN-context backbone
   (add a second SA-FP stage), target 25-35 % mIoU at 1-2 M params.
   Pure-PyTorch, no spconv dependency. Estimated 1-2 days.
2. **W6-W8**: attempt Path 1 (spconv 1.x source build) in parallel
   with §IV.D vacuity AUROC scale-up. If it succeeds, the §IV.0
   absolute mIoU lifts to Cylinder3D-class numbers; if it fails, Path 4
   carries us to the IROS-acceptable 50-60 % floor.
3. **Fallback (W12)**: if both Path 1 and Path 4 stall, attempt
   MinkUNet via MinkowskiEngine (which has cu126 wheels for sm_90, may
   also work for sm_120).

## Why Path 4 (extend PointNet2Lite) is highest-leverage now

- We **already** have PointNet2Lite delivering 17.92 % mIoU on the
  official SemKITTI split.
- The §IV.0 narrative is already valid: M1 (EDL) wins on calibration at
  matched backbone tier; the absolute mIoU floor is a backbone-capacity
  story.
- Adding one more SA-FP stage (kNN + max-pool + interp + MLP) is
  conceptually a 1-day implementation against the existing
  `pointnet2_lite.py`.
- The result is a defensible "small RandLA-Net" backbone that the IROS
  reviewer can compare to alongside Cylinder3D (if Path 1 succeeds) or
  *instead of* Cylinder3D (if Path 1 fails).

## Action items

- [ ] Sketch a `PointNet2Lite_v2` with 2 SA-FP stages
- [ ] Re-train on multi-seq, target > 25 % mIoU
- [ ] If Path 1 also succeeds, report both rows in §IV.0 final table
- [ ] Update research_plan v4 §3.9 to retire Path 2 (PVKD)
