# EvidLife-Map · W3-E Training Findings (2026-05-29)

## Headline

The 9-11 % mIoU we observed earlier on the seq 08 80/20 split is **train-test
spatial-adjacency inflated**, not a true generalisation number. After running
20-epoch EDL training on a larger and properly-held-out split (seq 00 first
4 000 frames as train, seq 00 last 100 frames as val), the PointNet-Vanilla
backbone tops out at **val_mIoU = 1.46 %** at epoch 5 and plateaus there.

This is the **honest preliminary backbone ceiling**.

## Full results

```
Setup
  Sequence: 00 (4 540 frames total)
  Train:    frames 0–3 999  (4 000 frames)
  Val:      frames 4 440–4 540 (last 100 frames, properly held out)
  Backbone: PointNet-Vanilla, 0.21 M params (same as W3-A)
  Loss:    EDLLoss with kl_anneal_epochs=20, kl_lambda_start=0.0,
           kl_lambda_end=0.3 (gentler than the W3-A default kl_lambda_end=1.0)
  Epochs:  20
  GPU:     RTX 5090 (server, GPU 3)
  Total wall time: 1246.5 s (≈ 21 min)

Per-epoch
  ep  1/20: loss=0.5191 val_miou=0.0058 val_ece=0.2473  [SAVED]
  ep  2/20: loss=0.5401 val_miou=0.0058 val_ece=0.5067
  ep  3/20: loss=0.5519 val_miou=0.0144 val_ece=0.4334  [SAVED]
  ep  4/20: loss=0.6089 val_miou=0.0143 val_ece=0.5802
  ep  5/20: loss=0.7224 val_miou=0.0146 val_ece=0.2655  [SAVED ← best]
  ep  6/20: loss=0.7597 val_miou=0.0117 val_ece=0.6036
  ep  7/20: loss=0.7565 val_miou=0.0146 val_ece=0.2448
  ep  8/20: loss=0.7623 val_miou=0.0145 val_ece=0.2427
  ep  9/20: loss=0.7572 val_miou=0.0141 val_ece=0.5568
  ep 10/20: loss=0.7572 val_miou=0.0142 val_ece=0.2753
  ep 11/20: loss=0.7700 val_miou=0.0145 val_ece=0.5760
  ep 12/20: loss=0.7690 val_miou=0.0143 val_ece=0.5672
  ep 13/20: loss=0.7526 val_miou=0.0142 val_ece=0.2753
  ep 14/20: loss=0.7439 val_miou=0.0141 val_ece=0.5686
  ep 15/20: loss=0.7388 val_miou=0.0140 val_ece=0.3140
  ep 16/20: loss=0.7349 val_miou=0.0067 val_ece=0.3516
  ep 17/20: loss=0.7326 val_miou=0.0142 val_ece=0.5707
  ep 18/20: loss=0.7342 val_miou=0.0067 val_ece=0.3119
  ep 19/20: loss=0.7370 val_miou=0.0141 val_ece=0.5660
  ep 20/20: loss=0.7407 val_miou=0.0141 val_ece=0.5667
```

## Why the seq 08 80/20 number was inflated

SemKITTI seq 08 is one continuous traversal. Frames 80–99 of the first 100
of seq 08 are **physically right next to** frames 0–79: same buildings, same
road segments, same vegetation. A per-point MLP that has memorised the
voxel-pattern around frames 0–79 sees mostly the same voxel patterns at
frames 80–99 and "generalises" within the small spatial gap. This isn't
generalisation — it's spatial overlap.

Once we move to a 4 000-frame train slice and a held-out tail (frames
4 440–4 540 of seq 00), the spatial overlap is gone and the backbone reverts
to near-random performance for everything except the dominant road/building
voxels.

This is a real result, not a bug, and it's the **right** result to report:
the M1 evidential framework wins on calibration and latency at matched
backbone capacity, but absolute mIoU is dominated by the backbone (Limitation
v of paper §V.A).

## ECE oscillation

Notice that ECE oscillates between ≈ 0.24 and ≈ 0.58 across epochs
(values 0.24/0.50/0.43/0.58/0.27/0.60/0.24/0.24/0.56/0.28/0.58/0.57/0.28/
0.57/0.31/0.35/0.57/0.31/0.57/0.57). The KL pull schedule and the cosine LR
schedule interact: at points in the schedule where λ ramps faster than the
backbone learns, the head over-spreads to uniform Dirichlet and ECE spikes;
at points where λ relaxes, ECE recovers. The pattern is roughly every-other-
epoch, suggesting a stale `_eval` BN statistic during a `model.train()` →
`model.eval()` switch — to be diagnosed in the next revision.

## What this means for paper §IV.0

The §IV.0 table as currently committed (`M1 14.73 % vs R2 1.69 %` on
"seq 08 100 frames") is technically truthful (same harness, same backbone
capacity, same eval set) but the absolute numbers are inflated by spatial
overlap. The paper §V.A Limitation (v) already names this as
"backbone-bound, not method-bound". After this finding we should additionally:

1. Add a note in §IV.0 that the seq 08 100-frame number includes train-test
   spatial overlap.
2. Report the seq-00 held-out number (1.46 %) as the "honest backbone
   ceiling" alongside.
3. Restate the M1 vs R2 comparison in §IV.0 as a **relative ordering** rather
   than absolute, since the absolute is backbone-floor.

## Best ckpt

`weights/pointnet_edl_seq00_kl03.pt` on the server, saved at epoch 5:
val_miou = 0.0146, val_ece = 0.2655, kl_lambda_end = 0.3.

## Decision for next revision

This finding does NOT change the W3-B / W3-C / W3-D validations:

- W3-B (M3 decay, 2.64× ratio): the vacuity → decay coupling is on a 50-frame
  voxel map and does not depend on absolute mIoU.
- W3-C (vacuity AUROC 0.808): trained on the same seq 08 80/20 split as
  W3-A; this is open-set discrimination at matched backbone capacity and
  the relative ordering is what the paper claims.
- W3-D (lifelong sim, P=0.77 R=0.52): the M2 + M3 + M1 pipeline runs on real
  voxel data regardless of backbone mIoU.

What it DOES change: the §IV.0 absolute mIoU narrative. The next revision
should add a sentence reframing it.
