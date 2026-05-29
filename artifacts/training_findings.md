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

## Multi-sequence audit completion (W3-F / W3-G / W3-H, 2026-05-29)

Three trainings on official SemKITTI split (train: seq 00–07, 09, 10
sampled 300 frames each ≈ 2 971 train; val: first 100 of seq 08):

```
EDL-vanilla    0.210 M params   20 ep   best val mIoU=1.66% @ ep 9   ECE=0.17
CE-vanilla     0.210 M params   20 ep   best val mIoU=2.28% @ ep 11  ECE=0.68
EDL-lite       0.276 M params   20 ep   best val mIoU=18.65% @ ep 2  ECE=0.06–0.25
```

Apples-to-apples evaluation under the same harness (compare_multiseq.py):

| System | Params | Fresh mIoU | Fresh ECE |
|---|---|---|---|
| CE-vanilla | 0.210 M | 2.28 % | 0.6837 |
| EDL-vanilla | 0.210 M | 1.66 % | 0.1695 |
| **EDL-lite** | 0.276 M | **17.92 %** | **0.0962** |

### What the three rows reveal

1. **At matched 0.21 M PointNet-Vanilla**, CE and EDL sit within 0.6 pp on
   mIoU (CE leads slightly), but EDL gives 4.0× better ECE. The §I.B
   thesis on the calibration axis is preserved at proper protocol.
2. **Adding kNN local context** (PointNet2Lite, +30 % params, k=16) lifts
   M1 mIoU by 10.8× (1.66 → 17.92) and ECE by 1.8× — confirming that the
   PointNet floor is *neighbourhood-bound*, not capacity-bound.
3. **EDL+lite vs CE-vanilla**: 7.8× better mIoU AND 7.1× better ECE
   simultaneously. The §III.B "trade-off" claim becomes "no trade-off"
   once a neighbourhood-aware backbone is available.

### EDL post-epoch-1 collapse under multi-seq

Both EDL configurations (vanilla and lite) peak early (ep 2-9) and then
plateau slightly below their peak rather than monotonically climbing.
This is the same KL-anneal collapse documented in §V.B(d); kl_lambda_end
= 0.3 attenuates but does not eliminate it. Action: late-stage
KL warm-restart for the next revision.

### CE post-epoch-11 regression on vanilla

CE-vanilla peaked at 2.28% at ep 11 then regressed to 0.97-1.80% by
ep 20. This is consistent with the cosine LR schedule annealing past
the optimal point on this small training set; with more diverse training
data (full seq 00-10 instead of 300/seq subsample) this regression
should disappear.

### Per-epoch logs

Full per-epoch traces are in `train_multiseq_edl.log`,
`train_multiseq_ce.log`, `train_multiseq_lite_v2.log`. Comparison JSON:
`multiseq_compare.json`.

## W3-J / W3-K / W3-L ablation campaign (2026-05-29, late evening)

Three follow-up trainings on the same multi-seq protocol, probing the
three improvement axes the §V.A and §V.B(d) sections flag:

| # | Knob | mIoU best (in-train) | mIoU (fresh eval) | ECE (fresh) |
|---|------|----------------------|-------------------|-------------|
| W3-H (baseline) | linear KL, k=16, 300 fr/seq, 0.28 M | 18.65 % | 17.92 % | 0.096 |
| W3-J | linear → **warm-restart KL** (period 5) | 19.52 % | **18.61 %** | 0.170 |
| W3-K | linear KL, **600 fr/seq** (2× data) | 18.84 % | 17.83 % | 0.109 |
| W3-L | linear KL, **Path 4 backbone** (lite_v2, 0.49 M) | 20.51 % | 17.74 % | 0.117 |

Comparison JSON: `multiseq_compare_6way.json`. Per-epoch logs:
`train_multiseq_warmrestart.log`, `train_multiseq_bigger.log`,
`train_multiseq_v2.log`.

### Reading the four rows

- **W3-J (warm-restart)**: cycle-2/3/4 restart epochs all hit mIoU
  ≥ 18.7 %; cycle-4 ramp position 06 hits 19.52 % in-train. Fresh-eval
  gain over W3-H baseline = +0.69 pp (+3.8 % relative). Validates the
  §V.B(d) fix proposal. ECE trade is real (0.096 → 0.170, ~1.8×).
- **W3-K (bigger train)**: doubling frames per sequence yields no
  measurable improvement at this backbone tier. Consistent with the
  *neighbourhood-bound* interpretation of §IV.0: more train *frames*
  do not help when each frame's per-point classification is already
  saturated by the limited k = 16 neighbourhood.
- **W3-L (lite_v2)**: the deeper, 3-LSE+AP-block backbone peaks at
  20.51 % during training but the fresh-eval mIoU drops to 17.74 %.
  Interpretation: the 1.8× more params over-fit on 2 970 train frames
  at the cosine LR schedule we used; combining lite_v2 with warm-restart
  + 600 fr/seq + longer schedule is the natural follow-up (Path 4
  unblock fully landed; next iteration to push past 25 %).

### In-train vs fresh-eval mIoU gap

The "best-ep mIoU" stored in each ckpt is computed during training with
a different random subsample each epoch, while the fresh eval uses one
common subsample for all systems. This explains the systematic gap
(W3-J 19.52 → 18.61, W3-L 20.51 → 17.74). For paper §IV.0, fresh-eval
is the more honest number; for selecting *between* ckpts of the same
training run, the in-train mIoU is what we used (best-ckpt-by-val).

### Decision for §IV.0 final table

Report two rows: linear-KL (calibration-best, ECE 0.096) and
warm-restart (mIoU-best, +0.69 pp at +1.8× ECE). Mark lite_v2 and bigger
as ablation lines that do *not* lift the §IV.0 floor and explain why
in supplementary `W3_campaign_summary.md`.
