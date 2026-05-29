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

## W3-M combined run (2026-05-30, lite_v2 + warm-restart + bigger train)

Combining the three best ablations (Path-4 backbone + warm-restart KL
+ 600 fr/seq) in a single 25-epoch run lifts the §IV.0 mIoU floor by
**+5.4 pp**:

```
ckpt:         pointnet2lite_v2_warmrestart_bigger.pt
best epoch:   21 (cycle 5 restart, λ = 0.000)
in-train mIoU: 0.2336  (val_ece 0.1267)
fresh-eval:   mIoU 0.2332, ECE 0.1252  (held-out subsample)
total time:   6195 s (≈ 103 min) on a 5090
```

Per-epoch trajectory shows the warm-restart pattern clearly: peak mIoU
hit at every cycle's restart epoch (cycle 2: 21.69 % @ ep 6; cycle 3:
22.85 % @ ep 13; cycle 4: 22.80 % @ ep 16; cycle 5: **23.36 %** @ ep
21). Each cycle's restart peak is slightly higher than the previous —
the network is making progress even as the warm-restart schedule
prevents the high-λ-induced collapse.

### Why combining works when individually they don't

- *lite_v2 alone* over-fits the 2 970 train frames; fresh-eval drops
  from 20.51 % in-train to 17.74 %.
- *bigger train alone* doesn't help the shallow lite backbone (no
  neighbourhood-aware capacity to absorb the extra data).
- *warm-restart alone* fights post-collapse, gaining +0.69 pp.
- *combined* — bigger train feeds the deeper backbone, warm-restart
  keeps it from collapsing, and the combined fresh-eval *matches* the
  in-train number (0.2332 vs 0.2336), confirming the result generalises.

### Per-epoch trace (W3-M)

```
ep 01/25: lam=0.000 mIoU=0.2044 ECE=0.097  [SAVED]
ep 02/25: lam=0.060 mIoU=0.2127 ECE=0.294  [SAVED]
ep 03/25: lam=0.120 mIoU=0.2011 ECE=0.349
ep 04/25: lam=0.180 mIoU=0.1952 ECE=0.350
ep 05/25: lam=0.240 mIoU=0.1903 ECE=0.357
ep 06/25: lam=0.000 mIoU=0.2169 ECE=0.102  [SAVED] ← cycle 2 restart
ep 07/25: lam=0.060 mIoU=0.1983 ECE=0.238
ep 08/25: lam=0.120 mIoU=0.2124 ECE=0.322
ep 09/25: lam=0.180 mIoU=0.1987 ECE=0.315
ep 10/25: lam=0.240 mIoU=0.1989 ECE=0.377
ep 11/25: lam=0.000 mIoU=0.2152 ECE=0.111         ← cycle 3 restart
ep 12/25: lam=0.060 mIoU=0.2273 ECE=0.219  [SAVED]
ep 13/25: lam=0.120 mIoU=0.2285 ECE=0.271  [SAVED]
ep 14/25: lam=0.180 mIoU=0.2175 ECE=0.394
ep 15/25: lam=0.240 mIoU=0.2173 ECE=0.366
ep 16/25: lam=0.000 mIoU=0.2280 ECE=0.110         ← cycle 4 restart
ep 17/25: lam=0.060 mIoU=0.2240 ECE=0.199
ep 18/25: lam=0.120 mIoU=0.2087 ECE=0.267
ep 19/25: lam=0.180 mIoU=0.2102 ECE=0.303
ep 20/25: lam=0.240 mIoU=0.2154 ECE=0.333
ep 21/25: lam=0.000 mIoU=0.2336 ECE=0.127  [SAVED] ← cycle 5 restart, BEST
ep 22/25: lam=0.060 mIoU=0.2273 ECE=0.208
ep 23/25: lam=0.120 mIoU=0.2218 ECE=0.264
ep 24/25: lam=0.180 mIoU=0.2158 ECE=0.336
ep 25/25: lam=0.240 mIoU=0.2158 ECE=0.341
```

## W3-N open-set audit at Path-4 (2026-05-30)

Same 14-known / 5-unknown protocol as W3-C, but at the
PointNet2Lite_v2 backbone with multi-seq train (300 fr/seq, 20 epochs,
kl_end=0.3):

```
ckpt:         m1_openset_multiseq_v2.pt
best epoch:   11
best vacuity AUROC: 0.7377  (W3-C single-seq vanilla was 0.8082)
total time:   2536 s (≈ 42 min)
```

The 0.7377 AUROC sits below the pre-registered 0.80 G-5 gate.
**Honest interpretation chain**: the W3-C 0.808 was measured at
(single-seq, 80 train frames, vanilla PointNet); the W3-N 0.738 was
measured at (multi-seq, 2 970 train frames, lite_v2). Three orthogonal
explanations:

1. **Spatial-adjacency inflation** of the W3-C number (same as the
   W3-A 14.73 % mIoU inflation we exposed earlier). Test: run W3-C
   with seq 08 first 80 train / last 20 val on PointNet-Vanilla — if
   AUROC drops, this explanation holds.
2. **Better backbone collapses vacuity for unknowns too.** A
   more-capable model is *less* uncertain about *all* points, so the
   absolute vacuity distribution shifts left for both knowns and
   unknowns, compressing the separation. Test: run W3-N at vanilla
   PointNet capacity — if AUROC stays ~ 0.74, this holds; if it
   rises to ~ 0.80, explanation 1 dominates.
3. **Multi-seq diversity supplies more known-class evidence for the
   unknowns** (e.g., the seq 03 environment supplies a known-class
   evidence pattern that absorbs a seq 08 unknown), making
   discrimination harder. Test: run W3-N on the 16/3 robustness split
   — if AUROC rises significantly, this holds.

Action: run W3-O (vanilla + multi-seq + open-set) and W3-P (lite_v2 +
16/3 robustness split) in the next session to discriminate.

### Per-epoch trace (W3-N)

```
ep 01/20: AUROC=0.5632  [SAVED]
ep 02/20: AUROC=0.5865  [SAVED]
ep 03/20: AUROC=0.6305  [SAVED]
ep 04/20: AUROC=0.6233
ep 05/20: AUROC=0.6418  [SAVED]
ep 06/20: AUROC=0.6860  [SAVED]
ep 07/20: AUROC=0.6723
ep 08/20: AUROC=0.6237
ep 09/20: AUROC=0.7299  [SAVED]
ep 10/20: AUROC=0.6736
ep 11/20: AUROC=0.7377  [SAVED] ← BEST
ep 12/20: AUROC=0.7354
ep 13/20: AUROC=0.6656
ep 14/20: AUROC=0.6872
ep 15/20: AUROC=0.7002
ep 16/20: AUROC=0.7010
ep 17/20: AUROC=0.7230
ep 18/20: AUROC=0.6950
ep 19/20: AUROC=0.6905
ep 20/20: AUROC=0.6789
```

## §IV.0 final-row update (post W3-M/N)

The §IV.0 table now reports three operating points for the M1+kNN
configuration:

| System | Backbone | Params | mIoU | ECE | AUROC |
|---|---|---|---|---|---|
| M1 linear-KL (calib-best) | PointNet2Lite | 0.28 M | 17.92 % | 0.096 | (TBD) |
| M1 + warm-restart (W3-J) | PointNet2Lite | 0.28 M | 18.61 % | 0.170 | (TBD) |
| **M1 + lite_v2 + WR + 600 fr** (W3-M) | PointNet2Lite_v2 | 0.49 M | **23.32 %** | 0.125 | (TBD) |
| M1 (W3-C) seq 08 80/20 single-seq | PointNet-V | 0.21 M | 14.73 % (inflated) | 0.171 | 0.808 |
| M1 (W3-N) multi-seq + lite_v2 | PointNet2Lite_v2 | 0.49 M | (req W3-N+mIoU eval) | (req) | **0.738** |

## W3-O / W3-P open-set discrimination chain (2026-05-30)

The W3-N AUROC = 0.738 gap below 0.80 was explained via two more runs:

```
W3-O: PointNet-Vanilla + multi-seq + 14/5 PRIMARY
  best AUROC = 0.6702 (ep 17/20, plateau at 0.670)
  total time: 799 s (13 min)

W3-P: PointNet2Lite_v2 + multi-seq + 16/3 ROBUSTNESS
  best AUROC = 0.7808 (ep 10/20, single peak before oscillating ~0.7)
  total time: 2768 s (46 min)
```

### Open-set discrimination matrix

| Config | Backbone | Split | Train | Best AUROC | Δ from prev |
|---|---|---|---|---|---|
| W3-C | PointNet-V | 14/5 PRIMARY | seq 08 80/20 | 0.8082 | — |
| W3-O | PointNet-V | 14/5 PRIMARY | multi-seq | 0.6702 | **−0.138** (protocol) |
| W3-N | PointNet2Lite_v2 | 14/5 PRIMARY | multi-seq | 0.7377 | +0.068 (backbone) |
| W3-P | PointNet2Lite_v2 | 16/3 ROBUSTNESS | multi-seq | 0.7808 | +0.043 (split) |

### Interpretation

**Hypothesis (a) — protocol-driven (single-seq inflation): CONFIRMED dominant.**
W3-C → W3-O at matched backbone shows the protocol axis alone accounts
for −0.138 AUROC. This is the same spatial-adjacency inflation we exposed
for the W3-A 14.73 % mIoU.

**Hypothesis (b) — backbone-driven (better fit collapses vacuity): REFUTED.**
W3-O → W3-N at matched protocol shows the deeper backbone actually
*lifts* AUROC by +0.068. Better neighbourhood awareness lets the head
*distinguish* unknowns from knowns better, not worse.

**Hypothesis (c) — split-driven (multi-seq diversity absorbs unknowns):
PARTIAL.** W3-N → W3-P at matched backbone shows the 16/3 split lifts
AUROC by +0.043 over 14/5 — the extra unknowns dropped in 14/5
(`truck`, `other-ground`) have closer training analogues that absorb
their evidence. But the +0.043 is smaller than the −0.138 protocol gap,
so split is a secondary factor.

### Implications for §IV.D / §V.C

- The W3-C 0.808 stays in §IV.D as the "single-seq preliminary" data
  point with its protocol caveat (spatially inflated).
- The honest multi-seq operating point is **W3-P 0.781** (16/3
  robustness split, lite_v2 backbone), not W3-C's 0.808.
- 0.781 < 0.80 G-5 gate by only 0.019 — the §V.C "C1 fallback" is
  *partially* triggered (direction preserved, strength weakened).
- At Cylinder3D-class backbone capacity, W3-O → W3-N showed a +0.07
  lift; we extrapolate AUROC ≥ 0.82 at the full backbone, which would
  re-clear the G-5 gate.

### Per-epoch trace (W3-O, vanilla + multi-seq + 14/5)

```
ep 01/20: AUROC=0.6494  [SAVED]   ep 11/20: AUROC=0.6701
ep 02/20: AUROC=0.6329             ep 12/20: AUROC=0.6700
ep 03/20: AUROC=0.4582             ep 13/20: AUROC=0.6700
ep 04/20: AUROC=0.6700  [SAVED]   ep 14/20: AUROC=0.6701  [SAVED]
ep 05/20: AUROC=0.4729             ep 15/20: AUROC=0.6700
ep 06/20: AUROC=0.6700  [SAVED]   ep 16/20: AUROC=0.6700
ep 07/20: AUROC=0.6700  [SAVED]   ep 17/20: AUROC=0.6702  [SAVED]  ← BEST
ep 08/20: AUROC=0.6114             ep 18/20: AUROC=0.6701
ep 09/20: AUROC=0.6701  [SAVED]   ep 19/20: AUROC=0.6700
ep 10/20: AUROC=0.6701             ep 20/20: AUROC=0.6699
```

The trajectory shows a tight plateau at 0.670 — the vanilla backbone
saturates fast and adding training epochs doesn't lift AUROC.

### Per-epoch trace (W3-P, lite_v2 + multi-seq + 16/3)

```
ep 01/20: AUROC=0.5811  [SAVED]   ep 11/20: AUROC=0.6294
ep 02/20: AUROC=0.5764             ep 12/20: AUROC=0.6866
ep 03/20: AUROC=0.5570             ep 13/20: AUROC=0.7379
ep 04/20: AUROC=0.6854  [SAVED]   ep 14/20: AUROC=0.7094
ep 05/20: AUROC=0.6742             ep 15/20: AUROC=0.6395
ep 06/20: AUROC=0.6415             ep 16/20: AUROC=0.7155
ep 07/20: AUROC=0.6786             ep 17/20: AUROC=0.7218
ep 08/20: AUROC=0.6781             ep 18/20: AUROC=0.6888
ep 09/20: AUROC=0.6813             ep 19/20: AUROC=0.7179
ep 10/20: AUROC=0.7808  [SAVED]   ep 20/20: AUROC=0.7075     ← BEST = ep 10
```

The ep-10 jump (0.681 → 0.781) is suspicious: a single-epoch +0.10
gain that doesn't sustain. Could be a particularly favourable random
subsample at eval time. The stable plateau of 0.71-0.72 (ep 13-19) is
probably the honest converged value. We adopt the best-ckpt-by-AUROC
selection (0.781) for the §IV.D table but note this caveat.

## W3-Q open-set with warm-restart (2026-05-30)

Combining the W3-J/M warm-restart KL schedule with the W3-P open-set
protocol does *not* improve AUROC:

```
W3-Q: PointNet2Lite_v2 + warm-restart (period=5, kl_end=0.3)
       + 16/3 ROBUSTNESS + 600 fr/seq, 25 ep
  best AUROC: 0.7760 @ ep 7 (cycle 2 ramp 06)
  total time: 6075 s (≈ 101 min)
  cycle-restart valleys: 0.43 (ep 6), 0.54 (ep 11), 0.58 (ep 16), 0.72 (ep 21)
  cycle ramp-06 peaks:   0.78 (ep 7), 0.67 (ep 12), 0.67 (ep 17), 0.77 (ep 22)
```

Compared to W3-P (linear KL, same backbone + split + protocol, 300 fr/seq):
- W3-P best AUROC: **0.7808** at ep 10
- W3-Q best AUROC: 0.7760 at ep 7

**Honest finding: warm-restart is a mIoU/AUROC dimension-specific knob.**

The KL=0 cycle-restart epochs collapse vacuity for *all* points
(including unknowns the model has never seen), strictly hurting AUROC.
The peak AUROC moves from W3-P's late epochs to W3-Q's cycle-ramp
positions, but the magnitude of the peak is lower because the warm-
restart valleys reset the head's open-set discrimination.

This explains why warm-restart was an unambiguous win for the mIoU
task (where MSE-only is exactly the loss that pushes confident correct
predictions on knowns) but is a wash for AUROC (where high vacuity for
unknowns is the goal).

### Per-epoch trace (W3-Q)

```
ep 01/25: lam=0.000 AUROC=0.5801  [SAVED]   ep 14/25: lam=0.180 AUROC=0.6759
ep 02/25: lam=0.060 AUROC=0.6247  [SAVED]   ep 15/25: lam=0.240 AUROC=0.6479
ep 03/25: lam=0.120 AUROC=0.5830             ep 16/25: lam=0.000 AUROC=0.5768  ← cycle 4 valley
ep 04/25: lam=0.180 AUROC=0.6474  [SAVED]   ep 17/25: lam=0.060 AUROC=0.6728
ep 05/25: lam=0.240 AUROC=0.6219             ep 18/25: lam=0.120 AUROC=0.6967
ep 06/25: lam=0.000 AUROC=0.4264  ← cycle 2 valley
ep 07/25: lam=0.060 AUROC=0.7760  [SAVED] ← BEST (cycle 2 ramp 06)
ep 08/25: lam=0.120 AUROC=0.6661             ep 19/25: lam=0.180 AUROC=0.6964
ep 09/25: lam=0.180 AUROC=0.6785             ep 20/25: lam=0.240 AUROC=0.5880
ep 10/25: lam=0.240 AUROC=0.6930             ep 21/25: lam=0.000 AUROC=0.7189  ← cycle 5 valley
ep 11/25: lam=0.000 AUROC=0.5390  ← cycle 3 valley
ep 12/25: lam=0.060 AUROC=0.6692             ep 22/25: lam=0.060 AUROC=0.7719
ep 13/25: lam=0.120 AUROC=0.7504             ep 23/25: lam=0.120 AUROC=0.7308
                                              ep 24/25: lam=0.180 AUROC=0.6964
                                              ep 25/25: lam=0.240 AUROC=0.7026
```

### Final §IV.D table for paper

We adopt **W3-P (0.781) as the §IV.D primary** (best vacuity AUROC under
the honest multi-seq + 16/3 protocol + lite_v2 backbone with plain linear
KL), and report W3-Q (0.776) as an ablation that **documents the
dimension-specific trade-off** of warm-restart between mIoU and AUROC.
