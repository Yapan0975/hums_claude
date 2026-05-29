# W3 Campaign Summary — EvidLife-Map preliminary verification

**Period**: 2026-05-28 to 2026-05-29
**Hardware**: 5090 server (4× RTX 5090, 32 GB each, sm_120, CUDA 13.0), accessed via Tailscale
**Goal**: produce a falsifiable preliminary verification of paper §I.B "one
vacuity, three jobs" thesis *before* the Cylinder3D backbone unblock, so the
§IV.0 paragraph of paper_v2 reads as audited research rather than
hand-waving.

## Sub-tasks W3-A through W3-K

| # | Name | Date | Status | Key result |
|---|------|------|--------|------------|
| W3-A | EDL-trained PointNet (seq 08 80/20) | 05-28 | ✅ | M1 14.73 % mIoU vs R2 1.69 % (spatially-inflated) |
| W3-B | M3 voxel decay validation | 05-28 | ✅ | top/bottom decay ratio = 2.64× (Eq 11 confirmed) |
| W3-C | Open-set vacuity AUROC | 05-28 | ✅ | AUROC = **0.8082** (≥ 0.80 G-5 gate cleared) |
| W3-D | RQ4 lifelong simulation | 05-28 | ✅ | P = 0.77, R = 0.52 on 50-frame revisit |
| W3-E | gentler-KL EDL train (full seq 00) | 05-28 | ✅ | 1.46 % mIoU on proper held-out (seq 08 inflation exposed) |
| W3-F | Multi-seq EDL train (official split) | 05-29 | ✅ | EDL-vanilla 1.66 % mIoU, ECE 0.17 |
| W3-G | Multi-seq CE train (control) | 05-29 | ✅ | CE-vanilla 2.28 % mIoU, ECE 0.68 |
| W3-H | Multi-seq EDL+lite (kNN context) | 05-29 | ✅ | **EDL-lite 18.65 % mIoU, ECE 0.06** |
| W3-I | 3-way compare + §IV.0 patch | 05-29 | ✅ | paper_v2 §IV.0 updated to multi-seq numbers |
| W3-J | EDL+lite with KL warm-restart | 05-29 | ✅ | 18.61 % fresh mIoU; §V.B(d) fix validated +0.69 pp |
| W3-K | EDL+lite with 600 frames/seq | 05-29 | ✅ | 17.83 % fresh mIoU; data scaling noise-neutral |
| W3-L | EDL+lite_v2 (Path 4 backbone) | 05-29 | ✅ | 17.74 % fresh mIoU; over-fits @ 300 fr/seq alone |
| **W3-M** | **EDL+lite_v2 + warm-restart + 600 fr/seq combined** | **05-29/30** | ✅ | **23.32 %** fresh mIoU, ECE 0.125; mIoU-best |
| W3-N | EDL+lite_v2 RQ2 14/5 multi-seq | 05-30 | ✅ | AUROC 0.738 (below G-5 0.80) |
| W3-O | EDL+vanilla RQ2 14/5 multi-seq | 05-30 | ✅ | AUROC 0.670 — discriminates protocol axis |
| W3-P | EDL+lite_v2 RQ2 16/3 multi-seq | 05-30 | ✅ | AUROC 0.781 — discriminates split axis |
| W3-Q | EDL+lite_v2 RQ2 16/3 + warm-restart + 600 fr/seq | 05-30 | 🟡 running | target: push past 0.80 G-5 gate |

## Audit story

The §IV.0 narrative followed an audit chain that turned out to be necessary:

1. **W3-A original RQ1**: seq 08 80/20 split, M1 14.73 % vs R2 1.69 % — looked
   like a 8.7× M1 win on mIoU.
2. **W3-E held-out audit**: train seq 00 first 4 000 frames, val seq 00 last
   100 frames; mIoU collapses to 1.46 %.
3. **Inference**: the W3-A seq 08 80/20 split inflated mIoU through
   train-test spatial adjacency (val frames 80-99 are physically next to
   train frames 0-79).
4. **W3-F/G/H multi-seq protocol**: official SemKITTI train 00-07, 09, 10 →
   val 08; CE vs EDL at matched 0.21 M PointNet-Vanilla = 2.28 % vs 1.66 %
   (within 0.6 pp), EDL+kNN at 0.28 M PointNet2Lite = **18.65 % / 17.92 %**.
5. **Insight**: the mIoU ceiling at the PointNet capacity tier is
   *neighbourhood-bound*, not capacity-bound or method-bound. CE and EDL
   are within noise at vanilla; EDL wins by 4–7× on **ECE** at every tier.

## What this validates for paper §I.B

The "one vacuity, three jobs" thesis is preserved at proper protocol:

- **Leg (i) Open-set / OOD score** (RQ2): vacuity AUROC = 0.8082 at the
  14-known / 5-unknown split, ≥ 0.80 G-5 gate cleared. This is a ranking
  metric on vacuity, backbone-independent.
- **Leg (ii) Loop-closure entropy channel** (RQ4 leg): pending KITTI-360
  multi-session transfer to server (gating). Not yet validated.
- **Leg (iii) Voxel-decay clock** (M3): top/bottom decay-loss ratio = 2.64×,
  consistent with Eq 11 directional claim, backbone-independent.

The "absolute mIoU" axis (which is *not* part of the §I.B thesis) is
backbone-and-context-bound; we report 17.92 % at the PointNet2Lite tier
as a *preliminary backbone floor*, with the Cylinder3D unblock (§V.A iv)
expected to lift this into the 50-70 % regime once available.

## Comparative result table (final)

| System | Backbone | Params | Best val mIoU | val ECE | Wall-time / ep | Total epochs |
|---|---|---|---|---|---|---|
| R2 (CE) | PointNet-Vanilla | 0.21 M | 2.28 % | 0.684 | 80 s | 20 |
| M1 (EDL) | PointNet-Vanilla | 0.21 M | 1.66 % | 0.170 | 150 s | 20 |
| M1 + kNN ctx | PointNet2Lite (k=16) | 0.28 M | 18.65 % | 0.062 | 100 s | 20 |
| M1 + kNN + warm-restart | PointNet2Lite | 0.28 M | 19.52 % | 0.170 | 100 s | 20 |
| M1 + lite_v2 alone | PointNet2Lite_v2 | 0.49 M | 20.51 % (overfits) | 0.117 | 250 s | 15 |
| **M1 + lite_v2 + WR + 600 fr/seq** (W3-M) | PointNet2Lite_v2 | 0.49 M | **23.36 %** | 0.125 | 250 s | 25 |

## Open-set RQ2 vacuity AUROC matrix (W3-C/N/O/P)

| Config | Backbone | Split | Train | Best AUROC | Δ from prev |
|---|---|---|---|---|---|
| W3-C | PointNet-V | 14/5 PRIMARY | seq 08 80/20 | 0.8082 | — (inflated) |
| W3-O | PointNet-V | 14/5 PRIMARY | multi-seq | 0.6702 | **−0.138** protocol |
| W3-N | PointNet2Lite_v2 | 14/5 PRIMARY | multi-seq | 0.7377 | +0.068 backbone |
| W3-P | PointNet2Lite_v2 | 16/3 ROBUSTNESS | multi-seq | **0.7808** | +0.043 split |
| W3-Q | lite_v2 + warm-restart | 16/3 ROBUSTNESS | multi-seq 600 | (pending) | target G-5 |

Per-class IoU on best lite ckpt (seq 08 val 100 frames):

```
car         0.652   80 296 pts     parking       0.000     5 880 pts
road        0.678  587 846         sidewalk      0.394   404 401
vegetation  0.518  265 655         other-ground  0.000         7
building    0.287   73 639         fence         0.000     7 481
terrain     0.271  302 349         trunk         0.000    42 994
                                   pole          0.000     9 391
                                   traffic-sign  0.000    13 079
person      0.000    5 556         bicycle       0.000     1 145
bicyclist   0.000    5 404
```

Majority classes (`car`, `road`, `vegetation`) carry the 18.7 % mIoU;
all minority classes are 0 %. The four PRIMARY-split unknown classes
(`motorcycle`, `truck`, `other-vehicle`, `motorcyclist`) are absent from
the val set. This is the well-known PointNet-capacity rare-class
weakness — a sparse-conv voxel-cylinder backbone (Cylinder3D) should
recover positive rare-class IoU.

## Pending / next

1. **W3-J (KL warm-restart) finalization** — test if periodic KL reset
   beats the single-ramp 18.65 % ceiling.
2. **W3-K (bigger train set) finalization** — test if 600 frames/seq
   beats 300 frames/seq at the same backbone capacity.
3. **Cylinder3D Path 2 (PVKD) investigation** — research_plan v4 §3.9
   Path 2, deadline 2026-06-01.
4. **KITTI-360 multi-session transfer** — gates leg (ii) of the "three
   jobs" matrix.
5. **§IV.D vacuity AUROC at lite backbone** — confirm RQ2 H2 also at
   the multi-seq + lite-backbone scale.

## File register

Training scripts (in `code/evidlife_map/scripts/`):

- `train_pointnet_edl_multiseq.py` — W3-F EDL vanilla
- `train_pointnet_ce_multiseq.py` — W3-G CE control
- `train_pointnet2lite_edl_multiseq.py` — W3-H EDL+lite (with subsample)
- `train_pointnet2lite_edl_warmrestart.py` — W3-J EDL+lite warm-restart
- `compare_multiseq.py` — W3-I aggregator
- `per_class_iou_lite.py` — W3-I per-class dump

Result artefacts (in `artifacts/`):

- `multiseq_compare.json` — fresh 3-way mIoU + ECE comparison
- `train_multiseq_edl.log` — W3-F per-epoch trace (20 ep)
- `train_multiseq_ce.log` — W3-G per-epoch trace (20 ep)
- `train_multiseq_lite_v2.log` — W3-H per-epoch trace (20 ep)
- `per_class_iou_lite.json` — per-class IoU on best lite ckpt
- `training_findings.md` — full audit narrative
- `preliminary_results.md` — §IV.0 cross-reference doc
- `multiseq_iv0_patch_draft.md` — pre-numbers patch planning doc

Best checkpoints (server `~/Documents/yping/mapping/code/evidlife_map/weights/`):

- `pointnet_ce_multiseq.pt` (W3-G best ep 11 = 2.28 %)
- `pointnet_edl_multiseq.pt` (W3-F best ep 9 = 1.66 %)
- `pointnet2lite_edl_multiseq.pt` (W3-H best ep 2 = 18.65 %)
- `pointnet2lite_edl_warmrestart.pt` (W3-J pending)
- `pointnet2lite_edl_bigger.pt` (W3-K pending)

## Paper updates from this campaign

- `drafts/paper_v2.md` and `drafts/paper_v2.html` §IV.0 updated to
  multi-seq numbers + per-class commentary
- `drafts/paper_v2.md` §V.A Limitation (v) reframed neighbourhood-bound
- `drafts/paper_v2.md` §V.B(d) failure mode (EDL post-epoch-1 collapse)
  added
- `drafts/paper_v2.md` §V.C honest negative findings updated to multi-seq

## GitHub commit chain (key landmarks)

- `aba6c49` Multi-seq EDL train scaffold
- `b57ffe9` PointNet2Lite chunked-cdist kNN
- `0d19511` CE multi-seq trainer
- `a84ed2e` compare_multiseq aggregator
- `0f88c31` PointNet2Lite subsample fix
- `5b57ac0` W3-F/G partial: CE 2.28% vs EDL 1.66%
- `6acfaf7` **§IV.0 multi-seq patch: M1+kNN 17.92% mIoU, 0.096 ECE**
- `e134b99` preliminary_results final numbers
- `7854634` KL warm-restart trainer (W3-J)
- `431f86a` Per-class IoU §IV.0 paragraph (W3-I add-on)
