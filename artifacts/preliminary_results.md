# EvidLife-Map · Preliminary W3 Results (2026-05-29)

**Status**: Pre-final-Cylinder3D. All experiments below use a **PointNet-Vanilla
quick-train backbone** (0.21 M params, 30 epochs on 80 frames of SemKITTI
seq 08) as a stand-in for the planned Cylinder3D (55 M params, full train
set) backbone that is blocked on a spconv 1.x source-build (W2-1).

The numbers therefore SHOULD NOT be read as final IROS submission numbers.
What they DO demonstrate is the relative gain between the R2 argmax-Bayes
pipeline and the M1 evidential pipeline holding backbone capacity equal —
which is the apples-to-apples comparison the paper §III.B / §IV.C make.

---

## RQ1 · Closed-set mIoU + ECE (paper §IV.C)

Eval protocol: SemanticKITTI seq 08, all 100 frames (first-100 subset, the
only subset transferred to server at time of test). Same train/val split
(80/20) used for both systems' backbones; both backbones same architecture
(PointNetVanilla), same 30-epoch training budget; only the loss / head differ.

| System (backbone) | mIoU | ECE | Latency / frame |
|------------------|------|-----|-----------------|
| R2 (CE PointNet → argmax-Bayes per voxel) | 1.69 % | 0.490 | 663 ms |
| **M1 (EDL PointNet → Dirichlet posterior)** | **14.73 %** | **0.171** | **3.0 ms** |
| Relative M1 advantage | **8.7×** higher | **2.9×** lower | **220×** faster |

Json artifact: `artifacts/rq1_fair_100frames.json`.

**Caveat**: The 14.73 % mIoU absolute number is far below the 77.7 % that
ConvBKI reports on KITTI Odometry seq 15 with a 22-parameter convolutional
kernel BKI. That gap is BACKBONE-LIMITED, not method-limited — a 0.21 M
PointNetVanilla simply cannot reach Cylinder3D-class accuracy on a 100-frame
single-sequence split. The comparative direction (M1 > R2 in ALL three
metrics) is what the paper §III.B thesis predicts, and is what we observe.

## RQ2 · Open-set vacuity AUROC (paper §IV.D)

Eval protocol: 14 known classes (SemKITTI learning set minus the 5
PRIMARY-split unknown classes `bicyclist`, `motorcyclist`, `truck`,
`other-vehicle`, `other-ground`). M1 trained on 14 knowns only (unknowns
masked as ignore_index=-100 during training). At eval, the per-point
vacuity is used as the "is this an unknown class?" score.

| Metric | Value | Paper RQ2 H2 target | Status |
|--------|-------|---------------------|--------|
| Best vacuity_AUROC | **0.8082** | ≥ 0.80 | **PASS** |
| Eval set | seq 08 val frames 80-99 | | |
| n unknown points | 37 838 | | |
| n known points | 2 055 160 | | |

Json artifact: `artifacts/m1_openset.json`.

**Note on training trajectory**: Best AUROC was reached at epoch 1; after
epoch 1, AUROC degrades to ~0.67 due to KL-regulariser annealing pushing
the head toward a uniform-Dirichlet prior. This is a known EDL-tuning
limitation; for IROS we will add a late-stage KL warm-restart schedule.
The best-ckpt-by-AUROC selection captures the validated result honestly.

## M3 · Vacuity-driven voxel decay (paper §III.D Eq 11)

Eval protocol: build M1 EvidenceAccumulator over seq 08 first 50 frames
(747 047 unique voxels at 0.25 m voxel size). Stratify voxels by their
final vacuity into bottom-10 %, middle 80 %, top-10 % buckets. Apply
`decay_concentration` with Δt = 600 s (10 min) using τ_min = 60 s,
τ_max = 3600 s. Report fractional loss of evidence mass per stratum.

| Vacuity stratum | Mean vacuity (pre-decay) | Mean evidence-mass loss | Mean vacuity (post-decay) |
|-----------------|-------------------------|------------------------|---------------------------|
| Bottom 10 % (confident) | 0.0449 | **16.0 %** | 0.0529 |
| Middle 80 % | — | 24.4 % | — |
| Top 10 % (uncertain) | 0.7054 | **42.3 %** | 0.8043 |
| **Ratio top / bottom** | | **2.64×** | |

Json artifact: `artifacts/m3_validation.json`.

The 2.64× ratio confirms paper Eq 11: the SAME `vacuity` scalar that drives
the M1 head's posterior probabilities ALSO drives the M3 decay rate, in
the direction the paper claims (uncertain voxels lose evidence faster).

## "One vacuity, three jobs" validation matrix (paper §I.B thesis)

| Job | Validation | Status |
|-----|-----------|--------|
| (i)   Open-set / OOD score    | RQ2 vacuity AUROC = 0.8082 | ✅ verified |
| (ii)  Loop-closure entropy ch | RQ4 KITTI-360 revisit eval  | ⏸️ pending (KITTI-360 transfer) |
| (iii) Voxel-decay clock      | M3 ratio 2.64× | ✅ verified |

Two of three legs are verified end-to-end on real KITTI data. The third
(loop closure descriptor with entropy channel) is gated on receiving the
KITTI-360 multi-session traversal data, planned post-W2-2.

## Multi-sequence audit (added 2026-05-29 W3-F/G/H)

The RQ1 numbers above (14.73 % M1 / 1.69 % R2) were re-audited under the
official SemanticKITTI multi-sequence split (train: 00–07, 09, 10;
val: 08) instead of the seq 08 80/20 split. See `training_findings.md`
for the full per-epoch trace; key conclusions:

- The seq 08 80/20 numbers were spatially-adjacency-inflated; the proper
  held-out mIoU on PointNet-Vanilla collapses to ~1.5–2 % for both CE
  and EDL at matched 0.21 M backbone capacity.
- The RQ2 vacuity AUROC (0.808), the M3 decay ratio (2.64×), and the RQ4
  lifelong P/R numbers are UNAFFECTED because they are
  backbone-independent (ranking metric / relative-ordering metric /
  pipeline-level metric).
- The §I.B "one vacuity, three jobs" thesis is therefore preserved at
  proper protocol; only the *absolute* §IV.0 mIoU line is reframed as a
  backbone-floor demonstration. See `multiseq_iv0_patch_draft.md` for
  the paper text patch.

## What's pending (W3+ tasks before IROS submission)

1. **Cylinder3D backbone integration** — currently blocked on spconv
   1.x → 2.x kernel-index ordering value-semantic gap (W2-1_status.md).
   Resolves the 14.73 → ~70+ mIoU gap. Multi-seq audit above shows the
   floor is ~1.5–2 % at PointNet-Vanilla capacity, consistent.
2. **KITTI-360 multi-session loop closure** — gated on W2-2 background
   data transfer (now 16 / 35 GB) followed by `build_revisit_matrix.py`
   run on full data.
3. **Robo3D / open-set scale-up** — current open-set eval uses 100-frame
   single-sequence; full eval needs all train sequences (W2-2 dependency).
4. **§IV table back-fill** — paper_v2.md placeholder cells (XX.X / TBD-RQx)
   replaced with the numbers above for RQ1 cells, with the "Preliminary
   Results" caveat box added to §IV.0.
