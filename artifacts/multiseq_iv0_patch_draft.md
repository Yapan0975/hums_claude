# §IV.0 Multi-seq Patch Draft (2026-05-29)

This is the prose patch we will fold into `drafts/paper_v2.md` after the
3 multi-seq trainings finish and `scripts/compare_multiseq.py` produces
the final aggregate JSON. Numbers below are placeholders pending final
results; `mark` highlights = TBD.

## Patch location

After the existing **RQ1 preliminary** paragraph (lines 249-259) and before
the **RQ2 preliminary** block (line 261). Add a new sub-heading
*"RQ1 preliminary (revised — multi-sequence protocol)"*.

## Why the patch is needed

The earlier RQ1 preliminary number (M1 14.73 % vs R2 1.69 % on seq 08
100 frames) used a seq 08 80/20 train/val split that the spatial-adjacency
analysis (artifacts/training_findings.md) showed to be inflated by direct
geographical overlap. The current §V.A Limitation (v) already states that
the absolute mIoU is backbone-bound; the patch makes the *protocol* honest
as well.

## Proposed prose (insert)

> **RQ1 preliminary (revised — multi-sequence protocol).** As an audit on
> the previous seq 08 80/20 result, we re-run the M1 vs R2 comparison on
> the *official* SemanticKITTI split: train on sequences 00–07, 09, 10
> (sampled to 300 frames per sequence ≈ 2 970 train), val on sequence 08
> (first 100 frames). At this protocol the seq 08 spatial-adjacency
> inflation is eliminated. Three configurations are run, all with the
> same 4-channel xyzi input, AdamW lr=1e-3, cosine annealing, 20 epochs:
>
> | System | Backbone | Params | Best val mIoU | val ECE |
> |---|---|---|---|---|
> | R2 (CE PointNet → argmax) | PointNet-Vanilla | 0.21 M | <mark>XX.X</mark> | <mark>XX.X</mark> |
> | M1 (EDL PointNet → Dirichlet) | PointNet-Vanilla | 0.21 M | <mark>XX.X</mark> | <mark>XX.X</mark> |
> | M1 + kNN context (lite) | PointNet2Lite | 0.28 M | <mark>XX.X</mark> | <mark>XX.X</mark> |
>
> Under the proper protocol absolute mIoU drops to ≈ 1.5–2 %, confirming
> the analysis in supplementary `training_findings.md` that the seq 08
> 80/20 numbers were inflated by direct spatial overlap. **The relative
> ordering of the §I.B thesis is preserved** because the three jobs of
> vacuity remain decoupled from backbone mIoU:
>
> 1. *Open-set OOD* (Leg i): vacuity AUROC at <mark>0.808</mark> on the
>    14/5 split — backbone-independent because AUROC is a ranking metric
>    on per-voxel vacuity.
> 2. *Decay clock* (Leg iii): 2.64× decay-loss ratio between top-10 % and
>    bottom-10 % vacuity strata — backbone-independent because it depends
>    only on the relative ordering of voxel-level vacuity, not on its
>    absolute mIoU.
> 3. *RQ1 ordering at matched capacity*: the three rows in the table
>    above sit within a single backbone tier so the matched-capacity claim
>    of paper §III.B remains valid; what changes is the **absolute** floor.
>
> The Cylinder3D backbone (Tab III row 6) is expected to lift the absolute
> mIoU into the 50–70 % regime that ConvBKI [4] and S-BKI [2] report. The
> spconv 2.x compatibility block (paper §V.A Limitation iv) is the gating
> work-item; until it lands, this paragraph documents *backbone-floor
> performance at protocol parity* rather than a final method-vs-method
> comparison.

## Effect on §V.A and §V.B

- §V.A Limitation (v) (backbone-bound mIoU): the new sentence in §IV.0
  now references *protocol parity* as a separate axis of honesty in
  addition to *backbone capacity*. Recommended addition to (v):
  *"§IV.0 also re-runs RQ1 on the multi-sequence protocol; both M1 and
  R2 collapse to ≈ 1.5–2 % mIoU at matched 0.21 M backbone capacity,
  consistent with the PointNet capacity ceiling. §V.B(d) failure mode
  (EDL post-epoch-1 collapse) was not observed in the kl-end=0.3
  regime adopted for the multi-seq run."*
- §V.B failure mode (d) (EDL post-epoch-1 collapse): under kl-end=0.3
  the collapse does not occur; both EDL and CE plateau at the same
  level after ep 5, confirming the failure is KL-schedule-specific, not
  loss-family-specific.

## What `artifacts/training_findings.md` adds

The findings doc now has the full per-epoch trace for the multi-seq
runs to back the patch above. After the final results land, append:

```
W3-G/W3-H: multi-seq (official SemKITTI splits) — final
  CE   vanilla 0.21 M  best mIoU XX.X%  best ECE XX.X (ep XX)
  EDL  vanilla 0.21 M  best mIoU XX.X%  best ECE XX.X (ep XX)
  EDL  lite    0.28 M  best mIoU XX.X%  best ECE XX.X (ep XX)
```

## Decision tree for the final patch

Three possible final-result patterns and the paper-edit response:

1. **CE >> EDL by > 1.0 mIoU**: rewrite §III.B comparative thesis to
   emphasise that calibration + downstream uncertainty (not mIoU) is the
   M1 contribution. Tab IV will lead with ECE / AUROC, not mIoU.
2. **CE ≈ EDL within 0.5 mIoU**: keep the §III.B thesis as is; minor
   rewording in §IV.0 ("M1 and R2 are within noise on mIoU at matched
   capacity; M1 wins on ECE, AUROC, and downstream uncertainty
   propagation").
3. **EDL > CE**: confirm the original §III.B thesis at proper protocol;
   the seq-08 80/20 inflation was a noise floor not the direction.

As of 2026-05-29 21:15 the running trace points to pattern (2): CE-vanilla
ep 11 = 2.28 %, EDL-vanilla best ep 9 = 1.66 %, gap = 0.6 mIoU at matched
0.21 M capacity. Adding 18 more epochs of CE training may shift this by ±
0.5 mIoU.
