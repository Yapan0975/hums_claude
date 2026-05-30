# Phase 1 — R1 Methodology Review

**Persona**: CMU Robotics Institute Associate Professor, NeurIPS 2024 Area Chair, IJRR/T-RO reviewer for 30+ manuscripts. Documented protocol-fault retractions in AV benchmarking.

**Recommendation**: Major Revision  
**Confidence**: 5/5  
**Weighted average**: 65/100

## Critical
- **W1 (Critical)** — Asymmetric protocol upgrade: §VI "23.3% mIoU vs matched-capacity R2 at 2.3%" is technically false. W3-M graduated along 5 knobs (lite_v2 backbone, warm-restart KL, 600 fr/seq, 25 epochs, best-ckpt-by-mIoU) while R2 stayed at W3-G vanilla + linear-CE + 300 fr/seq + 20 epochs. No "R2 + lite_v2 + WR + 600 fr" companion exists. Run W3-G+ (CE-lite_v2-WR-600-25); report 4-row {CE,EDL}×{vanilla, lite_v2-WR-600} matrix.
- **W2 (Critical)** — Cherry-picked best-ckpt-by-target-metric across 2 incompatible KL schedules. 23.32% mIoU (W3-M) and 0.781 AUROC (W3-P) come from DIFFERENT trained systems but co-listed in abstract/§VI. W3-Q proves unified schedule gives 0.776 < 0.781. Report single-schedule Pareto frontier.

## Major
- **W3** — §IV.B "common harness" promise unverified for R2-reimpl: 4-7× ECE win compares EDL-tuned head vs softmax-Bayes head with no temperature scaling. Need Guo 2017 temperature scaling + best-ckpt-by-ECE for R2.
- **W4** — §IV.C "rare-class commentary" un-testable at preliminary: bicycle/person/bicyclist/trunk/pole/traffic-sign all 0% IoU at W3-H AND W3-M. Substitute moderately-rare classes (building, terrain, sidewalk); add mean-vacuity bicycle-vs-car probe.
- **W5** — §V.B(d) cycle-restart valley AUROC=0.4264 BELOW RANDOM, anti-discriminative. Currently under-investigated.

## Dimension scores
| Dim | Score | Descriptor |
|---|---|---|
| Originality (20%) | 72 | Strong |
| Methodological Rigor (25%) | 58 | Adequate |
| Evidence Sufficiency (25%) | 52 | Adequate |
| Argument Coherence (15%) | 78 | Strong |
| Writing Quality (15%) | 80 | Strong |
| **Weighted avg** | **65** | Major Revision |

**Note**: Recoverable with single W3-G+ run + Pareto-frontier reframing. Goes from Major→Accept-Minor without changing §III.
