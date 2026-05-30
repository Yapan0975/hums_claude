# Editorial Decision Letter — Round 1

## Decision: **Major Revision**

Per IRON RULE #4 (Devil's Advocate found 3 CRITICAL issues), Accept is forbidden. All 5 reviewers independently recommend Major Revision with no dissent.

## Synthesis of Reviewer Recommendations
| Reviewer | Role | Recommendation | Confidence | Weighted |
|---|---|---|---|---|
| EIC | IROS PC chair | Major Revision | 4/5 | 70 |
| R1 | Methodology (CMU) | Major Revision | 5/5 | 65 |
| R2 | Domain (MIT-SPARK) | Major Revision | 5/5 | 63 |
| R3 | Perspective (MIT CSAIL) | Major Revision | 3/5 | 63 |
| R4 | Devil's Advocate | Major Revision (stretch Reject) | — | — |
| **Consensus** | | **Major Revision** | | **~65** |

## Consensus findings (multi-reviewer)

### C1 (CRITICAL) — Two-checkpoint / two-KL-schedule / best-ckpt oracle breaks "one model, three jobs"
Raised by R1 W2 + R3 W2 + R4 CRITICAL #1 (independently re-derived 3 times). Paper-internal contradiction: §V.B(d) admits 23.32% mIoU (W3-M) and 0.781 AUROC (W3-P) come from differently-trained systems, but §I.B + §VI co-list them. W3-Q proves unified schedule gives 0.776 < 0.781.

### C2 (CRITICAL) — R2 baseline asymmetric protocol upgrade + R2-reimpl faithfulness
Raised by R1 W1 + R2 W3 (independent angles, convergent conclusion). 5 asymmetric knobs upgraded for W3-M but not for R2 baseline; no reproduce.yaml; "CE PointNet argmax" is not R2's iterative Bayes filter.

### C3 (CRITICAL) — Devil's Advocate "vacuity-in-particular load-bearing" challenge unrefuted
Raised by R4 CRITICAL #2 + R3 W1 (substantively supports). Decoupling Ablation specified by R4: vacuity-everywhere vs dissonance-for-decay vs softmax-entropy-for-M2. Pareto-dominance pass condition.

### C4 (MAJOR) — Submission-window-vs-scope risk + 132% page budget
EIC W1 + W3. Path-1 strongly preferred; Path-4 at 23.32% NOT IROS-acceptable per EIC.

### C5 (MAJOR) — §I.C C1/C2/C3 + §VI extrapolation over-reach
EIC W4 + R2 W4 + R2 W3 + R4 W7 + R3 W5. 4 reviewers — densest consensus after C1.

### C6 (MAJOR) — SLIM-VDB "first comparable ECE" mis-claim + missing prior-art (R2 W5)
Factual error; reputational cost; single-reviewer-raised but expected for venue.

### C7 (MAJOR) — No closed-loop / decision-relevance demonstration (R3 W3 + R4 So-What M2)

### C8 (MAJOR) — Rare-class evaluation un-testable at preliminary backbone (R1 W4 + R4 W6)

### C9 (MAJOR) — Spatial-kernel rejection un-ablated (R2 W1 + R4 W4)

## Path-to-Accept (7 numbered conditions)

This paper moves Major Revision → Accept in round 2 if and only if:

1. **Single-checkpoint joint-serving** (P0-A1 / W3-R) — unified KL schedule; one checkpoint; all 4 headline metrics; rewrite abstract/§I.B/§VI; withdraw co-listed W3-M+W3-P framing
2. **R2 matched-protocol 4-row matrix + iterative-Bayes reconstruction row** (P0-A2 / W3-S, W3-T) — drop "generalises R2 Bayes filter" unless proved
3. **Decoupling Ablation** (P0-A3 / W3-U, W3-V, W3-W) — Pareto-dominance verdict; if fail, reframe §I.B
4. **§I.C C3 + §VI rewrite + Path declaration** (Path-1 OR Path-4 with mandatory rewrite)
5. **KITTI-360 M2 single-revisit verification with β-sweep** (P0-A4 / W3-X) — closes leg (ii)
6. **All P1 items B1-B8** including SLIM-VDB factual fix, R2-reimpl temperature scaling
7. **Page budget compliance** (8 pages IEEE)

**Conditions 1, 2, 3 are NON-NEGOTIABLE.** Failing any → Reject in round 2.

## P0 estimated cost (RTX 5090)
- W3-R: 6-10 GPU-hr (single-checkpoint joint serving)
- W3-S + W3-T: 8-12 GPU-hr + ~1 week R2 reconstruction engineering
- W3-U, W3-V, W3-W: ~2 hr (Decoupling Ablation)
- W3-X: 6-10 GPU-hr + KITTI-360 data prep
- **Total P0**: ~25-40 GPU-hr + ~2 weeks engineering

Achievable within 10-month window. Path-1 closure is the strategic gate.

## Final note for authors
The strengths are real and would be costly to discard:
- Eq. 12 conjugate-Dirichlet closed form (R3 S1, R4 acknowledged)
- §V honest audit trail (R-EIC S3, R1 S1, R3 S2, R4 acknowledged)
- nvblox `EvidentialLayer<VoxelType>` (R2 S2, R4 acknowledged)
- Tab II citable lineage table (R2 S5)

Reject pressure exists because strengths sit inside an integration thesis (§I.B) that authors' own data (W3-Q, §V.B(d), §V.C) do not currently support. **Conditions 1, 2, 3 are non-negotiable.** Path-to-Accept is feasible in 10 months.

If Decoupling Ablation fails Pareto-dominance, please do not argue around the result — reframe §I.B in the same revision. Panel rewards honest reframe far more than defensive rescue, and §V.C demonstrates the authors already know how.

— *Editorial Synthesizer, on behalf of the 5-reviewer Phase 1 panel, 2026-05-30*
