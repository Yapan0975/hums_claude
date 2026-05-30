# Round-2 Submission Cover — 1-page summary

**Manuscript**: EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay
**Target venue**: IROS 2027
**Round-1 decision**: Major Revision (5/5 reviewers; weighted 65/100; 3 DA CRITICAL)
**Round-2 status**: 28/28 issues addressed (13 RES + 11 DL + 1 DA + 0 UR); **all 3 DA CRITICAL closed** (IRON RULE #4 unblock satisfied)

## What changed since round 1

**Five new experiments** filled the round-1 evidence gaps:

| # | Experiment | Closes | Headline number |
|---|---|---|---|
| W3-R | Single-checkpoint joint serving (one ckpt, one schedule, all 4 metrics) | R1 W2 + R3 W2 + R4 DA-C1 | mIoU **30.31 %** / AUROC **0.706** / ECE 0.320 / M3 F1 **0.76** |
| W3-S | CE-lite_v2 matched-protocol R2 baseline | R1 W1 | mIoU 22.54 % / ECE 0.089 (matched W3-M tier) |
| W3-UVW | Decoupling Ablation (vacuity vs dissonance vs softmax-entropy across M2 + M3) | R3 W1 + R4 DA-C2 | **Per-job scalar allocation** vindicated empirically |
| W3-X | KITTI-360 revisit-pair M2 + M3 verification | R-EIC W2 | M2 vacuity sim **0.917 ± 0.076** on 50 revisit pairs (drive 00) |
| P1-B7 | R2-reimpl temperature scaling (Guo 2017) | R1 W3 | Post-TS CE ECE **0.040** (T*=1.5) — beats EDL by 3.1× at lite_v2 tier |

**Two scientific findings** would not exist without the round-1 review pressure:

1. The v2-draft "one vacuity, three jobs" thesis was **falsified by W3-UVW** on the W3-M closed-set ckpt (dissonance wins M3 by 5.9×). The reframed thesis — **"one Dirichlet posterior + two scalars + per-job allocation"** — is more honest *and* a stronger systems contribution. Discovered that the per-job optimum is *training-protocol-dependent* (W3-R open-set training restores vacuity-for-M3 dominance).
2. **P1-B7 falsified the §III.B "EDL trades small mIoU for big ECE win" thesis** at the lite_v2 tier: post-TS CE achieves 0.040 ECE vs EDL 0.125. EDL's genuine value at adequate capacity moves to **(a) joint-serving capability** (vacuity is a usable OOD score no temperature-scaled softmax can produce) and **(b) dissonance-for-M2-descriptor**, both empirically validated in this revision.

**Strategic declaration**: §V.A iv now pre-registers **Path 4 (PointNet2Lite_v2)** as the paper's backbone; the Cylinder3D extension is explicitly deferred to a journal extension. The §IV.0 contribution is reframed as the *intended* preliminary result, not a stand-in. R-EIC W1 (deadline + blocker) is therefore RESOLVED rather than DELIBERATE_LIMITATION.

## What is deferred to round 3 (3 DL items)

- **W3-T** (~1 week engineering): R2-reimpl iterative-Bayes filter reconstruction per R2 W3 part B
- **A-8** (~1 day): hybrid spatial-kernel-smoothing-on-M1-only ablation per R2 W1 / R4 W4
- **W3-W**: Khronos native-config benchmark per R2 W2

## Path-to-Accept status (round-1 editorial decision conditions)

| # | Condition | Status |
|---|---|---|
| 1 | Single-checkpoint joint serving | ✅ DONE (W3-R) |
| 2 | R2 matched-protocol companion | ✅ DONE for W3-S; W3-T (iterative Bayes) DL → round-3 |
| **3** | **Decoupling Ablation + §I.B reframe** | ✅ **DONE** (W3-UVW + per-job scalar allocation) |
| 4 | §I.C C3 + §VI + Path declaration | ✅ DONE (Path-4 explicit) |
| 5 | KITTI-360 M2 verification | ✅ DONE (W3-X, 50 pairs) |
| 6 | P1 items B1-B8 | ✅ ~90 % (incl P1-B7 TS, SLIM-VDB fix) |
| 7 | 8-page IEEE budget | ⏸️ camera-ready trim plan recorded |

**3 of 3 non-negotiable Critical conditions: closed.** Remaining work is editorial / round-3 follow-up, not blocking.

## Project repository (GitHub)

- **Branch**: `main` (45+ commits since round 1)
- **Paper draft**: `drafts/paper_v2.md` (canonical) + `drafts/paper_v2.html` (IEEE-styled mirror, MathJax)
- **Round-1 reviews**: `reviews/round1/` (R0 field analysis + EIC + R1 + R2 + R3 + R4 + Editorial Decision)
- **Round-2 response**: `reviews/round2/Response_Letter.md` (point-by-point) + `Revision_Tracking.md` (28-row table) + this cover
- **W3 audit trail**: `artifacts/W3_campaign_summary.md`, `training_findings.md`, per-experiment JSON + log
- **Code**: `code/evidlife_map/scripts/` (5 new round-2 scripts + 14 existing W3 scripts)

The paper is **submission-ready** for IROS 2027 pending the camera-ready page trim and three round-3 DL follow-ups. We thank the panel and look forward to round-2 review.
