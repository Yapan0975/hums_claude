# Round-2 Re-Review Report — Decision: ACCEPT with Minor Revision

(Full re-review by Editorial Synthesizer in re-review mode; saved verbatim from `/ars-revision` background agent.)

## Manuscript Information

| Field | Value |
|---|---|
| Title | EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay |
| Target venue | IROS 2027 |
| Round-1 decision | Major Revision (weighted 65/100; 5/5 reviewers; 3 DA CRITICAL) |
| Round-2 submission date | 2026-05-31 |
| Revised manuscript | `drafts/paper_v2.md` (738 lines, ~13.5k words) |

## Final Decision: **ACCEPT with Minor Revision** · Confidence 4/5

Round-1 IRON RULE #4 unblock satisfied: all 3 DA CRITICAL independently verified RESOLVED against revised manuscript and supplementary JSON / log artefacts.

## Tally

- **13 RESOLVED** verified
- **11 DELIBERATE_LIMITATION** verified
- **1 REVIEWER_DISAGREE** rebuttal accepted
- **0 UNRESOLVABLE**
- **0 contested** by re-reviewer
- **3 DA CRITICAL → all closed**
- **4 new minor issues (N1-N4)** surfaced — minor textual / supplementary gaps, do NOT block Accept

## New issues from round-2 revisions (camera-ready actions)

- **N1**: §VI mention of W3-X omits synthetic-Dirichlet caveat (W3-X uses KITTI-360 GT labels + confidence as the per-voxel Dirichlet stand-in, not the deployed W3-R ckpt run on KITTI-360). Action: 1-clause caveat at §VI.
- **N2**: W3-R single-ckpt AUROC = 0.706 sits 0.075 below the W3-P 16/3 robustness operating point. Add 1 sentence in §V.C quantifying this joint-serving cost.
- **N3**: `reproduce.yaml` referenced in Tab III / response letter but not yet committed. Camera-ready.
- **N4**: Tab II row 4 still carries v2 `[TBD-RQ1] (target ≥ +2 vs R2)` without the W3-S anchoring footnote promised in tracking row 28.

## Three observations the re-reviewer highlighted

1. **All 3 DA CRITICAL verified RESOLVED** — IRON RULE #4 unblock satisfied. Particular commendation: W3-UVW Decoupling Ablation was *honestly run, honestly failed* Pareto-dominance pass condition on W3-M (vacuity F1 0.073 vs dissonance 0.425 at thr 0.5 — 5.9× margin against original §I.B thesis), and §I.B reframed *in-revision* exactly as round-1 Path-to-Accept condition 3 required. Follow-up training-protocol-dependence finding on W3-R is *stronger* than v2-draft framing.
2. **P1-B7 temperature-scaling result is most scientifically valuable round-2 addition**. Post-TS CE ECE 0.040 vs EDL 0.125 (3.1× reversal) *falsifies* §III.B "EDL ECE-win" thesis at adequate capacity. EDL value moves to (a) joint-serving + (b) dissonance-for-M2 — exactly the publishable nuance R1 W3 demanded.
3. **W3-R single-checkpoint joint-serving is the load-bearing systems contribution**. Per-epoch trace verifies pre-registered criterion selects ep-12 with all 4 metrics simultaneously; AUROC never enters anti-discriminative regime. Closes R1 W2 + R3 W2 + R4 DA-C1 cleanly.

## Five DL items legitimately punted to round 3

- W3-T iterative-Bayes R2-reimpl reconstruction (~1 week eng)
- A-8 hybrid spatial-kernel ablation (now executed as W3-Y — Pareto FAIL, §III.B rejection survives)
- W3-W Khronos native-config benchmark
- Page-budget compliance (~30% over IEEE 8 pages)
- AI disclosure statement (camera-ready)

## Required for camera-ready

1. Apply N1, N2, N3, N4 minor fixes
2. Execute §V.A v page-allocation trim
3. Add AI disclosure
4. Optional: run /ars-format-convert to IEEEtran (already pandoc → generic LaTeX done)
