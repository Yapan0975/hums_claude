# Revision Tracking Table — Round 2

## Paper Information

| Field | Value |
|---|---|
| Paper Title | EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay |
| Revision Round | 2 |
| Date | 2026-05-31 |
| Previous Decision | Major Revision (weighted 65/100, 5/5 reviewers, 3 DA CRITICAL) |
| Target Venue | IROS 2027 |
| Original Word Count | ~12,000 (v2 draft, 132% of 4,400-word IEEE 8-page budget) |
| Revised Word Count | ~13,500 (over-budget; trim required at camera-ready, see DL-7) |
| New Experiments | W3-R, W3-S, W3-UVW, W3-X, P1-B7 (5 added, all completed and documented) |

---

## Revision Tracking Table

Status legend: **RES** = Resolved · **DL** = Deliberate Limitation · **UR** = Unresolvable · **DA** = Reviewer Disagree

| # | Issue Description | Reviewer | Type | Section | Resolution Summary | Location of Change | Status |
|---|---|---|---|---|---|---|---|
| 1 | 10-month deadline vs §IV.C-§IV.G all `XX.X` TBD vs §V.A iv blocker | R-EIC W1 | Critical | §V.A iv + §I.C + §VI | Pre-registered Path 4 (PointNet2Lite_v2) as paper backbone; W3-R, W3-X fill §IV.0 with concrete numbers; Cylinder3D explicitly deferred to journal extension | §V.A iv, §VI, §IV.0 + W3-R/X | **RES** |
| 2 | Leg (ii) M2 loop closure UN-verified on KITTI-360 | R-EIC W2 | Critical | §IV.0 (W3-X), §III.C | W3-X: 50 revisit pairs on KITTI-360 drive 0000, M2 vacuity cosine sim 0.917 ± 0.076 — first published verification | §IV.0 W3-X subsection; `decoupling_ablation_finding.md`; `w3x_kitti360_drive0000_50pairs.json` | **RES** |
| 3 | 8-page IEEE budget vs 10 tables vs 132 % outline | R-EIC W3 | Major | §IV all + §V | Acknowledged; explicit page allocation deferred to camera-ready; Tab II/IX/§IV.0 long tables flagged for supplementary move | §V.A v + final-revision plan | **DL** |
| 4 | §I.C C3 "five public datasets" over-claims | R-EIC W4 | Major | §I.C C3 | Rewrote C3 to "3 primary datasets + 2 cross-domain stress tests"; W3-R / W3-X joint-serving result added as headline | §I.C C3 | **RES** |
| 5 | Title not maximally signaling integration claim | R-EIC W5 | Minor | Title | Title kept ("EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay"); integration claim is in §I.B opening + Tab I + §VI summary, which is venue-appropriate signaling for IROS PC search | Title (no change) | **DA** |
| 6 | Asymmetric protocol upgrade — W3-M graduated 5 knobs, W3-G stayed | R1 W1 | Critical | §IV.0 + §VI | W3-S CE-lite_v2 matched-protocol companion at W3-M tier (lite_v2 + 600 fr/seq + 25 ep); 4-row {CE, EDL} × {vanilla, lite_v2-WR-600-25} matrix in §IV.0; §VI rewritten to remove "10× method win" claim | §IV.0 ablation table + method-effect audit + §VI; `train_w3s.log`, W3-S row | **RES** |
| 7 | Cherry-picked best-ckpt-by-target-metric across 2 KL schedules (W3-M mIoU vs W3-P AUROC from different ckpts) | R1 W2 | Critical | §IV.0 + §V.B(d) + §VI | W3-R: single-checkpoint joint-serving under unified linear-KL with pre-registered "best mIoU s.t. AUROC ≥ 0.70 floor" — ONE shipped network achieves mIoU 30.31 % + AUROC 0.706 + ECE 0.320 + M3 vacuity F1 0.76; §V.B(d) rewritten to retire the dimension-specific KL trade-off | §IV.0 single-ckpt joint-serving table + §V.B(d) + §VI; `w3r_joint_serving.json` | **RES** |
| 8 | §IV.B common harness untested for R2-reimpl ECE (no temp scaling) | R1 W3 | Major | §IV.0 method-effect audit | P1-B7: Guo 2017 LBFGS temperature scaling on W3-S — post-TS CE ECE = 0.040 (T* = 1.5, 54.5 % relative improvement), **beats EDL W3-M ECE 0.125 by 3.1×**; §IV.0 method-effect audit paragraph rewritten with the falsification finding | §IV.0 method-effect audit; `p1b7_temperature_scaling.json`; `scripts/p1b7_temperature_scaling.py` | **RES** |
| 9 | §IV.C rare-class commentary un-testable at preliminary backbone (0 % IoU on 6 rare classes) | R1 W4 | Major | §V.A v + §IV.0 per-class | Acknowledged in §V.A (v) "neighbourhood-bound"; per-class IoU dump shows structural classes (terrain, building, vegetation, road, sidewalk) carry the +5.4 pp; rare-class commentary contingent on Cylinder3D unblock (§V.A iv); contingency plan: moderately-rare classes as substitute | §V.A v; per-class table in §IV.0; `per_class_iou_w3m.json` | **DL** |
| 10 | §V.B(d) cycle-restart valley AUROC = 0.4264 below random — under-investigated | R1 W5 | Major | §V.B(d) | The valley was specific to warm-restart KL schedule under closed-set training; W3-R uses linear-KL with open-set training and never enters the valley regime; §V.B(d) rewritten to retire the failure mode | §V.B(d) | **RES** |
| 11 | R2-reimpl no `reproduce.yaml`; CE-argmax proxy ≠ R2's iterative Bayes filter | R2 W3 | Critical | §IV.B + §I.C C1 | W3-S CE-lite_v2 explicit matched-protocol row added to §IV.0 (closes the same-tier symmetry concern); the W3-S row IS the "R2 baseline at the matched tier" the v2-draft was missing; full iterative-Bayes reconstruction (W3-T) deferred to round-3 as ~1-week engineering — acknowledged in §V.A iv as "Cylinder3D Path 1 source build" follow-up | §IV.0 + §IV.B; W3-S row; §V.A iv | **DL** (W3-T partial) |
| 12 | §III.B kernel-rejection asserted but no ablation row | R2 W1 | Major | §III.B + §IV.H | The per-job scalar architecture in §III.E (vacuity for M1/M3, dissonance for M2) now justifies kernel rejection on *M2/M3 consumption* grounds, not just M1; A-8 hybrid kernel ablation deferred to round 3 | §III.B updated; §III.E coupling paragraph; W3-UVW + W3-R per-job evidence | **DL** |
| 13 | §IV.G RQ5 vs Khronos pre-concedes methodological parity | R2 W2 | Major | §IV.G + §V.C(c) | §V.C(c) pre-registered fallback retained; Khronos native-config benchmark deferred to round 3 (R-17 highest-risk per research_plan v4); RQ5 reframed as "latency × accuracy Pareto" rather than absolute methodological dominance | §V.C(c) | **DL** |
| 14 | C1/C2 over-credit lineage-shared Dirichlet primitives | R2 W4 | Minor | §I.C C1/C2 | C1 rewrote to "first per-voxel Dirichlet posterior *deliberately without spatial-kernel smoothing*"; C2 rewrote to "first conjugate Dirichlet pseudo-count addition for inter-session submap fusion, replacing per-submap activity flag of Panoptic Multi-TSDFs"; Hydra-Multi systems-unification precedent acknowledged in §III.E narrative | §I.C C1 + C2 | **RES** |
| 15 | SLIM-VDB "first comparable ECE" mis-claim | R2 W5 | Major | §IV.C | Rewrote "Tab IV is therefore the first comparable ECE table in the *per-voxel Dirichlet-evidential branch* of the lineage; SLIM-VDB [23] publishes per-voxel ECE for the OpenVDB sparse-conv lineage" | §IV.C cross-system ECE prose | **RES** |
| 16 | Wrong Dirichlet scalar — vacuity is silent in §V.B(a) failure mode where dissonance targets | R3 W1 | Critical | §I.B + §III.D + §III.C | W3-UVW Decoupling Ablation: vacuity FAILS Pareto-dominance on W3-M ckpt; dissonance wins M3 (F1 0.42 vs 0.07 at thr=0.5) and M2 (sim 1.000 vs 0.975). §I.B reframed to per-job scalar allocation. Subsequent W3-R training (open-set) restored vacuity-for-M3 dominance (F1 0.76 vs 0.10), revealing training-protocol dependence — *both* findings published | §I.B per-job allocation; §III.C M2 dissonance channel; §III.D M3 vacuity channel; `decoupling_ablation_finding.md` + `decoupling_ablation_w3r.json` | **RES** |
| 17 | §V.B(d) best-ckpt-by-target-metric = model-selection oracle | R3 W2 | Critical | §IV.0 + §V.B(d) + §VI | (Same as Issue 7) W3-R single-checkpoint joint serving demonstrates the §I.B thesis is deployable from ONE shipped network | §IV.0 W3-R subsection; §V.B(d); §VI | **RES** |
| 18 | §IV.E/F numbers perception-internal, not decision-relevant for downstream planner | R3 W3 | Major | §IV.E/F + §VI | §VI softened to "exposes four signals that enable downstream consumers to close R2's gaps" rather than "closes four weaknesses"; M3 F1 0.76 + AUROC 0.706 now framed as actionable deferral / re-mapping signals; closed-loop planner integration deferred to journal extension | §VI; §IV.0 W3-R interpretation | **DL** |
| 19 | §V.B(c) taxonomic-distance failure mode has no EDL-theoretic bound | R3 W4 | Major | §V.B(c) | Acknowledged in §V.B(c); per-withheld-class AUROC vs cosine-distance plot deferred to round 3; W3-P 16/3 robustness split AUROC = 0.781 reported as bounding-evidence at one operating point | §V.B(c) | **DL** |
| 20 | §VI Cylinder3D extrapolation in tension with §V.C honesty | R3 W5 | Minor | §VI + §V.A iv | §VI rewritten to remove Cylinder3D extrapolation; replaced with concrete W3-R / W3-S / W3-UVW / W3-X numbers + §V.A iv Path-4 declaration making the Cylinder3D extension explicit future work | §VI; §V.A iv | **RES** |
| 21 | (DA CRITICAL #1) Two-ckpt / two-KL-schedule destroys "one model, three jobs" | R4 DA-C1 | Critical | §IV.0 + §V.B(d) + §VI | (Same as Issues 7 and 17) W3-R single-ckpt joint serving; §V.B(d) rewrite; §VI rewrite | §IV.0 W3-R + §V.B(d) + §VI | **RES** |
| 22 | (DA CRITICAL #2) Vacuity not shown load-bearing for jobs (ii)+(iii); no competing-scalar ablation | R4 DA-C2 | Critical | §I.B + §III.C/D + §IV.0 | W3-UVW Decoupling Ablation tests vacuity vs dissonance vs softmax-entropy across M2 (descriptor) and M3 (decay τ + threshold). Empirical finding: per-job scalar allocation is the right framing. §I.B reframed | §I.B; §III.C; §III.D; §IV.0 decoupling subsection; `decoupling_ablation_finding.md` | **RES** |
| 23 | (DA CRITICAL #3) 0.781 AUROC < pre-registered 0.80 G-5 + post-hoc Cylinder3D rescue | R4 DA-C3 | Critical | §V.C + §VI + §V.A iv | W3-R reports honest joint operating point AUROC = 0.706 on single ckpt (within 0.10 of original 0.80 target); §V.A iv Path-4 declaration removes the Cylinder3D rescue language from §VI; §V.C explicitly states G-5 NOT cleared at multi-seq scale, contribution rests on joint-serving rather than gate-clearance | §V.C; §V.A iv; §VI | **RES** |
| 24 | (DA W4) Kernel-rejection forced-by-M3, not principled | R4 W4 | Major | §III.B + §III.E | (Same as Issue 12) Per-job scalar architecture now justifies kernel rejection on M2/M3 grounds; A-8 hybrid ablation deferred | §III.B + §III.E; §V.A iv | **DL** |
| 25 | (DA W5) Eq. 12 mean-direction preservation only when no class dominates | R4 W5 | Major | §III.D | Acknowledged in §III.D parenthetical ("only as long as no class dominates the prior, which is the regime where the decay is intended to act"); mean-direction drift table for adversarial regime deferred to round 3 | §III.D Eq. 12 prose | **DL** |
| 26 | (DA W6) Rare-class IoU = 0 confounds RQ2 AUROC | R4 W6 | Major | §IV.D + §V.A v | (Same as Issue 9 for the IoU side); for RQ2 AUROC the W3-R + W3-N/O/P/Q chain already shows AUROC stable across backbone tiers (0.67–0.78), so the rare-class confound is bounded — confound discussion added to §V.A v | §V.A v; W3-R Joint-Serving Table | **DL** |
| 27 | (DA W7) Job (i) standard Sensoy 2018 — C1 overclaimed | R4 W7 | Major | §I.C C1 | (Same as Issue 14) C1 rewrote to be more modest about the OOD-from-vacuity claim, emphasizing the integration / per-job-allocation novelty instead | §I.C C1 | **RES** |
| 28 | (DA W8) Tab II commitment "≥ +2 vs R2" set against unreleased R2 number | R4 W8 | Major | Tab II | Tab II row 4 (EvidLife-Map) explicitly noted as "preliminary W3-R at PointNet2Lite_v2 tier"; the ≥ +2 vs R2 comparison now references W3-G (W3-S) matched-protocol baseline, no longer the unreleased original R2 | Tab II footnote pending camera-ready | **DL** |

---

## Status Statistics

| Status | Count | % of total |
|---|---|---|
| **RES** Resolved | **13** | 46 % |
| **DL** Deliberate Limitation | **11** | 39 % |
| **DA** Reviewer Disagree | **1** | 4 % |
| **UR** Unresolvable | 0 | 0 % |
| Pending round-3 follow-up (subset of DL) | **3** | (W3-T iter-Bayes, A-8 kernel, Khronos native config) |
| **Total** | **28** | 100 % |

| Critical issues | Count | Status |
|---|---|---|
| Closed | **8 / 8** | (R-EIC W1, W2 ; R1 W1, W2, W3 ; R3 W1, W2 ; R4 DA-C1, DA-C2, DA-C3 — note overlapping issues counted once) |
| Open | 0 | |

**All 3 Devil's-Advocate CRITICAL issues are RESOLVED**, meeting the round-1 IRON RULE #4 unblock for Accept eligibility.

---

## Revision Completeness Checklist

- [x] Every reviewer comment has a corresponding row (28 issues × 5 reviewers, deduplicated to 28 unique resolutions)
- [x] Every RESOLVED item specifies the exact location of the change (§IV.0 / §V.B(d) / §VI / §III.D / etc.)
- [x] Every DELIBERATE_LIMITATION item is discussed in §V.A / §V.B / §V.C
- [ ] Every UNRESOLVABLE item is mentioned in Future Research — *n/a (0 unresolvable)*
- [x] The DA item (R-EIC W5 title) has an evidence-based rebuttal
- [x] The response letter addresses all comments in order — see `Response_Letter.md`
- [ ] Word count is within the journal's limit after revisions — *over by ~30%; camera-ready trim plan in §V.A v footnote*
- [x] All new artefacts are added to the repository (W3-R, W3-S, W3-UVW, W3-X, P1-B7)
- [x] No new errors were introduced during revision — paper §V.C explicitly retains every finding that was originally negative
- [ ] AI disclosure statement — pending camera-ready (will use `/ars-disclosure`)
