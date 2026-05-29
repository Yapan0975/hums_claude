---
version: v2 (2026-05-28) — supersedes v1 (radar axis dropped due to K-Radar GT constraint)
v1 file retained: paper_outline_v1.md (same directory; not modified)
v2 companion plan: research_plan.md (v2; v1 frozen at research_plan_v1.md)
---

# Paper Outline v2

**Title:** *EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay*

**Target:** IROS 2027, 8 pages double-column IEEE conference format, ≤ 30 references.
**Companion:** `research_plan.md` (v2; this directory).
**Conventions:** word counts are upper bounds per IROS double-column (≈ 600-700 words / column on the IROS template; tighter than RA-L's 700-750). Citation tokens like `[A1]`, `[C9]`, `[B4.1]` map to the entries of `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md`; only those are allowed.

---

## Abstract (1 column, ~200 words)

**Claim.** EvidLife-Map is a LiDAR-only online metric-semantic mapping system whose per-voxel posterior is a closed-form **Dirichlet-evidential** distribution; its single scalar **vacuity** mass is reused as (i) the open-set / OOD detector, (ii) the submap descriptor's entropy channel for confidence-aware loop closure, and (iii) the trigger for a **vacuity-conditioned conjugate voxel decay** that ages stale evidence across multi-session revisits. The same statistical quantity wires three modules — closing R2's [A1] under-specified Bayes filter (audit S1/S2/S5) and the three lifelong gaps (audit T1/T2/T3/T4) in one paper.

**Headline numbers (to be measured; targets from research plan §5 C3):** ≥ +2 mIoU and ≥ −20 % ECE over a faithful R2-reimpl on SemanticKITTI; ≤ 50 % of R2-reimpl's mean mIoU drop and ≥ 0.80 AUROC vacuity-as-corruption-detector on Robo3D-SemanticKITTI; ≥ 80 % stale-voxel removal precision on KITTI-360 multi-session revisits; ≥ 5 Hz on Jetson Orin NX (16 GB).

**Code, Docker, Robo3D corruption pipeline, and KITTI-360 revisit-pair builder** released.

**Citations in abstract:** none (IEEE convention).

---

## §I Introduction (~0.75 column, ~500 words)

### I.A Motivation (~0.35 col)
- *Claim:* Outdoor robot autonomy across weeks of repeated traversal needs a voxel map that is calibrated (knows what it doesn't know), open-set-aware (admits unknowns), and lifelong-maintainable (ages stale evidence) — three properties no current online MSM system delivers in one stack.
- *Cites:* R2 baseline [A1] (the under-specified Bayes filter / pessimistic unknown handling / no lifelong); Khronos [A2/D1] (orthogonal time factorisation); Clio [A3/D2] (task-driven open-set scene graph); ConvBKI [C9] (kernel Bayesian voxel competitor); KITTI-360 (multi-session arena).

### I.B Contributions (~0.4 col, bullets)
- **C1** — *Dirichlet-evidential per-voxel posterior whose vacuity scalar serves three jobs simultaneously*: open-set OOD score, submap descriptor entropy channel, lifelong decay trigger. Generalises R2's [A1] unspecified Bayes filter and contrasts with ConvBKI's [C9] kernel-Bayesian smoothing.
- **C2** — *Confidence-aware loop closure with parameter-free conjugate Dirichlet inter-session submap fusion* + *vacuity-conditioned voxel decay τ(m_u)*, closing audit T1/T2/T3/T4 in one system.
- **C3** — *Empirical evidence on 5 public datasets* (SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, Robo3D-SemanticKITTI, SemanticSpray) that the same calibrated representation wins on closed-set mIoU/ECE, corruption robustness, traversability safety, and multi-session lifelong precision.
- **C4** — *Reproducibility + deployability package*: code, Docker, ROS 2 launch, KITTI-360 revisit builder, Robo3D pipeline, Jetson Orin NX benchmark.

**Figure 1 (teaser):** R2-vs-EvidLife-Map on a KITTI-360 second-visit frame: left shows R2's stale-voxel hallucinations (deleted car still in map), right shows EvidLife-Map's vacuity-aged voxel re-write. Caption ~50 words.

---

## §II Related Work (~1.0 column, ~650 words)

### II.A Online Metric-Semantic Mapping (~0.35 col)
- *Claim:* The TSDF + per-voxel Bayes-filter lineage from Voxblox-class systems is the closest comparison set; recent extensions add 4D dynamic factorisation (Khronos) or scene-graph compression (Hydra family, Clio).
- *Cites:* Voxblox [C1], Voxblox++ [C3], PanopticFusion [C4], Voxfield [C5], NvBlox [C10], Hydra [A4], Hydra-Multi [A5], Kimera [C6], Kimera-Multi [C7], Khronos [A2], R2 [A1].

### II.B Probabilistic and Open-set Semantic Voxel Fusion (~0.35 col)
- *Claim:* S-BKI / ConvBKI / LatentBKI provide kernel-Bayesian spatial smoothing; OpenVox provides Bernoulli open-vocab. We are the first to use **EDL Dirichlet** at the voxel level *and* to reuse the resulting vacuity as a lifelong decay signal — the second move is what distinguishes v2 from a pure evidential-mapping method paper.
- *Cites:* S-BKI [C8], ConvBKI [C9], LatentBKI [D10], OpenVox [A8/D8], SLIM-VDB [A9], Clio [A3], HOV-SG [B1.6], ConceptFusion [B1.1], OpenScene [B1.3], Open-Fusion [A10].

### II.C Long-term Semantic Map Maintenance and Loop Closure (~0.3 col)
- *Claim:* Semantic-aided LCD (SA-LOAM, Semantic-Graph + GAT) and cross-session change detection (PlaneSDF, SLAM2REF) exist as separate components; we integrate confidence-aware LCD with conjugate Dirichlet inter-session fusion under the same per-voxel statistical model, so the loop-closure entropy channel and the decay rule both consume the same vacuity.
- *Cites:* SA-LOAM [B4.4], LiDAR-LCD-with-Semantic-GAT [B4.1], PlaneSDF [B4.2], SLAM2REF [B4.3], LTC-Mapping [A12], Kimera-Multi [C7].

**Citation budget for §II:** ~24 distinct refs.

---

## §III Method (~3 columns, ~1900 words; condensed for IROS 8 p)

### III.A System Overview (~0.4 col)
- *Claim:* Three-module pipeline (M1 evidential posterior, M2 loop closure + submap fusion, M3 vacuity-conditioned decay) sharing one per-voxel state `α_v ∈ R^{20}`.
- *Figure 2 (system overview):* Block diagram from research plan §3.1 (M1+M2+M3 with vacuity arrows showing the "one scalar, three jobs" wiring). Mandatory.
- *Cites:* [A1] (baseline architecture); [C10] (NvBlox SDF backbone); [B2.7] (GS-LIVO as photometric-Gaussian contrast, declared out of scope here).

### III.B M1 — Evidential Per-Voxel Posterior (~0.7 col)
- *Claim:* Eqs. M1.1-M1.3 (research plan §3.6). Posterior is closed-form; vacuity is the open-set / OOD score; recovers R2's argmax-Bayes as a special case (evidence weights unbounded, unknown channel pinned to zero); explicitly contrasts with S-BKI/ConvBKI kernel-Bayesian inference because we want **per-voxel** epistemic uncertainty uncontaminated by neighbour smoothing (needed by M3).
- *Cites:* [C8, C9, D10] (probabilistic voxel mapping context); [A1] (recovery as special case).
- *Table I:* closed-form expressions for posterior mean / variance / vacuity, comparing EDL-Dirichlet (ours) vs S-BKI vs ConvBKI vs R2 argmax-Bayes (5 rows × 4 cols).

### III.C M2 — Confidence-aware Loop Closure + Submap Fusion (~0.6 col)
- *Claim:* Eq. M2.1 submap descriptor `d(S) = (h_class(S), h_entropy(S))`; matching via cosine on h_class + EMD on h_entropy; verified by semantic-ICP on class-consistent confident voxels. Parameter-free conjugate Dirichlet inter-session fusion `α_v ← α_v^{s1} + α_v^{s2}` (independent observations under Dirichlet conjugacy).
- *Cites:* [B4.1] (semantic-graph + GAT precedent), [B4.4] (SA-LOAM), [A4] (Hydra per-submap descriptor), [C7] (Kimera-Multi pose graph context), [B4.2] (PlaneSDF cross-session inspiration).
- *Figure 3:* loop-closure descriptor schematic — h_class + h_entropy as two stacked bars, matched across two submaps from different sessions.

### III.D M3 — Vacuity-driven Voxel Decay (~0.6 col)
- *Claim:* Eq. M3.1 conjugate exponential decay with `τ(m_u) = τ_max(1 − m_u) + τ_min · m_u`. Confidently-known voxels age slowly (≈ 1 h), high-vacuity voxels age fast (≈ 1 min). This single equation is the v2 thesis (research plan §1.4) made concrete: the same vacuity scalar that flagged unknowns in M1 and weighted descriptor entropy in M2 now sets the staleness clock.
- *Cites:* [C5] (Voxfield non-projective SDF context), [C10] (NvBlox hash voxel), [A12] (LTC-Mapping long-term object handling), [B4.2] (PlaneSDF change detection alternative).
- *Figure 4:* time-lapse of a single voxel's `α_v` and vacuity trajectory across (i) static evidence (vacuity stays low, decay slow), (ii) static evidence then sudden disappearance (vacuity rises, decay accelerates), (iii) new evidence (vacuity drops, decay slows again). One frame per panel, 3 panels.

### III.E Implementation Details (~0.3 col, terse)
- LiDAR semantic head: **Cylinder3D** (pretrained, last layer fine-tuned with Sensoy 2018 EDL loss to 20-D output: 19 SemanticKITTI classes + 1 unknown channel).
- Backbone: **NvBlox** [C10] forked, `α_v` (20 × float32 = 80 B) replaces label-probability vector; per-voxel memory ~180 B (vs R2 ~100 B; documented in §IV).
- LVIO: FAST-LIO2 on KITTI-360, R3LIVE on SemanticKITTI single-session, public LIO config on SemanticSpray.
- **Hardware:** trained and evaluated on 1× RTX 4060 (16 GB); **deployment benchmarked on Jetson Orin NX (16 GB), target ≥ 5 Hz**. This is the key deployability claim in §IV.G.
- *Cites:* [C10] NvBlox; [B2.7] GS-LIVO Orin NX comparison line.

**Method-section citation budget:** ~10 refs (most reused from §II).

---

## §IV Experiments (~2 columns, ~1300 words)

### IV.A Datasets, Metrics, Protocol (~0.25 col)
- *Claim:* Five public datasets (SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, Robo3D-SemanticKITTI, SemanticSpray); all dense GT or deterministic-corruption labels; **zero manual annotation needed**.
- *Table II (datasets × RQ matrix):* 5 rows × 4 cols (one per RQ), check-mark which dataset answers which RQ.

### IV.B Baselines and Implementation (~0.2 col)
- *Claim:* Six baselines: R2-reimpl [A1], NvBlox-vanilla [C10], ConvBKI [C9], Kimera-Semantics [C6], Khronos [A2] (closed-set RQ1 only), Clio-LiDAR-stub [A3] (open-set RQ2 backup only). All hyperparameters, seeds, and CUDA versions in supplementary.

### IV.C RQ1 — Closed-set mIoU and Calibration (~0.35 col)
- *Claim (target):* EvidLife-Map achieves ≥ +2 mIoU over R2-reimpl and ≥ −20 % ECE over ConvBKI on SemanticKITTI seq 08, 11-21; cross-domain check on nuScenes-LiDARSeg confirms the gap survives a taxonomy shift.
- *Table III (main result):* method × {mIoU, mAcc, ECE, Brier, AUROC-unknown, latency-ms, GPU-MB}; 6 rows × 7 cols.

### IV.D RQ2 — Corruption Robustness on Robo3D (~0.4 col)
- *Claim (target):* ≤ 50 % of R2-reimpl's mean mIoU drop across 8 Robo3D corruption types at severity 3; vacuity-as-corruption-detector AUROC ≥ 0.80 averaged across corruptions.
- *Cites:* Robo3D (Kong 2023, ICCV-23; **note: not in lit_scan.md** — must add Robo3D as a new entry in the final BibTeX or cite via the SemanticKITTI baseline tradition; flagged for citation-check pass).
- *Figure 5:* per-corruption mIoU-drop bar chart, EvidLife-Map vs R2-reimpl vs ConvBKI; 8 corruption types × 3 systems.
- *Table IV (RQ2):* per-corruption mIoU + vacuity-AUROC; 8 rows × 4 cols.

### IV.E RQ3 — Uncertainty-aware Traversability on SemanticSpray (~0.25 col)
- *Claim (target):* ≥ +10 pp safe-region recall and ≥ −30 % false-traversable rate vs R2-style hard-rule baseline on SemanticSpray passive replay.
- *Cites:* [B3.9].
- *Table V (RQ3):* method × {safe-region recall, false-traversable rate, deferral rate, coverage}; 3 rows × 4 cols.

### IV.F RQ4 — Lifelong Multi-Session on KITTI-360 (~0.4 col)
- *Claim (target):* ≥ 80 % stale-voxel removal precision over 5+ revisit pairs; ECE drift across sessions ≤ 1.5× single-session ECE; map size growth ≤ 1.7× across 5 sessions (sub-linear thanks to voxel-hash deduplication).
- *Figure 6:* per-revisit-pair stale-voxel precision-recall curves; one curve per pair, overlaid.
- *Table VI (RQ4):* revisit-pair × {stale-voxel P, R, F1; ECE; map size growth}; 5 rows × 5 cols.

### IV.G Ablations, Runtime, Jetson Deployment (~0.3 col)
- *Claim:* A-1 (Dirichlet ±), A-2 (vacuity-conditioned τ ±), A-3 (entropy descriptor channel ±), A-4 (uncertainty-aware traversability ±), A-5 (conjugate inter-session fusion ±) each individually positive on ≥ 1 metric. **Jetson Orin NX** runs at ≥ 5 Hz on KITTI-360 streaming replay (target; to be measured).
- *Cites:* [B2.7] GS-LIVO Jetson contrast.
- *Table VII (ablations + runtime):* variable × {SemanticKITTI mIoU, KITTI-360 stale precision, Orin NX Hz}; 5 rows × 3 cols.
- *Figure 7 (optional, drop if overflow):* latency CDF on 4060 vs Orin NX.

**Experiments-section citation budget:** ~5 new refs beyond §II/III.

---

## §V Discussion (~0.5 column, ~330 words)

- **Limitations.** (i) LiDAR-only — no claim on RGB / radar weather robustness; the v1 multi-modal story is explicitly out of scope. (ii) Robo3D is synthetic corruption; real adverse-weather behaviour not characterised (we did not use K-Radar / Boreas / CADC). (iii) KITTI-360 multi-session pairs are odometry-overlap constructed (5+ pairs); not the same as long-duration months-apart revisits Boreas would offer. (iv) Lifelong tested single-robot only — Hydra-Multi-class collaborative scenarios out of scope. (v) Jetson Orin NX benchmark is replay-based; no real-robot deployment.
- **Failure modes.** (a) When confidently-wrong evidence accumulates (e.g., persistent mirror surface ghost returns), vacuity stays artificially low and M3 does not trigger decay — a known evidential-deep-learning failure inherited from Sensoy 2018. (b) When two sessions have non-overlapping classes (e.g., one captured before construction, one after, no shared landmarks), M2's class-histogram descriptor falls back to entropy-histogram only and precision degrades. (c) On extremely sparse Robo3D severity-5 corruptions, the EDL head can collapse to high-vacuity-everywhere, which is honest but unhelpful — vacuity rises but ranking power AUROC drops.
- **Honest comparisons we did not win.** State explicitly any condition where ConvBKI / Khronos / Kimera-Semantics beat us; the v2 thesis is *integration* across three jobs, not point-wise SOTA on any single one. (Reviewer-credibility move; closes audit E6.)
- *Cites:* [A2] (Khronos orthogonal future combination — time × uncertainty); [B2.7] (GS-LIVO photometric complement).

---

## §VI Conclusion (~0.25 column, ~170 words)

One paragraph: recap of C1-C4. Single forward-looking sentence: *"Combining the per-voxel uncertainty axis of EvidLife-Map with the spatio-temporal short-/long-term factorisation of Khronos [A2] is a promising next step toward a fully integrated calibrated 4D MSM."* No new claims, no new cites.

---

## References (~1 column, ≤ 30 refs)

Drawn exclusively from `lit_scan.md` **plus one externally-added Robo3D citation** (flagged for citation-check pass; not in lit_scan but unavoidable for RQ2). Mandatory inclusions (audit + lit-scan reviewer-threat list):

- **R2 baseline** — [A1].
- **Top-3 reviewer-threat must-cites** — Khronos [A2/D1], GS-LIVO [B2.7/D6], Clio [A3/D2].
- **Section-C foundations** — Voxblox [C1], Voxblox++ [C3], PanopticFusion [C4], Voxfield [C5], NvBlox [C10], Kimera [C6], Kimera-Multi [C7], ConvBKI [C9], S-BKI [C8].
- **Probabilistic / open-vocab competitors** — OpenVox [A8/D8], LatentBKI [D10], Open-Fusion [A10], SLIM-VDB [A9].
- **Open-vocab scene-graph context** — HOV-SG [B1.6], ConceptFusion [B1.1], OpenScene [B1.3].
- **Loop closure / long-term maintenance** — SA-LOAM [B4.4], LiDAR-LCD-with-Semantic-GAT [B4.1], PlaneSDF [B4.2], SLAM2REF [B4.3], LTC-Mapping [A12].
- **Datasets** — SemanticSpray [B3.9]; SemanticKITTI / nuScenes-LiDARSeg / KITTI-360 (standard handles, brought in via BibTeX); **Robo3D (Kong et al., ICCV-23 — must be added as new entry, not in lit_scan.md)**.
- **Hydra family for completeness** — Hydra [A4], Hydra-Multi [A5].

That is 28 cited entries from lit_scan + 4 dataset-handle entries (SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, Robo3D) = 32 references. Trim 2 at submission (likely drop A5 Hydra-Multi and B4.3 SLAM2REF), keeping threat-list and dataset cites intact.

---

## Figure / Table Budget Summary

| Asset | Section | Purpose |
|-------|---------|---------|
| Fig. 1 | §I teaser | R2 vs EvidLife-Map on KITTI-360 second visit (stale-voxel handling) |
| Fig. 2 | §III.A | System diagram (M1+M2+M3 with vacuity arrows) |
| Fig. 3 | §III.C | Loop-closure descriptor (h_class + h_entropy) schematic |
| Fig. 4 | §III.D | Single-voxel `α_v` + vacuity time-lapse, 3 panels |
| Fig. 5 | §IV.D | Per-corruption mIoU-drop bars (Robo3D) |
| Fig. 6 | §IV.F | Per-revisit-pair stale-voxel PR curves |
| Fig. 7 (optional) | §IV.G | Latency CDF, 4060 vs Orin NX |
| Tab. I | §III.B | Closed-form posterior comparison |
| Tab. II | §IV.A | Datasets × RQ matrix |
| Tab. III | §IV.C | Main results RQ1 (mIoU / ECE / Brier / AUROC / latency / mem) |
| Tab. IV | §IV.D | Per-corruption mIoU + vacuity-AUROC (RQ2) |
| Tab. V | §IV.E | Traversability RQ3 |
| Tab. VI | §IV.F | Multi-session lifelong RQ4 |
| Tab. VII | §IV.G | Ablations + Jetson runtime |

Total: 6 figures + 7 tables (with Fig. 7 optional, droppable to 6+7 if §IV.G overflows). Conservative for an 8-page IROS.

---

## Word / Column Budget Check (8-page IROS estimate)

| Section | Target words | Target columns |
|---------|--------------|----------------|
| Abstract | ~200 | 1 col |
| §I Intro | ~500 | 0.75 |
| §II Related | ~650 | 1.0 |
| §III Method | ~1900 | 3.0 |
| §IV Experiments | ~1300 | 2.0 |
| §V Discussion | ~330 | 0.5 |
| §VI Conclusion | ~170 | 0.25 |
| References (~30 refs × ~25 words/ref) | ~750 | 1.0 |
| **Sub-total text** | **~5800** | **~9.5 cols** |
| Figures + tables (estimated visual area) | — | ~6.5 cols equivalent |
| **Total** | — | **~16 cols ≈ 8 pages double-column** |

Headroom: tight but feasible; Fig. 7 and Table VII rows are the first to cut if overflow.

---

*End of paper outline v2. Aligned with `research_plan.md` (v2). v1 outline frozen at `paper_outline_v1.md`. Advance to `/ars-full` only when all 5 §9 Go criteria of the v2 plan are met.*
