# Paper Outline v1

**Title:** *EviRad-Map: Evidential LiDAR + 4D-Radar Online Metric-Semantic Mapping for Weather-Robust Robot Navigation*

**Target:** IEEE RA-L (with IROS / ICRA option), 8 pages double-column IEEE conference format, ≤ 30 references.
**Companion:** `research_plan.md` (this directory).
**Conventions used in this outline:** word counts are upper bounds per IEEE RA-L double-column (≈ 700-750 words / column). Citation tokens like `[A1]`, `[B3.6]`, `[C9]` map to the entries of `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md`; only those are allowed.

---

## Abstract (1 column, ~200 words)

**Claim.** A LiDAR + 4D-radar online metric-semantic mapping system, *EviRad-Map*, that replaces the unspecified per-voxel Bayes filter common to R2-class systems with a closed-form Dirichlet-evidential update, fuses radar and LiDAR evidence through a weather-aware trust schedule, and routes the resulting per-voxel vacuity (uncertainty mass) into a deferring traversability head.
**Headline numbers (to be measured, targets from §C3 of research plan):** +3 mIoU and −20 % ECE over a faithful R2 reimplementation on SemanticKITTI, ≥ 80 % of clear-weather mIoU retained on K-Radar fog/rain/snow (LiDAR-only baseline retains ≤ 60 %), +15 pp closed-loop navigation success rate on SemanticSpray-style wet road, with end-to-end latency comparable to NvBlox on Jetson Orin NX.
**Code & data release** stated explicitly.
**Citations in abstract:** none (per IEEE convention).

---

## §I Introduction (~0.75 column, ~525 words)

### I.A Motivation (~0.4 col)
- *Claim:* Outdoor robot autonomy under adverse conditions demands a metric-semantic map whose semantic uncertainty is honest at the voxel level, and whose modality stack includes a weather-robust radar branch — neither of which today's online MSM systems provide.
- *Cites:* R2 baseline [A1]; Khronos as 4D-MSM leader [A2/D1]; GS-LIVO as HKUST sibling Gaussian-Splat alternative [B2.7/D6]; Clio as task-driven open-set [A3/D2]; weather-robust 3D perception trend [B3.1, B3.6].

### I.B Contributions (~0.35 col, bullets)
- C1 — *First open-source online MSM to fuse LiDAR + 4D-radar at the voxel level*, motivated by the absence of such a system in a 2022-2026 lit scan (lit_scan §Search Failures bullet 3). *Cite:* [A1, A2, B3.6, B5.1].
- C2 — *Dirichlet-evidential per-voxel posterior with explicit vacuity*, generalising the under-specified Bayes filter of R2 [A1] and the kernel-Bayes formulation of ConvBKI [C9] / LatentBKI [D10].
- C3 — *Modality-aware trust schedule + conjugate evidence decay*, enabling weather-robust fusion and lifelong map maintenance with closed-form update equations.
- C4 — *Reproducibility package* (code + Docker + ROS 2 launch + K-Radar pseudo-GT pipeline), directly addressing the reproducibility gap of the R2 baseline.

**Figure 1 (teaser):** Side-by-side R2-vs-EviRad-Map on a K-Radar fog scene; left shows R2's hallucinated road semantics, right shows EviRad-Map's vacuity-shaded reliable region. Caption ~50 words.

---

## §II Related Work (~1.0 column, ~700 words)

### II.A Online Metric-Semantic Mapping (~0.4 col)
- *Claim:* The TSDF + per-voxel Bayes-filter lineage from Voxblox-class systems is the closest comparison set; recent extensions add 4D dynamic factorisation or scene-graph compression.
- *Cites:* Voxblox [C1], Voxblox++ [C3], PanopticFusion [C4], Voxfield [C5], NvBlox [C10], Hydra [A4], Hydra-Multi [A5], Kimera [C6], Kimera-Multi [C7], Khronos [A2], R2 [A1].
- *Figure / table:* none.

### II.B Probabilistic and Open-set Semantic Voxel Fusion (~0.3 col)
- *Claim:* S-BKI / ConvBKI / LatentBKI provide kernel-Bayesian uncertainty; OpenVox provides Bernoulli open-vocab. We are the first to use *Dirichlet evidential* fusion at the voxel level, which gives closed-form vacuity without spatial smoothing assumptions.
- *Cites:* S-BKI [C8], ConvBKI [C9], LatentBKI [D10], OpenVox [A8/D8], SLIM-VDB [A9], Clio [A3], HOV-SG [B1.6], ConceptFusion [B1.1], ConceptGraphs [B1.2], OpenScene [B1.3], OpenMask3D [B1.4], Open-Fusion [A10].

### II.C Multi-modal Robust Perception under Adverse Weather (~0.3 col)
- *Claim:* The detection literature has converged on LiDAR + 4D-radar fusion for weather robustness; semantic *mapping* has not yet been demonstrated. Adverse-condition image segmentation and point-cloud denoising are mature components we can build on, not duplicate.
- *Cites:* K-Radar [B3.1], L4DR [B3.6], RaSS [B5.2], SegNet4D [B5.3], BEVFusion [B5.4], TripleMixer [B3.7], ACDC [B3.3], WeatherProof [B3.8], SemanticSpray [B3.9], Boreas [B3.4], CADC [B3.5], ZOD [B3.2].

**Citation budget for §II:** ~30 distinct refs; this is the bulk of the bibliography.

---

## §III Method (~3 columns, ~2100 words)

### III.A System Overview (~0.4 col)
- *Claim:* Four-module pipeline (cf. R2's four-module structure but with cross-modal evidence flow and uncertainty back-channel).
- *Figure 2 (system overview):* Block diagram from research_plan §3.1; mandatory.
- *Cites:* [A1] for baseline architecture; [C10] for NvBlox SDF backbone choice; [B2.7] to contrast with photometric-Gaussian alternative.

### III.B Module 1 — Evidential Per-Voxel Posterior (~0.7 col)
- *Claim:* Eq. 1-2 (research plan §3.6). Posterior is closed form; vacuity is the open-set / OOD score; recovers R2's argmax Bayes as a special case when evidence weights are unbounded and the unknown channel is fixed to zero.
- *Cites:* [C8, C9, D10] (positional within probabilistic voxel mapping); [A1] (recovery as special case).
- *Table I*: closed-form expressions for posterior mean, variance, vacuity, comparing EDL-Dirichlet (ours) vs S-BKI vs ConvBKI vs R2 argmax-Bayes. (1 col-wide, 5 rows × 4 cols.)

### III.C Module 2 — Modality-Aware Trust Allocation (~0.6 col)
- *Claim:* Eq. 3. Trust weights `w(modality, weather)` are learned on K-Radar normal vs adverse splits and modulate evidence (not posterior labels), so under heavy fog the LiDAR branch's evidence decays smoothly to zero while vacuity rises — preventing the failure mode where a fragile modality's wrong label dominates a robust modality's right label.
- *Cites:* [B3.1, B3.6, B5.2, B3.7] for weather signal computation; [B5.4] for general multi-modal fusion baseline.
- *Figure 3:* trust-schedule curves vs weather indicator (LiDAR fog tolerance vs radar SNR floor).

### III.D Module 3 — Adaptive Multi-Resolution Hash Voxel with Conjugate Decay (~0.6 col)
- *Claim:* Two-level hash voxel (0.05 m near / 0.25 m far) with per-voxel Dirichlet evidence; Eq. 4 conjugate decay preserves posterior mean while inflating vacuity, enabling lifelong stale-voxel rewriting.
- *Cites:* [C5] (non-projective SDF), [C10] (hash voxel), [A6] (multi-resolution panoptic TSDF), [B4.2] (PlaneSDF change detection inspiration), [A1] (R2 0.25 m comparison).
- *Figure 4:* time-lapse of stale-voxel vacuity rising and being overwritten.

### III.E Module 4 — Uncertainty-Aware Traversability and Loop Closure (~0.5 col)
- *Claim:* Eq. 5. Traversability head consumes class mean *and* vacuity, with three-way output (drivable / non-drivable / defer-and-replan), addressing R2's audit S5 over-conservative "unknown = untraversable" failure. Loop closure: per-submap class-histogram + entropy-histogram descriptor, verified by semantic-ICP on confident voxels.
- *Cites:* [A1] (R2 hard rule); [B4.1] (semantic loop closure precedent); [B4.4] (SA-LOAM); [A4, C7] (pose graph optimisation context).

### III.F Implementation Details (~0.2 col, deliberately terse)
- LiDAR semantic head: Cylinder3D, pretrained on SemanticKITTI.
- Radar semantic head: trained via cross-modal distillation from LiDAR head (RaSS-style [B5.2]).
- Backbone: NvBlox [C10] forked, with our Dirichlet evidence vector replacing the label probability vector.
- Runs at 10-15 Hz on RTX 4090; ~7 Hz on Jetson Orin NX (target; to be measured).

**Method-section citation budget:** ~12 refs (most already cited in §II).

---

## §IV Experiments (~2 columns, ~1400 words)

### IV.A Datasets, Metrics, Protocol (~0.3 col)
- *Claim:* Three public datasets (SemanticKITTI, nuScenes-LiDARSeg, K-Radar) + one closed-loop sim (SemanticSpray + Isaac Sim). K-Radar pseudo-GT protocol declared transparently (research plan §4.3).
- *Cites:* [B3.1, B3.9]; SemanticKITTI / nuScenes-LiDARSeg via standard handles (not yet in lit_scan; bring in only if word-budget allows).
- *Table II (data + metrics):* dataset × tier-of-metric matrix.

### IV.B Baselines and Implementation (~0.2 col)
- *Claim:* Five baselines: R2-reimpl. [A1], NvBlox-vanilla [C10], ConvBKI [C9], OpenVox [A8/D8], Khronos [A2] (RQ1-restricted comparison). Stretch: L4DR-voted [B3.6].
- *Cites:* as above.

### IV.C Main Results — RQ1 closed-set mIoU and ECE (~0.4 col)
- *Claim (target):* EviRad-Map (LiDAR-only configuration for fair comparison) achieves ≥ +3 mIoU over R2-reimpl. and ≥ −20 % ECE over ConvBKI on SemanticKITTI seq 08, 11-21.
- *Table III (main result):* method × {mIoU, mAcc, ECE, Brier, AUROC-unknown, latency-ms, GPU-MB}; 6 rows × 7 cols.

### IV.D Robustness — RQ2 LiDAR + 4D-Radar under fog/rain/snow (~0.4 col)
- *Claim (target):* EviRad-Map retains ≥ 80 % of clear-weather mIoU on K-Radar adverse splits; LiDAR-only baseline retains ≤ 60 %.
- *Cites:* [B3.1] (K-Radar), [B3.6] (L4DR comparison).
- *Figure 5:* mIoU-vs-weather-severity curves, ours vs LiDAR-only vs L4DR-voted.
- *Table IV (radar contribution):* Δ-mIoU per condition per modality combination.

### IV.E Downstream Navigation — RQ3 closed-loop on SemanticSpray + Isaac Sim (~0.2 col)
- *Claim (target):* +15 pp navigation success rate and −50 % collision rate over R2-reimpl., at equal path length.
- *Cites:* [B3.9].
- *Table V (nav):* method × {success, collision, path-length ratio, deferral rate}; 3 rows.

### IV.F Ablations (~0.3 col)
- *Claim:* Each of A-1…A-6 (research plan §4.4) is individually positive on at least one metric.
- *Table VI (ablation):* variable × {SemanticKITTI mIoU, K-Radar adverse mIoU, nav success}; 6-7 rows × 3 cols.

### IV.G Runtime, Memory, Lifelong (~0.2 col)
- *Claim:* end-to-end latency mean + 99-pct on RTX 4090 and Jetson Orin NX; per-voxel memory vs ConvBKI / R2; on Boreas / KITTI-360 revisit, ≥ 80 % stale-voxel removal precision.
- *Cites:* [B3.4] Boreas; [B2.7] GS-LIVO Orin claim contrast.
- *Figure 6:* latency CDF; memory-vs-voxel-count curve.

**Experiments-section citation budget:** ~6 new refs beyond §II/III.

---

## §V Discussion (~0.5 column, ~350 words)

- **Limitations.** (i) Pseudo-GT on K-Radar is the principal evaluation risk; we mitigate via manual sparse-GT sanity check but cannot fully eliminate. (ii) No real-robot deployment; closed-loop only in Isaac Sim. (iii) Loop closure tested only on single-session revisits (not multi-robot). (iv) Radar branch is currently trained per-vendor (RETINA 4D); cross-vendor transfer untested.
- **Failure modes.** (a) When both modalities agree on a wrong label (e.g. mirror surfaces causing both LiDAR ghosting and radar mis-detection), vacuity stays low — evidential fusion cannot cure this. (b) Extremely sparse radar returns (small objects > 30 m) collapse the radar evidence to vacuity, recovering LiDAR-only behaviour.
- **Honest comparisons we did not win.** State explicitly any benchmark/condition where a baseline beat us; reviewer-credibility move (cf. R2 audit E6).
- *Cites:* [A2] (Khronos as orthogonal future combination), [B2.7] (GS-LIVO as photometric complement).

---

## §VI Conclusion (~0.25 column, ~175 words)

One-paragraph reprise of contributions; one-sentence forward look ("combining the modality + uncertainty axis here with the spatio-temporal axis of Khronos [A2] is a promising next step"). No new claims, no new cites.

---

## References (~1 column, ≤ 30 refs)

Drawn exclusively from `lit_scan.md`. Mandatory inclusions (audit + lit-scan reviewer-threat list):

- **R2 baseline & journal extension** — [A1].
- **Top-3 reviewer-threat must-cites** — Khronos [A2/D1], GS-LIVO [B2.7/D6], Clio [A3/D2].
- **Section-C foundations** — Voxblox [C1], Voxblox++ [C3], PanopticFusion [C4], Voxfield [C5], NvBlox [C10], Kimera [C6], Kimera-Multi [C7], ConvBKI [C9], S-BKI [C8].
- **Probabilistic / open-vocab competitors** — OpenVox [A8/D8], LatentBKI [D10], Open-Fusion [A10], SLIM-VDB [A9].
- **Open-vocab / scene-graph context** — HOV-SG [B1.6], ConceptFusion [B1.1], ConceptGraphs [B1.2], OpenScene [B1.3], OpenMask3D [B1.4].
- **Weather + radar** — K-Radar [B3.1], L4DR [B3.6], RaSS [B5.2], SegNet4D [B5.3], BEVFusion [B5.4], TripleMixer [B3.7], ACDC [B3.3], SemanticSpray [B3.9], Boreas [B3.4], CADC [B3.5].
- **Hydra family for scene-graph completeness** — Hydra [A4], Hydra-Multi [A5].

That is 31 entries; trim 1-2 at submission (likely drop A5 Hydra-Multi and B3.5 CADC if word budget bites, keeping the threat-list mandatory cites intact).

---

## Figure / Table Budget Summary

| Asset | Section | Purpose |
|-------|---------|---------|
| Fig. 1 | §I teaser | R2 vs ours on K-Radar fog |
| Fig. 2 | §III.A | System diagram |
| Fig. 3 | §III.C | Trust schedule curves |
| Fig. 4 | §III.D | Stale-voxel vacuity time-lapse |
| Fig. 5 | §IV.D | mIoU-vs-weather curve |
| Fig. 6 | §IV.G | Latency CDF + memory curve |
| Tab. I | §III.B | Closed-form posterior comparison |
| Tab. II | §IV.A | Datasets × metrics matrix |
| Tab. III | §IV.C | Main results (RQ1) |
| Tab. IV | §IV.D | Radar contribution per condition (RQ2) |
| Tab. V | §IV.E | Navigation closed-loop (RQ3) |
| Tab. VI | §IV.F | Ablations |

Total: 6 figures + 6 tables. Conservative for 8 pages; one figure could be moved to supplementary if §IV.G overflows.

---

*End of paper outline v1. Aligned with `research_plan.md`. Both files are the deliverables of `academic-pipeline` Stage 1; advance to `/ars-full` only when all 5 Go criteria of research_plan §9 are met.*
