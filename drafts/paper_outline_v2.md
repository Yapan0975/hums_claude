---
version: v3 (2026-05-28) — patches: RQ2→open-set, +Khronos dynamic comparison, +Jetson demo video
v2 archived as: paper_outline_v2_archived.md (same directory; not modified after v3 patch)
v1 archived as: paper_outline_v1.md (same directory)
v3 companion plan: research_plan.md (v3; v2 frozen at research_plan_v2.md)
filename note: per user spec, v3 contents may live in paper_outline_v2.md (filename retained) — v2 contents preserved at the *_archived sibling
---

# Paper Outline v3

**Title:** *EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay and Open-Set Vacuity*

**Target:** IROS 2027, 8 pages double-column IEEE conference format, ≤ 30 references.
**Companion:** `research_plan.md` (v3; this directory).
**Conventions:** word counts are upper bounds per IROS double-column (≈ 600-700 words / column on the IROS template). Citation tokens like `[A1]`, `[C9]`, `[B4.1]` map to the entries of `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md`; only those are allowed (plus the externally-added dataset handles flagged in §References).

---

## §0.6 v2 → v3 outline change log (new in v3)

| Section | v2 | v3 |
|---|---|---|
| Abstract | Robo3D corruption AUROC headline | Open-set AUROC + AUPR headline; Robo3D demoted to "fallback evaluation also reported" |
| §I Motivation | "robust under corruption" | "open-set / OOD-aware in mapping" |
| §III Method | M1/M2/M3 only | adds §III.F **Dynamic-Object Handling Discussion** (sets up Khronos head-to-head) |
| §IV Experiments | RQ2 = Robo3D | RQ2 = open-set 14/5 + nuScenes reverse-check; **RQ5 (new) = Khronos dynamic head-to-head Table V**; **§IV.H (new) = Embedded Demo (supplementary video)** |
| §IV.G | Ablations + Jetson table | Ablations + Jetson table + cross-ref to §IV.H demo |
| Contributions | C4 deploy "optional" | C4 deploy **firm + demo video** |
| Figures/Tables | 6 fig + 7 tab | 6 fig + 8 tab (Table V is the new Khronos dynamic head-to-head) |
| References | 28 + 4 dataset handles = 32 | unchanged count (open-set protocol cites already-listed Sensoy 2018 via Cylinder3D-context implicit) |

---

## Abstract (1 column, ~200 words)

**Claim.** EvidLife-Map is a LiDAR-only online metric-semantic mapping system whose per-voxel posterior is a closed-form **Dirichlet-evidential** distribution; its single scalar **vacuity** mass is reused as (i) **the open-set / OOD score that flags voxels of unknown categories**, (ii) the submap descriptor's entropy channel for confidence-aware loop closure, and (iii) the trigger for a **vacuity-conditioned conjugate voxel decay** that ages stale evidence across multi-session revisits. The same statistical quantity wires three modules — closing R2's [A1] under-specified Bayes filter (audit S1/S2/S5) and the three lifelong gaps (audit T1/T2/T3/T4) in one paper.

**Headline numbers (to be measured; targets from research plan §5 C3):** ≥ +2 mIoU and ≥ −20 % ECE over a faithful R2-reimpl on SemanticKITTI; **open-set vacuity AUROC ≥ 0.80 / AUPR ≥ 0.60 on the SemanticKITTI 14-known/5-unknown split (closed-set mIoU drop ≤ 1.5)**; **dynamic-object mIoU within 3 of Khronos** [A2] at ≥ 5 Hz on Jetson Orin NX; ≥ 80 % stale-voxel removal precision on KITTI-360 multi-session revisits.

**Code, Docker, open-set split definition, KITTI-360 revisit builder, and a 30-second Jetson Orin NX online-mapping demo video released.**

**Citations in abstract:** none (IEEE convention; named systems cite-suppressed).

---

## §I Introduction (~0.75 column, ~500 words)

### I.A Motivation (~0.35 col)
- *Claim:* Outdoor robot autonomy across weeks of repeated traversal needs a voxel map that is calibrated (knows what it doesn't know), **open-set / OOD-aware in mapping** (admits voxels belonging to categories outside the training taxonomy), and lifelong-maintainable (ages stale evidence) — three properties no current online MSM system delivers in one stack.
- *Cites:* R2 baseline [A1] (the under-specified Bayes filter / pessimistic unknown handling / no lifelong); Khronos [A2/D1] (orthogonal time factorisation, **now experimentally compared on dynamic frames in v3 §IV.G**); Clio [A3/D2] (task-driven open-set scene graph); ConvBKI [C9] (kernel Bayesian voxel competitor); KITTI-360 (multi-session arena).

### I.B Contributions (~0.4 col, bullets)
- **C1** — *Dirichlet-evidential per-voxel posterior whose vacuity scalar serves three jobs simultaneously*: **open-set / OOD score for unknown-category voxels**, submap descriptor entropy channel, lifelong decay trigger. Generalises R2's [A1] unspecified Bayes filter and contrasts with ConvBKI's [C9] kernel-Bayesian smoothing.
- **C2** — *Confidence-aware loop closure with parameter-free conjugate Dirichlet inter-session submap fusion* + *vacuity-conditioned voxel decay τ(m_u)* + **dynamic-scene competitiveness against Khronos [A2] without an explicit 4D model**, closing audit T1/T2/T3/T4 in one system.
- **C3** — *Empirical evidence on 5 public datasets* (SemanticKITTI including the 14/5 open-set split and the dynamic split, nuScenes-LiDARSeg, KITTI-360, SemanticSpray) that the same calibrated representation wins on closed-set mIoU/ECE, open-set OOD detection, dynamic-scene competitiveness, traversability safety, and multi-session lifelong precision.
- **C4** — *Reproducibility + firm deployability package*: code, Docker, ROS 2 launch, open-set split files, KITTI-360 revisit builder, Jetson Orin NX benchmark, **and a 30-second supplementary demo video of online mapping on the Orin NX**.

**Figure 1 (teaser):** R2-vs-EvidLife-Map on a KITTI-360 second-visit frame: left shows R2's stale-voxel hallucinations (deleted car still in map), right shows EvidLife-Map's vacuity-aged voxel re-write — with a small inset showing a known/unknown vacuity overlay on an open-set frame from the SemanticKITTI 14/5 split. Caption ~60 words.

---

## §II Related Work (~1.0 column, ~650 words)

### II.A Online Metric-Semantic Mapping (~0.3 col)
- *Claim:* The TSDF + per-voxel Bayes-filter lineage from Voxblox-class systems is the closest comparison set; recent extensions add 4D dynamic factorisation (Khronos) or scene-graph compression (Hydra family, Clio).
- *Cites:* Voxblox [C1], Voxblox++ [C3], PanopticFusion [C4], Voxfield [C5], NvBlox [C10], Hydra [A4], Hydra-Multi [A5], Kimera [C6], Kimera-Multi [C7], Khronos [A2], R2 [A1].

### II.B Probabilistic and Open-set Semantic Voxel Fusion (~0.4 col)
- *Claim:* S-BKI / ConvBKI / LatentBKI provide kernel-Bayesian spatial smoothing; OpenVox provides Bernoulli open-vocab. We are the first to use **EDL Dirichlet** at the voxel level *and* to reuse the resulting vacuity as **(i) an open-set / OOD score** and (ii) a lifelong decay signal — the joint use is what distinguishes v3 from a pure evidential-mapping method paper.
- *Cites:* S-BKI [C8], ConvBKI [C9], LatentBKI [D10], OpenVox [A8/D8], SLIM-VDB [A9], Clio [A3], HOV-SG [B1.6], ConceptFusion [B1.1], OpenScene [B1.3], Open-Fusion [A10].

### II.C Long-term Semantic Map Maintenance and Loop Closure (~0.3 col)
- *Claim:* Semantic-aided LCD (SA-LOAM, Semantic-Graph + GAT) and cross-session change detection (PlaneSDF, SLAM2REF) exist as separate components; we integrate confidence-aware LCD with conjugate Dirichlet inter-session fusion under the same per-voxel statistical model, so the loop-closure entropy channel and the decay rule both consume the same vacuity. **Dynamic-object handling is touched lightly in §III.F where we set up the head-to-head against Khronos.**
- *Cites:* SA-LOAM [B4.4], LiDAR-LCD-with-Semantic-GAT [B4.1], PlaneSDF [B4.2], SLAM2REF [B4.3], LTC-Mapping [A12], Kimera-Multi [C7].

**Citation budget for §II:** ~24 distinct refs (unchanged from v2).

---

## §III Method (~3 columns, ~1900 words; condensed for IROS 8 p)

### III.A System Overview (~0.4 col)
- *Claim:* Three-module pipeline (M1 evidential posterior, M2 loop closure + submap fusion, M3 vacuity-conditioned decay) sharing one per-voxel state `α_v ∈ R^{C+1}` — where C is 19 (closed-set RQ1/RQ4/RQ5) or 14 (open-set RQ2).
- *Figure 2 (system overview):* Block diagram from research plan §3.1 (M1+M2+M3 with vacuity arrows showing the "one scalar, three jobs" wiring; **the (i) job arrow now labelled "open-set / OOD score" in v3**). Mandatory.
- *Cites:* [A1] (baseline architecture); [C10] (NvBlox SDF backbone); [B2.7] (GS-LIVO as photometric-Gaussian contrast, declared out of scope here).

### III.B M1 — Evidential Per-Voxel Posterior (~0.7 col)
- *Claim:* Eqs. M1.1-M1.3 (research plan §3.6). Posterior is closed-form; **vacuity acts as natural OOD score** (M1.3 directly consumed by §IV.D RQ2); recovers R2's argmax-Bayes as a special case (evidence weights unbounded, unknown channel pinned to zero); explicitly contrasts with S-BKI/ConvBKI kernel-Bayesian inference because we want **per-voxel** epistemic uncertainty uncontaminated by neighbour smoothing (needed by M3 and by RQ2 OOD).
- *Cites:* [C8, C9, D10] (probabilistic voxel mapping context); [A1] (recovery as special case).
- *Table I:* closed-form expressions for posterior mean / variance / vacuity, comparing EDL-Dirichlet (ours) vs S-BKI vs ConvBKI vs R2 argmax-Bayes (5 rows × 4 cols).

### III.C M2 — Confidence-aware Loop Closure + Submap Fusion (~0.6 col)
- *Claim:* Eq. M2.1 submap descriptor `d(S) = (h_class(S), h_entropy(S))`; matching via cosine on h_class + EMD on h_entropy; verified by semantic-ICP on class-consistent confident voxels. Parameter-free conjugate Dirichlet inter-session fusion `α_v ← α_v^{s1} + α_v^{s2}` (independent observations under Dirichlet conjugacy).
- *Cites:* [B4.1] (semantic-graph + GAT precedent), [B4.4] (SA-LOAM), [A4] (Hydra per-submap descriptor), [C7] (Kimera-Multi pose graph context), [B4.2] (PlaneSDF cross-session inspiration).
- *Figure 3:* loop-closure descriptor schematic — h_class + h_entropy as two stacked bars, matched across two submaps from different sessions.

### III.D M3 — Vacuity-driven Voxel Decay (~0.5 col)
- *Claim:* Eq. M3.1 conjugate exponential decay with `τ(m_u) = τ_max(1 − m_u) + τ_min · m_u`. Confidently-known voxels age slowly (≈ 1 h), high-vacuity voxels age fast (≈ 1 min). This single equation is the v3 thesis (research plan §1.4) made concrete: the same vacuity scalar that flagged **unknown-category voxels** in M1 (RQ2 lead) and weighted descriptor entropy in M2 now sets the staleness clock.
- *Cites:* [C5] (Voxfield non-projective SDF context), [C10] (NvBlox hash voxel), [A12] (LTC-Mapping long-term object handling), [B4.2] (PlaneSDF change detection alternative).
- *Figure 4:* time-lapse of a single voxel's `α_v` and vacuity trajectory across (i) static evidence (vacuity stays low, decay slow), (ii) static evidence then sudden disappearance (vacuity rises, decay accelerates), (iii) new evidence (vacuity drops, decay slows again). One frame per panel, 3 panels.

### III.E Implementation Details (~0.3 col, terse)
- LiDAR semantic head: **Cylinder3D** (pretrained, last layer fine-tuned with Sensoy 2018 EDL loss; **two heads in v3**: 20-D for closed-set RQ1/RQ4/RQ5, 15-D for open-set RQ2).
- Backbone: **NvBlox** [C10] forked, `α_v` (20 or 15 × float32) replaces label-probability vector; per-voxel memory ~180 B closed-set / ~160 B open-set (vs R2 ~100 B; documented in §IV).
- LVIO: FAST-LIO2 on KITTI-360, R3LIVE on SemanticKITTI single-session, public LIO config on SemanticSpray.
- **Hardware:** trained and evaluated on 1× RTX 4060 (16 GB); **deployment benchmarked + 30-s demo video on Jetson Orin NX (16 GB), target ≥ 5 Hz**. This is the firm deployability claim in §IV.G and the supplementary video referenced in §IV.H.
- *Cites:* [C10] NvBlox; [B2.7] GS-LIVO Orin NX comparison line.

### III.F Dynamic-Object Handling Discussion (NEW in v3, ~0.5 col)
- *Claim:* EvidLife-Map does **not** explicitly model 4D space-time. Two structural mechanisms in our pipeline cause **vacuity to naturally degrade on moving-object voxels**: (a) inconsistent per-voxel evidence across frames keeps `Σ α_v` low relative to inter-class disagreement, raising vacuity, and (b) the M3 decay (M3.1) under elevated vacuity drives `τ → τ_min ≈ 60 s`, actively flushing transient evidence. The result: dynamic objects are handled *implicitly* through (a)+(b); Khronos handles them *explicitly* through a dedicated learned short-/long-term factoriser. This sets up the fair-comparison framing in §IV.G Table V.
- *Claim (reviewer-defense):* An explicit short-term motion segmenter bolted onto our pipeline would *override* vacuity's natural-instability signal in M3, performing the same job twice with two parameter sets. v3 deliberately keeps the implicit channel only, documents the trade-off, and lets the Table V numbers decide.
- *Cites:* Khronos [A2] (the explicit factorisation we compare against); LTC-Mapping [A12] (long-term object handling lineage).
- *No new figure for §III.F (overflow budget).* The mechanism (a)+(b) is illustrated indirectly by Figure 4 panel (ii) (sudden disappearance → vacuity rises → decay accelerates).

**Method-section citation budget:** ~12 refs (most reused from §II; +2 for §III.F).

---

## §IV Experiments (~2 columns, ~1300 words; tighter in v3 due to added Table V and §IV.H)

### IV.A Datasets, Metrics, Protocol (~0.25 col)
- *Claim:* Five public datasets (SemanticKITTI — including the v3 open-set 14/5 split and the dynamic split, nuScenes-LiDARSeg, KITTI-360, SemanticSpray). **Robo3D-SemanticKITTI retained as fallback substrate for RQ2 if open-set split is blocked at W5; numbers if used reported in supplementary.** All dense GT or deterministic labels; **zero manual annotation needed**.
- *Table II (datasets × RQ matrix):* 5 rows × 5 cols (one per RQ including the v3-new RQ5 dynamic).

### IV.B Baselines and Implementation (~0.2 col)
- *Claim:* Six baselines: R2-reimpl [A1], NvBlox-vanilla [C10], ConvBKI [C9], Kimera-Semantics [C6], **Khronos [A2] (both RQ1 closed-set AND RQ5 dynamic head-to-head in v3)**, Clio-LiDAR-stub [A3] (open-set RQ2 secondary). All hyperparameters, seeds, and CUDA versions in supplementary. If Khronos full reproduction fails by W20, Table V switches to a published-numbers comparison with disclaimer (research plan §6 R-17 mitigation).

### IV.C RQ1 — Closed-set mIoU and Calibration (~0.3 col)
- *Claim (target):* EvidLife-Map achieves ≥ +2 mIoU over R2-reimpl and ≥ −20 % ECE over ConvBKI on SemanticKITTI seq 08, 11-21; cross-domain check on nuScenes-LiDARSeg confirms the gap survives a taxonomy shift.
- *Table III (main result):* method × {mIoU, mAcc, ECE, Brier, latency-ms, GPU-MB}; 6 rows × 6 cols. (AUROC-unknown moved out of Table III in v3 → into Table IV for RQ2 lead.)

### IV.D RQ2 — Open-set Vacuity for OOD Detection (~0.4 col, v3 LEAD RQ; replaces v2 Robo3D-headline RQ2)
- *Claim (target):* On the SemanticKITTI 14-known/5-unknown open-set split (withheld classes: bicyclist, motorcyclist, truck, other-vehicle, other-ground; pre-registered in supplementary), EvidLife-Map's vacuity achieves **voxel-level AUROC ≥ 0.80 and AUPR ≥ 0.60** for unknown-vs-known discrimination, with **closed-set mIoU on the 14 known classes dropping by ≤ 1.5** versus the 19-class fully-supervised baseline. **Robustness split (16/3)** and **nuScenes reverse cross-domain check (AUROC ≥ 0.75 on nuScenes-only classes)** confirm the result is not split-cherry-picked.
- *Cites:* [C9] (ConvBKI as the closest probabilistic-voxel baseline for OOD comparison); [D10] (LatentBKI as open-vocab contrast); [A3] (Clio-LiDAR-stub as open-set secondary).
- *Figure 5:* vacuity histogram overlay — known-class voxels vs unknown-class voxels — on a representative SemanticKITTI val frame from the 14/5 split; clear bimodality is the visual demonstration of H2.
- *Table IV (RQ2 LEAD):* split (14/5 primary, 16/3 robustness, nuScenes reverse) × {AUROC, AUPR, closed-set-mIoU-on-known}; methods rows = {EvidLife-Map, ConvBKI, Clio-LiDAR-stub, R2-reimpl}. ~3 splits × 3 metrics × 4 methods = compact 4×9 table or split into 3 small sub-tables.
- *Fallback:* if open-set lead path falls through at W17 G-5, Table IV switches to Robo3D corruption columns (per-corruption mIoU + vacuity-AUROC) and the §IV.D claim re-references the research plan §2 H2-fallback formulation. Pipeline switch costs zero datasets.

### IV.E RQ3 — Uncertainty-aware Traversability on SemanticSpray (~0.2 col)
- *Claim (target):* ≥ +10 pp safe-region recall and ≥ −30 % false-traversable rate vs R2-style hard-rule baseline on SemanticSpray passive replay.
- *Cites:* [B3.9].
- *Table VI (RQ3) renumbered from v2 Table V:* method × {safe-region recall, false-traversable rate, deferral rate, coverage}; 3 rows × 4 cols.

### IV.F RQ4 — Lifelong Multi-Session on KITTI-360 (~0.35 col)
- *Claim (target):* ≥ 80 % stale-voxel removal precision over 5+ revisit pairs; ECE drift across sessions ≤ 1.5× single-session ECE; map size growth ≤ 1.7× across 5 sessions.
- *Figure 6:* per-revisit-pair stale-voxel precision-recall curves; one curve per pair, overlaid.
- *Table VII (RQ4) renumbered from v2 Table VI:* revisit-pair × {stale-voxel P, R, F1; ECE; map size growth}; 5 rows × 5 cols.

### IV.G Dynamic-Scene Comparison: EvidLife-Map vs Khronos (~0.35 col, NEW in v3 — RQ5 head-to-head; also covers Ablations and Jetson Runtime to keep section budget)
- *Claim RQ5 (target):* On the SemanticKITTI dynamic split (frames with ≥ 5 % moving-* GT points from seq 00, 04, 05, 07), EvidLife-Map achieves **dynamic-object mIoU within 3 mIoU of Khronos**, **static-region recall within 1 pp of Khronos**, while **maintaining ≥ 5 Hz on Jetson Orin NX** (Khronos is not designed for Orin NX). The framing in §III.F (implicit vacuity-instability vs explicit 4D factoriser) is what makes this a fair comparison.
- *Cites:* [A2] Khronos (the head-to-head); [A12] LTC-Mapping (related long-term object handling).
- **Table V (NEW in v3):** *Dynamic-Scene Comparison: EvidLife-Map vs Khronos on SemanticKITTI dynamic split.* Columns: {dynamic-object mIoU, static-region recall, end-to-end latency on RTX 4060, end-to-end latency on Jetson Orin NX, GPU memory}. Rows: {EvidLife-Map (ours), Khronos (reproduced or published — flagged), R2-reimpl (lower-bound reference)}. 3 rows × 5 cols.
- *Claim ablations (consolidated into §IV.G to keep section budget):* A-1 (Dirichlet ±), A-2 (vacuity-conditioned τ ±), A-3 (entropy descriptor channel ±), A-4 (uncertainty-aware traversability ±), A-5 (conjugate inter-session fusion ±), **A-7 (open-set unknown channel ± — v3 firm)** each individually positive on ≥ 1 metric. **Jetson Orin NX** runs at ≥ 5 Hz on KITTI-360 streaming replay (target; to be measured).
- *Cites:* [B2.7] GS-LIVO Jetson contrast.
- *Table VIII (ablations + runtime) renumbered from v2 Table VII:* variable × {SemanticKITTI mIoU, KITTI-360 stale precision, **open-set AUROC** (new column), Orin NX Hz}; 6 firm rows + A-6 stretch row × 4 cols.
- *Figure 7 (drop if overflow):* latency CDF on 4060 vs Orin NX.

### IV.H Embedded Demo (NEW in v3, supplementary, ~0.1 col main text)
- *Claim:* A 30-second supplementary MP4 (`evidlife_demo.mp4`) shows EvidLife-Map running live on Jetson Orin NX over a SemanticKITTI seq 08 rosbag replay (10 Hz native; 300 frames) with the metric-semantic voxel map, **traversability overlay (RQ3 output)**, and a vacuity heat-map overlay visible in real time. The video is referenced from §IV.G and the main-text mention is ~3 sentences ("A 30-second demo at native 10 Hz on the Orin NX, with traversability and vacuity overlays, is provided as supplementary `evidlife_demo.mp4`; capture and overlay scripts are released."). **Promotes C4 deployability claim from "benchmark only" to "benchmark + visible demo"** and pre-empts the R-10 "but does it really run?" reviewer doubt.
- *No figure or table in main text* — the asset is the video.

**Experiments-section citation budget:** ~5 refs new beyond §II/III (+1 in v3 for the Khronos head-to-head context that was previously orthogonality-only).

---

## §V Discussion (~0.5 column, ~330 words)

- **Limitations.** (i) LiDAR-only — no claim on RGB / radar weather robustness; the v1 multi-modal story is explicitly out of scope. (ii) **Open-set evaluation depends on the chosen 14/5 split**; we mitigate via the 16/3 robustness split and nuScenes reverse-check, but acknowledge taxonomic split sensitivity (research plan §6 R-15). (iii) KITTI-360 multi-session pairs are odometry-overlap constructed (5+ pairs); not the same as long-duration months-apart revisits Boreas would offer. (iv) Lifelong tested single-robot only — Hydra-Multi-class collaborative scenarios out of scope. (v) Jetson Orin NX benchmark + demo video are replay-based; no real-robot deployment. (vi) **The Khronos head-to-head (RQ5)** uses either the authors' reproduced release or, if reproduction fails, published numbers with a disclaimer — Table V flags which cells are reproduced vs quoted.
- **Failure modes.** (a) When confidently-wrong evidence accumulates (e.g., persistent mirror surface ghost returns), vacuity stays artificially low and M3 does not trigger decay — a known evidential-deep-learning failure inherited from Sensoy 2018; also leaks into RQ2 by producing low-vacuity false negatives for "unknown" classes that visually resemble known classes. (b) When two sessions have non-overlapping classes (e.g., one captured before construction, one after), M2's class-histogram descriptor falls back to entropy-histogram only and precision degrades. (c) On taxonomically-close unknown classes (e.g., withholding `truck` while keeping `car`) the AUROC degrades, which the robustness split is designed to reveal.
- **Honest comparisons we did not win.** State explicitly any condition where ConvBKI / Khronos / Kimera-Semantics beat us; the v3 thesis is *integration* across three jobs + dynamic-scene competitiveness, not point-wise SOTA on any single one. **For Khronos specifically: if Table V shows Khronos still wins dynamic mIoU by > 3, we acknowledge it and reframe the win as a latency/integration trade-off, not a methodological replacement.** (Reviewer-credibility move; closes audit E6.)
- *Cites:* [A2] Khronos (orthogonal future combination — time × uncertainty); [B2.7] (GS-LIVO photometric complement).

---

## §VI Conclusion (~0.25 column, ~170 words)

One paragraph: recap of C1-C4 with the v3 emphasis that vacuity now drives open-set OOD detection directly. Single forward-looking sentence: *"Combining the per-voxel uncertainty axis of EvidLife-Map with the explicit spatio-temporal short-/long-term factorisation of Khronos [A2] — beyond the head-to-head competition reported in §IV.G — is a promising next step toward a fully integrated calibrated 4D MSM."* No new claims, no new cites.

---

## References (~1 column, ≤ 30 refs)

Drawn from `lit_scan.md` plus a small number of externally-added dataset/benchmark citations (flagged for citation-check pass). Mandatory inclusions (audit + lit-scan reviewer-threat list):

- **R2 baseline** — [A1].
- **Top-3 reviewer-threat must-cites** — Khronos [A2/D1] (now both prose-orthogonality AND experimental-comparison in v3), GS-LIVO [B2.7/D6], Clio [A3/D2].
- **Section-C foundations** — Voxblox [C1], Voxblox++ [C3], PanopticFusion [C4], Voxfield [C5], NvBlox [C10], Kimera [C6], Kimera-Multi [C7], ConvBKI [C9], S-BKI [C8].
- **Probabilistic / open-vocab competitors** — OpenVox [A8/D8], LatentBKI [D10], Open-Fusion [A10], SLIM-VDB [A9].
- **Open-vocab scene-graph context** — HOV-SG [B1.6], ConceptFusion [B1.1], OpenScene [B1.3].
- **Loop closure / long-term maintenance** — SA-LOAM [B4.4], LiDAR-LCD-with-Semantic-GAT [B4.1], PlaneSDF [B4.2], SLAM2REF [B4.3], LTC-Mapping [A12].
- **Datasets** — SemanticSpray [B3.9]; SemanticKITTI / nuScenes-LiDARSeg / KITTI-360 (standard handles, brought in via BibTeX); **Robo3D (Kong et al., ICCV-23 — must be added as new entry if fallback path engages; otherwise omittable)**.
- **Hydra family for completeness** — Hydra [A4], Hydra-Multi [A5].

That is 28 cited entries from lit_scan + 3-4 dataset-handle entries (SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, and Robo3D only if used) = 31-32 references. Trim 2 at submission (likely drop A5 Hydra-Multi and B4.3 SLAM2REF), keeping threat-list and dataset cites intact. **v3-specific:** Sensoy 2018 EDL is cited via Cylinder3D-context implicit; no new external citation needed for the open-set protocol (the split is our construction, with citation back to Sensoy via the EDL method §III.B).

---

## Figure / Table Budget Summary (v3 — Table V added, others renumbered)

| Asset | Section | Purpose |
|-------|---------|---------|
| Fig. 1 | §I teaser | R2 vs EvidLife-Map on KITTI-360 second visit + small open-set vacuity inset |
| Fig. 2 | §III.A | System diagram (M1+M2+M3 with vacuity arrows; v3 "(i) = open-set OOD" label) |
| Fig. 3 | §III.C | Loop-closure descriptor (h_class + h_entropy) schematic |
| Fig. 4 | §III.D | Single-voxel `α_v` + vacuity time-lapse, 3 panels |
| Fig. 5 | §IV.D | **(v3 NEW content)** vacuity histogram overlay on open-set 14/5 split — known vs unknown bimodality |
| Fig. 6 | §IV.F | Per-revisit-pair stale-voxel PR curves |
| Fig. 7 (optional) | §IV.G | Latency CDF, 4060 vs Orin NX |
| Tab. I | §III.B | Closed-form posterior comparison |
| Tab. II | §IV.A | Datasets × RQ matrix (5×5 in v3) |
| Tab. III | §IV.C | Main results RQ1 (mIoU / ECE / Brier / latency / mem) |
| Tab. IV | §IV.D | **(v3 LEAD)** Open-set RQ2 — splits × {AUROC, AUPR, closed-set mIoU} across 4 methods |
| Tab. V | §IV.G | **(NEW in v3)** Dynamic-Scene Comparison: EvidLife-Map vs Khronos vs R2-reimpl |
| Tab. VI | §IV.E | Traversability RQ3 (renumbered from v2 Tab. V) |
| Tab. VII | §IV.F | Multi-session lifelong RQ4 (renumbered from v2 Tab. VI) |
| Tab. VIII | §IV.G | Ablations + Jetson runtime (renumbered from v2 Tab. VII; +open-set AUROC column) |
| Supp. | §IV.H | **(NEW in v3)** `evidlife_demo.mp4` — 30-second Jetson Orin NX online-mapping demo |

Total in main text: 6 figures + 8 tables (with Fig. 7 optional, droppable to 6+7 if §IV.G overflows). Plus 1 supplementary video.

---

## Word / Column Budget Check (8-page IROS estimate, v3 tighter than v2)

| Section | Target words | Target columns |
|---------|--------------|----------------|
| Abstract | ~200 | 1 col |
| §I Intro | ~500 | 0.75 |
| §II Related | ~650 | 1.0 |
| §III Method | ~1900 (incl. v3 §III.F +0.5 col Dynamic-Object Discussion) | 3.0 |
| §IV Experiments | ~1300 (compressed: §IV.D shorter, §IV.G absorbs RQ5 + ablations + Jetson; §IV.H ~3 sentences) | 2.0 |
| §V Discussion | ~330 | 0.5 |
| §VI Conclusion | ~170 | 0.25 |
| References (~30 refs × ~25 words/ref) | ~750 | 1.0 |
| **Sub-total text** | **~5800** | **~9.5 cols** |
| Figures + tables (estimated visual area; +1 table for Table V Khronos head-to-head) | — | ~7 cols equivalent (up from 6.5 in v2) |
| **Total** | — | **~16.5 cols ≈ 8.25 pages double-column** |

Headroom in v3 is **tighter than v2** (8.25 vs 8 pages). First cuts if overflow: Fig. 7 latency CDF (already optional), then merge Tab. VI + Tab. VII into a single combined "RQ3 + RQ4 supplementary" table, then push the A-6 stretch-ablation row in Tab. VIII to supplementary. The Table V (Khronos dynamic head-to-head) and the §IV.H demo-video reference are **non-negotiable in v3** — they are the load-bearing v3 additions.

---

*End of paper outline v3. Aligned with `research_plan.md` (v3). v2 outline frozen at `paper_outline_v2_archived.md`; v1 outline frozen at `paper_outline_v1.md`. Advance to `/ars-full` only when all 6 §9 Go criteria of the v3 plan are met.*
