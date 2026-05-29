---
version: v2 (2026-05-28) — supersedes v1 (radar axis dropped due to K-Radar GT constraint)
v1 archived as: research_plan_v1.md (same directory)
v2 axis: EvidLife-Map  =  candidate X (EvidVox)  +  candidate Y (Lifelong + Map Decay)
v2 venue: IROS 2027 (deadline ≈ 2027-03; ≈ 9 months horizon)
v2 compute: 1× RTX 4060 (16 GB) + 1× Jetson Orin NX (deployment check only)
v2 datasets: SemanticKITTI / nuScenes-LiDARSeg / KITTI-360 / Robo3D-corrupted-SemanticKITTI / SemanticSpray  (NO K-Radar, NO FusionPortable, NO 4D-radar)
---

# Research Plan (v2)

**Working title (primary):** *EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay and Open-Set Vacuity*

**Tagline (one line):** A LiDAR-only online metric-semantic mapping system that (i) replaces the unspecified per-voxel Bayes filter of R2-class systems with a closed-form Dirichlet-evidential posterior carrying explicit unknown mass, (ii) maintains the map across sessions through confidence-aware loop closure and voxel-hash submap fusion, and (iii) ages stale voxels via a vacuity-driven decay rule so the same representation answers three reviewer-credible questions — calibration, open-set, lifelong — without inflating the modality stack.

**Author / lead:** (TBD by user)
**Target venue:** IROS 2027 (8 pages double-column, conference systems track).
**Prepared:** 2026-05-28
**Stage:** Stage-1 architect output (v2 reset) of the `academic-pipeline` skill; precedes any `/ars-full` execution.
**Upstream inputs:**
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\r2_audit.md` (R2 6-dim technical audit)
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md` (54-entry literature scan)
- `D:\_7_sci\semantic_mapping\_Online_Metric_Semantic_Mapping_for_Autonomous.txt` (R2 cleaned full text)
- `D:\_7_sci\semantic_mapping\_new_paper\drafts\research_plan_v1.md` (frozen v1, for diff)

---

## §0 Direction Decision — v2 reset (mandatory before §1)

v1 selected candidate **W = EviRad-Map** (LiDAR + 4D-radar evidential fusion) as the primary axis, with candidate **X = EvidVox** (LiDAR-only evidential) as fallback. The v1 §9 Go/No-Go listed **G-3** ("K-Radar pseudo-GT validated by 50-frame manual sparse-GT sanity sample ≥ 75 % agreement, by end of Week 3") as the controlling gate.

**v2 trigger.** The user's resource brief (2026-05-28) sets the manual K-Radar voxel annotation budget to **zero person-days**. With no manual sparse-GT, G-3 cannot be evaluated at all; the entire pseudo-GT defence collapses to "trust a pretrained Cylinder3D on out-of-domain frames", which is precisely the audit weakness (E1/E3) we promised to fix. v1 §6 R-1 therefore deterministically fires and v1 §0 fallback engages.

**v2 primary axis:** **EvidLife-Map** = candidate X (EvidVox) **combined with** candidate Y (Lifelong + Loop + Map Decay from v1 §0 Top-2 axis). This restores the "two strong contributions" balance that pure X alone would not carry — a system paper needs both an algorithmic and a systems contribution to fill 8 IROS pages credibly without radar.

| Axis | X. Evidential Voxel Fusion (R2 Top-1) | Y. Lifelong + Loop + Decay (R2 Top-2) | **X+Y. EvidLife-Map (v2)** |
|---|---|---|---|
| Novelty | 3 — crowded by ConvBKI [C9] / LatentBKI [D10] | 2 — Khronos [A2] owns dynamic-scene 4D MSM | **4** — Dirichlet-vacuity-as-decay-signal is the bridge nobody yet ships in one system |
| Venue-fit IROS 8 p | 3 — feels like a method paper, hard to fill 8 p | 4 — IROS systems track loves it | **5** — algorithm + system, fits IROS double column comfortably |
| R2 audit delta covered | S1/S2/S5 | T1/T2/T3 + T4 | **S1/S2/S4/S5 + T1/T2/T3/T4** (8 of 30 audit findings closed) |
| SOTA differentiation | Must beat ConvBKI in mIoU | Uphill vs Khronos | **Modality-axis fight skipped**; we fight on **calibration + lifelong** axis, where Khronos/Clio/ConvBKI each cover only one |
| Data availability (no manual) | 5 — SemanticKITTI / nuScenes-LiDARSeg dense GT | 4 — KITTI-360 has multi-session revisits | **5** — every dataset is fully public and pre-labelled |
| Engineering effort on 1× 4060 | 2 — Dirichlet head + Bayes update | 4 — submap + loop + decay | **4** — large but every module deliverable on a 4060 if ablation matrix is compressed |
| Reviewer risk | Low | Med (Khronos head-to-head) | **Low** — we declare orthogonality to Khronos (we factorise *uncertainty + time-since-last-evidence*, Khronos factorises *short-/long-term motion*) |
| Compute fit RTX 4060 | 5 — fits | 4 — long-sequence eval tolerable | **4** — tight but feasible with reduced ablation matrix (§4.4) and reduced training-epoch budget |

**Decision (v2 primary):** **EvidLife-Map** — three integrated modules (M1 Dirichlet-evidential per-voxel posterior; M2 confidence-aware loop closure + multi-session voxel-hash submap fusion; M3 vacuity-driven voxel decay + map staleness ageing) trained and evaluated entirely on public LiDAR-only datasets, deployable on Jetson Orin NX.

**Fallback (single layer only, no further nesting).** If by Week 4 the Dirichlet evidential head fails to beat the argmax-Bayes baseline by ≥ +1 mIoU on SemanticKITTI seq 08, retreat to a **calibration-only paper**: drop M2 and M3 to brief sub-sections, keep M1 as the headline, target **RA-L** instead of IROS (4-page brevity hides the missing systems story). This is documented in §9 G-2.

---

## §0.5 v1 → v2 Change Log

| Dim | v1 | v2 | Reason |
|---|---|---|---|
| **Primary axis** | **EviRad-Map** (LiDAR + 4D-radar evidential fusion) | **EvidLife-Map** (LiDAR-only evidential + lifelong + decay) | User: zero person-days for K-Radar manual annotation → v1 G-3 unverifiable → v1 fallback engaged |
| **Title** | EviRad-Map: Evidential LiDAR + 4D-Radar Online MSM for Weather-Robust Robot Navigation | EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay | follows axis |
| **Datasets** | SemanticKITTI / nuScenes-LiDARSeg / **K-Radar** (+ secondary ACDC, SemanticSpray, Boreas reserve) | SemanticKITTI / nuScenes-LiDARSeg / **KITTI-360** / **Robo3D-corrupted SemanticKITTI** / SemanticSpray | K-Radar dropped (no manual GT); KITTI-360 added for lifelong revisits; Robo3D added for corruption robustness (replaces weather); ACDC dropped (image-only, RGB head no longer central); FusionPortable explicitly excluded (user: clean public only) |
| **RQ2** | "Does 4D-radar fusion close the weather gap?" | "Does vacuity detect open-set / corruption-unknown voxels reliably under Robo3D-style point-cloud corruptions?" (v2 RQ2 recommendation — see §2) | Radar branch gone; need a defensible RQ2 of equal ambition that runs on a 4060 |
| **RQ4** | Lifelong as appendix-grade stretch | Lifelong **promoted to first-class RQ4 in main paper** | IROS 8 p can carry it; v1 RA-L tightness no longer applies |
| **Modules** | 4 modules (LiDAR head, radar head, evidential fusion, trust schedule, decay, loop) | 3 modules: M1 evidential, M2 loop+submap-fusion, M3 vacuity-decay | leaner, no radar, no trust schedule |
| **Baselines** | R2-reimpl + NvBlox-vanilla + ConvBKI + OpenVox + Khronos + L4DR stretch | R2-reimpl + NvBlox-vanilla + ConvBKI + **Kimera-Semantics** + **Khronos** (closed-set RQ1 only) + **Clio-LiDAR-stub** (open-set RQ2 only) | Removed L4DR (no radar); added Kimera-Semantics as canonical Voxblox-class baseline; added Clio LiDAR stub for open-set comparison |
| **Ablation matrix** | 7 ablation variables (A-1…A-7) on 4090 | **5-7 ablation variables** but each on 2 sequences only on 4060; A-2 (radar ±) removed; A-3 (trust ±) removed; new A-1' (decay schedule shape), A-2' (loop descriptor) added | 4060 wall-clock ≈ 2.5× 4090 per epoch; ablation grid had to shrink |
| **Compute** | 1× **RTX 4090 (24 GB)** + Jetson Orin NX deployment check | 1× **RTX 4060 (16 GB)** + Jetson Orin NX deployment check | User hardware reality; halves batch sizes; doubles training wall clock |
| **Venue** | RA-L primary (4 p) + IROS fallback | **IROS 2027 primary** (8 p) directly; RA-L only as fallback if M2/M3 fail (§9 G-2) | User preference for IROS 8 p over RA-L 4 p; ample room for lifelong story |
| **Timeline** | 10 weeks to RA-L | ~36 weeks (≈ 9 months) to IROS 2027 deadline ≈ 2027-03 | Real horizon; allows 4060 slow training plus reviewer-sim cycle |
| **Risk register** | R-1 (K-Radar GT), R-5 (radar head) high | R-1/R-5 **retired**; new R-11 (4060 VRAM), R-12 (ConvBKI reimpl), R-13 (KITTI-360 revisit length), R-14 (evidential head instability) | Risks remapped to actual stack |

---

## §1 Problem Statement & Motivation (v2)

### 1.1 Scenario
A ground robot operates outdoors over weeks of repeated traversal of the same area (campus, urban block, industrial site). It must build, online and on-board (Jetson-class GPU), a dense voxel map labelled with closed-set semantics plus an explicit "unknown" channel, and must keep that map correct across (i) **single-session noise** (LiDAR drop-out, point-cloud corruption, sensor occlusion), (ii) **open-set encounters** (objects/textures outside the training set), and (iii) **inter-session change** (revisits weeks later with moved objects, construction, new vegetation). Camera and radar are *not* assumed; the system is LiDAR-only with optional RGB-derived priors at training time only.

### 1.2 R2 baseline summary and the four reviewer-credible weaknesses
R2 [A1] (Jiao et al., HKUST) couples LVIO with an NvBlox TSDF backbone and a confidence-aware HRNet [P1] segmenter, fused per-voxel through an **unspecified** iterative Bayes filter. The R2 audit, distilled to the four cripple-points an even mildly-attentive RA-L/IROS reviewer will raise:

1. **Methodological under-specification.** R2's Bayes update formula, prior, and confidence-into-fusion pathway are all missing (audit S1/S2/S4). Reproducible? No.
2. **No quantitative evidence.** Zero ablation, zero baseline number, zero mIoU, zero F-score; evaluation only on two self-recorded sequences (audit E1/E2/E3/E5).
3. **No open-set / OOD handling.** "Unlabeled = untraversable" (R2 p.3, audit S5); pessimistic and unusable for active exploration.
4. **No long-term / loop-closure / decay mechanism.** Lifelong drift, stale voxels, re-visits all unaddressed (audit T1/T2/T3/T4).

v1 sought to fix (1)+(3) algorithmically and add a fifth axis (multi-modal weather robustness). v2 drops the modality axis and instead pairs (1)+(3) algorithmic fix with a (4) systems fix — which is exactly the audit's largest cluster of unfilled weaknesses, and is achievable on a 4060.

### 1.3 Differentiation vs current SOTA (v2 re-framed)
The literature scan flags three SOTA threats; v2 declares orthogonality to each:
- **Khronos [A2/D1]** factorises *short-term motion vs long-term change in time*; v2 factorises *evidence strength vs vacuity in observation* and uses time only as a decay scalar in M3. The decay rule (Eq. M3.1) is dual to Khronos's "long-term change detector" — it requires no explicit change-detection network, only a Dirichlet-conjugate exponential. Reviewers can be told: "Khronos detects change; we let calibration *reveal* change."
- **Clio [A3/D2]** compresses *given a task*; v2 produces a *task-agnostic, uncertainty-aware substrate* that Clio could be re-implemented on top of. The Clio LiDAR stub (one of our open-set baselines, §4.2) is the explicit comparison.
- **ConvBKI [C9] / LatentBKI [D10]** use kernel Bayesian inference for *spatial smoothing*; v2 uses **EDL Dirichlet** for closed-form per-voxel epistemic uncertainty *without* spatial-kernel smoothing — which yields a sharper, more honest vacuity signal for the M3 decay rule, at the price of giving up smoothing's mIoU bump (which the M2 loop closure re-projection partially restores).

The novelty story for v2 is **not** "first to fuse modality X+Y" (v1 story); it is **first to wire one statistical quantity (Dirichlet vacuity) into three jobs simultaneously**: (a) open-set OOD score, (b) loop-closure descriptor entropy term, (c) lifelong decay trigger. That single quantity threading three modules is what reviewers will remember.

### 1.4 Thesis statement (v2)
> *A metric-semantic voxel map becomes simultaneously more calibrated, more open-set-aware, and more lifelong-maintainable when its per-voxel posterior is a closed-form Dirichlet-evidential distribution whose vacuity mass is reused as (i) the open-set / OOD score, (ii) the loop-closure descriptor's entropy channel, and (iii) the conjugate decay trigger that ages stale voxels — eliminating the three independent ad-hoc heuristics that prior systems (R2, ConvBKI, Khronos) use one each.*

---

## §2 Research Questions & Hypotheses (v2)

### RQ1 (preserved from v1) — Does Dirichlet-evidential per-voxel fusion outperform argmax-Bayes / S-BKI-style kernel Bayesian inference on closed-set dense semantic mapping?
- **H1.** On SemanticKITTI sequences 08, 11-21, an EDL-Dirichlet voxel update yields **≥ +2 mIoU** and **≥ −20 % ECE** versus an R2-reimplemented argmax-Bayes baseline at equal voxel size (0.25 m) and equal compute. (mIoU bar lowered from v1's +3 because we no longer have radar to push the upper end; ECE bar held.)
- **Boundary.** Holds for closed-set 19-class setting; for open-set we expect ECE win to grow and mIoU possibly to shrink because "unknown" steals mass from rare classes — quantified in RQ2.

### RQ2 (replaces v1 weather-robustness RQ) — **RECOMMENDED: corruption robustness via Robo3D.** Alternative: open-set vacuity vs unknown classes.
**Recommended formulation:**
> *Does the Dirichlet vacuity score reliably flag voxels whose point-cloud evidence is corrupted (fog-sim, snow-sim, beam-missing, range-noise), and does it degrade gracefully across the 8-corruption × 5-severity Robo3D-SemanticKITTI grid where ConvBKI's kernel-smoothed confidence stays artificially high?*

- **H2 (recommended).** On Robo3D-SemanticKITTI, EvidLife-Map's mean **mIoU drop across the 8 corruption types at severity 3 is ≤ 50 %** of the drop suffered by R2-reimpl and ≤ 70 % of the drop suffered by ConvBKI; simultaneously, voxel-level **AUROC of vacuity-as-corruption-detector ≥ 0.80** averaged across corruption types.
- **Boundary.** Robo3D is synthetic-corruption (deterministic transforms applied to clean SemanticKITTI), not real adverse weather; we acknowledge this in §V and use SemanticSpray real-wet-road as a sanity cross-check (no claim of leaderboard supremacy).

**Why recommend corruption (Robo3D) over open-set as the RQ2 lead?**
1. **Established benchmark with leaderboard numbers.** Robo3D (Kong et al., ICCV-23/24) publishes baseline mCE / mRR for every major LiDAR seg model; ConvBKI, Cylinder3D, SPVCNN numbers are already in the literature. Reviewers see a known scoreboard, not a self-defined one.
2. **Single dataset, no extra label work.** Open-set evaluation needs an out-of-distribution class split (SemanticKITTI → nuScenes-LiDARSeg "unknown" mapping, or REAL-D-style manual unknown labels). Robo3D corruption labels are deterministic transforms — zero annotation budget. Critical given v2's zero-annotation constraint.
3. **Vacuity is a natural corruption detector**, and the resulting story ("the same vacuity scalar that we use for decay ALSO detects corruption") deepens the §1.4 thesis. Open-set could be added as a secondary table without inflating the page count.
4. **Avoids reviewer trap of demanding a real adverse-weather dataset.** Without K-Radar / Boreas we cannot answer "real weather". Robo3D side-steps that: it is explicitly synthetic corruption, no real-weather claim made.

**Alternative formulation (open-set via vacuity), kept as the RQ2 backup if Robo3D reimpl is blocked at W3:**
> *Does vacuity rank unknown voxels (SemanticKITTI test classes withheld at training time) above known voxels by AUROC ≥ 0.80?*
This is a standard EDL evaluation pattern (Sensoy et al., NeurIPS-18 style) and is publishable, but covers less reviewer ground than the corruption story.

**Decision rule.** Go with **Robo3D corruption** (H2 above). If the Robo3D-SemanticKITTI corruption pipeline reimpl is not running by end of Week 5, fall back to open-set vacuity. Both rely only on existing public assets.

### RQ3 (kept conceptually from v1, descoped) — Does propagating evidential uncertainty into downstream traversability improve navigation safety on public closed-loop replay scenarios?
- **H3.** On SemanticSpray-style passive-replay traversability evaluation, an uncertainty-gated traversability head produces **≥ +10 pp safe-region recall** and **≥ −30 % false-traversable rate** versus an R2-style hard-rule baseline, at equal coverage. (Numbers softened from v1's closed-loop sim because there is no Isaac Sim time budget on a 4060 + Orin setup — RQ3 is passive replay only.)
- **Boundary.** Passive replay (no actuated navigation) — we explicitly do not claim closed-loop navigation success in v2.

### RQ4 (promoted from v1 appendix to main, given IROS 8 p budget) — How does the evidential map scale and degrade gracefully across multi-session revisits?
- **H4.** Across **KITTI-360 sequences 00, 02, 04, 05, 06, 07, 09, 10** (≥ 5 revisit pairs constructed from spatial overlap), the M2 confidence-aware loop closure + M3 voxel-hash submap fusion achieves: (i) **stale-voxel removal precision ≥ 80 %** over the second-pass trajectory after evidence decay (M3.1), (ii) **inter-session map ECE drift ≤ 1.5×** the single-session ECE, (iii) **multi-session memory footprint ≤ 1.7×** the single-session footprint (sub-linear in session count thanks to voxel-hash deduplication). All three measured on the KITTI-360 evaluation split.
- **Boundary.** Single-robot multi-session only; not claimed for collaborative multi-robot map merging (Hydra-Multi [A5] territory). Revisit pairs constructed by us via odometry-overlap (documented in §4.3); reviewers can audit the overlap script.

---

## §3 Technical Approach (v2)

### 3.1 System architecture (ASCII, v2 three modules)

```
                    +-------------------+
                    | LiDAR (OS1-128 /  |
                    | Velodyne HDL-64)  |
                    +---------+---------+
                              |
                              v
              +---------------+----------------+
              | LiDAR Semantic Head             |
              | (Cylinder3D pretrained on       |
              |  SemanticKITTI; closed-set 19   |
              |  + 1 "unknown" channel = 20-D   |
              |  Dirichlet evidence vector e_L) |
              +---------------+-----------------+
                              |
                              v
   +--------------------------+------------------------+
   | §3.2  M1  EVIDENTIAL PER-VOXEL POSTERIOR          |
   | per-voxel alpha_v in R^{20}                       |
   | vacuity m_u(v) = 20 / sum(alpha_v)                |
   | Eqs. M1.1-M1.3 (closed form, no kernel smoothing) |
   +-------------+---------------------+----------------+
                 |                     |
                 v                     v
   +-------------+---------+   +-------+---------------------+
   | LVIO state (R3LIVE-   |   |  RGB camera (training-time  |
   | class, reused)        |   |  prior only; OPTIONAL,      |
   +-----------+-----------+   |  zero at deployment)         |
               |               +-----------+------------------+
               v                           |
   +-----------+----------------+          |
   | Voxel-Hash Submap Backbone |<---------+
   | (NvBlox forked, alpha_v    |
   |  replaces label-prob vec)  |
   +-----+-----------+----------+
         |           |
         v           v
+--------+----+   +--+---------------------------------------+
| §3.3  M2     |  | §3.4  M3  VACUITY-DRIVEN DECAY +         |
| LOOP CLOSURE |  | MAP STALENESS AGEING                     |
| + SUBMAP     |  | Eq. M3.1 conjugate exponential decay     |
| FUSION       |  | tau = f(vacuity); high-vacuity ages fast |
| Eq. M2.1     |  | low-vacuity ages slowly = persistent map |
| descriptor   |  +--------------------+---------------------+
| d(S) =       |                       |
| (h_class,    +-----------+-----------+
| h_entropy)   |
+------+-------+           v
       |           +-------+---------------+
       v           | downstream queries:   |
+------+-------+   | (a) uncertainty-aware |
| pose graph   |   |     traversability    |
| optimise +   |   | (b) language / task   |
| reproject    |   |     layer (Clio-      |
| alpha_v      |   |     compatible)       |
+--------------+   +-----------------------+
```

### 3.2 M1 — Evidential per-voxel posterior (inherits v1 Module 1)
Replace R2's unspecified Bayes filter (audit S1/S2) by a Dirichlet-evidential posterior. Per voxel `v`, maintain accumulated evidence `α_v ∈ R^{C+1}` where C = 19 SemanticKITTI classes and the last channel is the open-set "unknown" channel. Vacuity is `m_u(v) = (C+1) / Σ α_v` (closed form, no spatial smoothing). New observations contribute evidence `e_v(z) = softplus(logit(z))` (no per-modality trust schedule; v2 has only one modality). Eqs.:

- **(M1.1)** Evidence accumulation: `α_v^{t+1} = α_v^{t} + e_v^{t+1}`.
- **(M1.2)** Posterior class mean: `E[p_c | α_v] = α_{v,c} / Σ_k α_{v,k}`.
- **(M1.3)** Vacuity (open-set / OOD score, reused in M2 and M3): `m_u(v) = (C+1) / Σ_k α_{v,k}`.

EDL Dirichlet (Sensoy et al., NeurIPS-18) rather than S-BKI / ConvBKI kernel-Bayesian inference, because (i) we want per-voxel epistemic uncertainty *without* the spatial-kernel coupling that makes vacuity contaminated by neighbour evidence — the M3 decay rule needs an honest per-voxel vacuity, and (ii) closed-form posterior keeps a 4060 viable.

### 3.3 M2 — Confidence-aware loop closure + multi-session voxel-hash submap fusion (new, replaces v1 §3.3 trust schedule)
Submaps sealed every 50 m or 30 s of trajectory. Per-submap descriptor:
- **(M2.1)** `d(S) = ( h_class(S),  h_entropy(S) )` where `h_class(S)` is the L1-normalised class histogram over confident voxels (vacuity below median), and `h_entropy(S)` is the histogram of per-voxel posterior entropy bucketed to 10 bins. Matching via cosine on `h_class` + Earth-Mover-Distance on `h_entropy`; verified by semantic-ICP restricted to class-consistent confident voxels.

Once a loop is verified, pose graph optimisation re-projects each voxel's `α_v` consistently across the loop; in submap overlap regions, `α_v` from different sessions are merged by **conjugate addition** `α_v ← α_v^{(s1)} + α_v^{(s2)}` (closed-form Dirichlet posterior over independent observations), so fusion is parameter-free.

Inter-session storage: voxel-hash key = `(int(x/v), int(y/v), int(z/v))`, value = `α_v`. New sessions write into the same hash; deduplication is implicit.

Inspired by: Kimera-Multi [C7] (pose graph + multi-robot), Hydra [A4] (per-submap descriptor), SA-LOAM [B4.4] (semantic-aided LCD), PlaneSDF [B4.2] (cross-session change detection). Distinct from all four because the *descriptor entropy channel* and the *parameter-free Dirichlet conjugate fusion* are both new.

### 3.4 M3 — Vacuity-driven voxel decay + map staleness ageing (new, sharper than v1 §3.4 decay)
Conjugate exponential decay on `α_v`, but with the decay time-constant `τ` itself a function of vacuity:

- **(M3.1)** `α_v^{t+Δ} = ((α_v^{t} − 1) · exp(−Δ / τ(m_u))) + 1`, where `τ(m_u) = τ_max · (1 − m_u) + τ_min · m_u`, with `τ_max ≈ 3600 s` (one hour) for confidently-known voxels (low vacuity, ages slowly = persistent map) and `τ_min ≈ 60 s` (one minute) for high-vacuity voxels (ages fast = transient / unknown / dynamic stuff). This single equation is the v2 thesis (§1.4) made concrete: vacuity *is* the staleness signal.

The decay preserves the posterior mean (M1.2) while inflating vacuity over time, so stale voxels become re-writable as new evidence arrives. Submap-level ageing: a whole submap whose median vacuity exceeds a threshold is flagged for re-observation by an exploration policy (out of scope for this paper but the hook is present).

Distinct from v1 §3.4 (which used a fixed `τ`) and from Voxblox / NvBlox (which use monotonically-growing weight with no decay at all, audit T2).

### 3.5 Implementation Details (terse, IROS budget)
- LiDAR semantic head: Cylinder3D, pretrained weights from authors' release; we fine-tune only the last layer to a 20-D output (19 classes + unknown) using EDL loss (Sensoy 2018) on SemanticKITTI train split. Pretrain on 4060 ≈ 2 days for the last-layer fine-tune (entire backbone frozen).
- Backbone: NvBlox [C10] forked; `α_v` (20 × float32 = 80 B) replaces the 19-class label probability vector. Memory per voxel grows from ~100 B (R2) to ~180 B (v2); §4.3 budgets this.
- LVIO state estimator: reused unchanged (any open-source LIO; we use the FAST-LIO2 release on KITTI-360, R3LIVE on SemanticKITTI single-session, doc'd in §4.6).
- Runs target: ≥ 10 Hz on RTX 4060, ≥ 5 Hz on Jetson Orin NX. Numbers to be measured; deployment check only.

### 3.6 Key equations (closed-form, all derived in `/ars-full`)
- **Eq. M1.1** evidence accumulation
- **Eq. M1.2** posterior class mean
- **Eq. M1.3** vacuity
- **Eq. M2.1** submap descriptor (class histogram + entropy histogram)
- **Eq. M3.1** vacuity-conditioned conjugate decay
Five equations total; v1 had five too. Same equation count, different semantics.

### 3.7 Methodological deltas, one line each (v2 set)
- vs **R2 [A1]:** specifies the Bayes filter (Eqs M1.1-M1.3 closed form); adds vacuity, adds decay, adds loop+fusion.
- vs **Khronos [A2]:** Khronos factorises short-/long-term in time via change-detection network; we let vacuity *be* the change signal via M3.1 (no separate network).
- vs **Clio [A3]:** Clio compresses given a task; we keep a task-agnostic uncertainty-aware substrate. Clio could sit on top.
- vs **ConvBKI [C9]:** kernel-Bayesian spatial smoothing; we use EDL Dirichlet — sharper vacuity (no neighbour contamination), needed by M3.1.
- vs **LatentBKI [D10]:** open-vocab BKI; we are closed-set + open-set unknown channel, so we cleanly separate "known classes" and "unknown mass" rather than embedding everything in a CLIP-feature manifold.
- vs **OpenVox [A8/D8]:** Bernoulli per-instance; we are Dirichlet over all classes including unknown, and we add lifelong M2+M3.
- vs **S-BKI [C8]:** same spirit (probabilistic semantic voxel) but with explicit vacuity for decay; no kernel smoothing.
- vs **Voxblox / Voxblox++ [C1, C3]:** Voxblox-class does TSDF only; we add semantic Dirichlet + lifelong.
- vs **Kimera-Semantics [C6]:** Kimera does scene-graph + multi-robot; we focus on per-voxel calibration + single-robot multi-session lifelong.

---

## §4 Experimental Plan (v2)

### 4.1 Datasets (final shortlist, 5 datasets — all public, all pre-labelled, zero manual budget)

| # | Dataset | Role in v2 | Why kept / why added |
|---|---------|------------|----------------------|
| D-a | **SemanticKITTI** (Behley 2019; community Cylinder3D split) | Primary closed-set dense GT for LiDAR mapping mIoU and ECE (RQ1) | Standard, reproducible, every cited competitor has results on it |
| D-b | **nuScenes-LiDARSeg** (Caesar 2020; LiDAR seg labels released 2021) | Cross-domain mIoU/ECE check (RQ1 generalisation) and the open-set RQ2-backup arena (nuScenes classes not in SemanticKITTI become "unknown") | The only mainstream AD dataset with LiDAR semantic GT *and* a different class taxonomy from SemanticKITTI, perfect for open-set splits |
| D-c | **KITTI-360** (Liao 2022; multi-session dense LiDAR semantic seg, 19 classes) | Primary lifelong / multi-session evaluation arena (RQ4) | Multi-session structure (revisits across drives), large enough for the M2 loop closure tests; has dense semantic GT |
| D-d | **Robo3D-corrupted SemanticKITTI** (Kong 2023, ICCV-23: 8 corruption types × 5 severities applied deterministically to SemanticKITTI val) | Corruption robustness evaluation (RQ2 recommended) | Established benchmark with public leaderboard; no annotation needed (corruption labels are deterministic) |
| D-e | **SemanticSpray** [B3.9] (RA-L-24 wet-road LiDAR seg) | Sanity cross-check for RQ2 (real wet-road vs Robo3D synthetic) and passive-replay traversability arena (RQ3) | Real-world LiDAR-only, no radar dependence, no manual annotation needed (authors released labels) |

**Datasets explicitly NOT used and why:**
- **K-Radar** — dropped (user: zero annotation budget for dense voxel pseudo-GT validation).
- **FusionPortable** — dropped (user instruction: clean public datasets only).
- **Boreas [B3.4]** — would be ideal for lifelong + multi-season but the LiDAR semantic GT is sparse; KITTI-360 covers the multi-session need with dense GT.
- **ACDC [B3.3]** — image-only adverse-condition seg; v2 has no central RGB semantic head, so ACDC contributes nothing.
- **CADC [B3.5]** — adverse-weather LiDAR detection, no dense semantic GT.
- **Replica / TUM / ScanNet** — indoor, off-topic.

### 4.2 Baselines (6 total, IROS 8 p budget)
1. **R2-reimpl.** Faithful Python/CUDA reimplementation of R2 [A1] (NvBlox + Cylinder3D + argmax-Bayes). No code released by R2, so we must build it ourselves; this is the v1 baseline-1, kept.
2. **NvBlox-vanilla + per-frame argmax.** Lower bound (no temporal fusion).
3. **ConvBKI [C9].** Authors' code; canonical probabilistic semantic voxel competitor (lit_scan reviewer-threat #3). **R-12 risks reimpl failure** — see §6.
4. **Kimera-Semantics [C6].** Authors' release; canonical Voxblox-class semantic baseline; covers the "did you cite the seminal MIT-SPARK ancestor" reviewer demand. New in v2 (v1 omitted).
5. **Khronos [A2]** — *RQ1 closed-set mIoU only on SemanticKITTI* (we do not compete on dynamic-scene 4D factorisation, which is its strength). Authors' release is reachable; if blocked, demoted to citation-only and a §V honest acknowledgement.
6. **Clio-LiDAR-stub** [A3] — Clio's LiDAR pathway used as the open-set RQ2-backup baseline; if Clio's release is RGB-D-only, we approximate with the OpenScene [B1.3] LiDAR distillation.

Stretch baseline (not Go criterion): **OpenVox [A8/D8]** if their code drops by W6.

### 4.3 Metrics
| Tier | Metric | Used in RQ | Notes |
|------|--------|-----------|-------|
| Mapping | mIoU, per-class IoU, F@5 cm reconstruction, voxel coverage, GPU memory peak, end-to-end latency (mean + 99-pct) | RQ1, RQ4 | closes audit E4/E5 |
| Uncertainty | Expected Calibration Error (ECE), Brier score, AUROC for *unknown* | RQ1, RQ2, RQ4 | distinguishes us from R2 and ConvBKI |
| Corruption | per-corruption mIoU drop on Robo3D-SemanticKITTI; mean Corruption Error (mCE); voxel-level AUROC of vacuity-as-corruption-detector | RQ2 (recommended) | replaces v1 weather Δ-mIoU |
| Open-set (backup) | AUROC of vacuity ranking unknown voxels above known | RQ2 (backup) | only run if Robo3D path fails by W5 |
| Traversability | safe-region recall, false-traversable rate, deferral rate | RQ3 | passive replay on SemanticSpray; no closed-loop sim in v2 |
| Lifelong | stale-voxel removal precision/recall over multi-session trajectory; ECE drift across sessions; map size growth | RQ4 | KITTI-360 multi-session split |

**Pseudo-GT?** Not needed in v2. Every dataset above has either dense semantic GT (D-a/D-b/D-c/D-d/D-e) or deterministic corruption labels (D-d). The v1 K-Radar pseudo-GT bridge is deleted.

### 4.4 Ablations (5-7 variables, compressed for 4060 budget)
| # | Variable | Question answered | Compute cost on 4060 (est.) |
|---|---|---|---|
| A-1 | ± Dirichlet evidential head (vs argmax-Bayes / vs softmax-Bayes) | RQ1: does evidential improve mIoU + ECE? | 3 × 2 days = 6 days |
| A-2 | ± vacuity-conditioned decay τ(m_u) (vs fixed-τ / vs no-decay) | RQ4 + M3 isolation: does vacuity-coupled decay beat fixed decay? | 3 × 1.5 days = 4.5 days |
| A-3 | ± entropy channel in submap descriptor (h_class only vs h_class + h_entropy) | RQ4 + M2 isolation: does the entropy channel improve loop precision? | 2 × 1.5 days = 3 days |
| A-4 | ± vacuity-aware traversability (vs hard threshold) | RQ3: does propagating uncertainty improve safety metrics? | 2 × 0.5 days = 1 day |
| A-5 | ± conjugate Dirichlet fusion across sessions (vs naive overwrite) | RQ4 + M2 isolation: does the parameter-free conjugate fusion help? | 2 × 1 day = 2 days |
| A-6 (kept-if-time) | ± voxel size (0.10 m vs 0.25 m) | defensive: are mIoU gains a voxel-size artefact? | 2 × 2 days = 4 days |
| A-7 (kept-if-time) | ± openset channel in M1 (20-D vs 19-D) | RQ2: does the dedicated unknown channel matter, or does vacuity alone suffice? | 2 × 2 days = 4 days |

**Total core ablations (A-1…A-5):** ≈ 16.5 GPU-days on 4060. **Including A-6 + A-7:** ≈ 24.5 days. Plus full main eval runs ≈ 10 days, total ≈ 35 GPU-days for ablations + main results. Fits in 5 weeks of W6-W10 (§7) with overhead for re-runs.

**4060 ablation budget verdict.** Realistically we can run **5 ablations (A-1…A-5) firmly + 1 of {A-6, A-7} as time allows**. Drop A-6 first if pressed (voxel-size confound is rebuttal-only); keep A-7 because the open-set channel is a methodological choice reviewers will probe.

### 4.5 Compute budget (v2 reality on 4060 + Orin NX)
- **Training and main eval:** single workstation, 1× **RTX 4060 (16 GB)**. Batch size ≈ ½ of a 4090 run; wall clock ≈ 2.5× per epoch (16 GB vs 24 GB + ~40 % fewer CUDA cores).
- **Deployment check:** 1× Jetson Orin NX (16 GB) — only for §III.E latency + memory; not used for training.
- **Estimated wall-clock budget across the 36 weeks:**
  - Cylinder3D last-layer EDL fine-tune on SemanticKITTI: ~2 days × 4060.
  - R2-reimpl bring-up: ~5 days (engineering, not GPU-bound).
  - ConvBKI baseline reproduction: ~3 days (if authors' code works).
  - Kimera-Semantics baseline: ~2 days.
  - Khronos closed-set RQ1 run on SemanticKITTI: ~2 days.
  - Main mapping eval over SemanticKITTI seq 08+11-21 per system: ~1 day × 6 systems = 6 days.
  - Robo3D corruption sweep (8 × 5 = 40 conditions, our system + 3 baselines): ~6 days.
  - KITTI-360 multi-session eval (8 sequences, ~5 revisit pairs): ~4 days.
  - SemanticSpray passive replay (RQ3): ~1 day.
  - Ablations: ≈ 25 GPU-days (§4.4 above).
  - Jetson Orin NX deployment latency / memory check: ~2 days.
  - **Total: ≈ 60 GPU-days of 4060 time.** At 5-6 productive GPU-days per calendar week (queue overhead, debugging, re-runs), that is ~12-15 weeks of pure runtime, comfortably inside the W4-W18 window of §7's 36-week IROS schedule.

### 4.6 Reproducibility hygiene (closes audit E7)
- Code + Docker + ROS 2 launch + EDL fine-tune script + Robo3D corruption applier (reused from Kong 2023 release) + KITTI-360 revisit-pair builder script + per-experiment seed list + hyperparameter table.
- LVIO choice documented per dataset: FAST-LIO2 on KITTI-360 (multi-session friendly), R3LIVE on SemanticKITTI single-session, public LIO config on SemanticSpray.

---

## §5 Expected Contributions (v2, 4 contributions)

- **C1 (Algorithm).** A Dirichlet-evidential per-voxel posterior whose **single vacuity scalar simultaneously serves three downstream jobs** — open-set / OOD score (Eq. M1.3), submap descriptor entropy channel (Eq. M2.1), and lifelong decay trigger (Eq. M3.1). Strictly generalises R2's unspecified Bayes filter (audit S1/S2 closed) and adds the open-set channel (audit S5 closed). The "one scalar, three jobs" framing is the v2 novelty hook.

- **C2 (System).** A complete online metric-semantic mapping system EvidLife-Map with confidence-aware loop closure, parameter-free conjugate Dirichlet inter-session submap fusion, and vacuity-conditioned voxel decay — closing audit T1/T2/T3/T4 in one paper. Deployable on Jetson Orin NX (§III.E target ≥ 5 Hz). First system to ship the lifelong + open-set + calibration triad together on a public LiDAR-only stack.

- **C3 (Empirical).** Across SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, Robo3D-corrupted SemanticKITTI, and SemanticSpray: (i) **≥ +2 mIoU and ≥ −20 % ECE** over a faithful R2-reimpl on SemanticKITTI (RQ1 H1); (ii) **≤ 50 %** of R2-reimpl's mean mIoU drop and **AUROC ≥ 0.80** for vacuity-as-corruption-detector on Robo3D (RQ2 H2); (iii) **≥ +10 pp safe-region recall** on SemanticSpray passive replay (RQ3 H3); (iv) **≥ 80 % stale-voxel removal precision and ≤ 1.5× ECE drift** on KITTI-360 multi-session (RQ4 H4). All numbers replaced by measured ones before submission; targets set as the §9 Go/No-Go thresholds.

- **C4 (Reproducibility + Deployability).** Code + Docker + ROS 2 launch + Jetson Orin NX deployment benchmark released — directly attacks R2 audit E7 ("复现性零保障"). The deployability claim ("first lifelong evidential MSM that runs at ≥ 5 Hz on a 16 GB Orin NX") is hardware-grounded; we report numbers, not aspirations.

---

## §6 Risk Register (v2)

| ID | Risk | Prob | Impact | Mitigation | Trigger to fall back |
|----|------|------|--------|------------|---------------------|
| R-2 (kept) | Khronos [A2] releases a multi-session / lifelong extension before IROS 2027 submission | M | H | Track arXiv weekly via post-research literature monitor; pre-register our differentiation as "vacuity-coupled decay, no separate change-detection network" | If Khronos-v2 lands on lifelong before us, sharpen claim to the "one vacuity, three jobs" calibration story |
| R-3 (kept) | GS-LIVO [B2.7] HKUST sibling lab releases semantic-GSplat extension | M | M | Differentiate on representation (voxel-Dirichlet vs Gaussian) and target (lifelong vs photo-realistic single-session) | If GS-LIVO-semantic ships, sharpen our "Jetson Orin NX deployable lifelong" angle |
| R-4 (kept) | Dirichlet evidential fusion fails to beat ConvBKI in mIoU | M | H | Calibration (ECE / AUROC unknown / Robo3D AUROC) is a separate axis; we win on calibration + lifelong even if mIoU ties | If ECE also ties, retreat per §0 fallback to calibration-only paper at RA-L (drop M2/M3 to brief sub-sections) |
| R-6 (kept) | KITTI-360 multi-session revisit overlap insufficient for RQ4 | M | M | Pre-compute overlap matrix in W3 (deliverable W3.b); if pairs < 5, augment with SemanticKITTI cross-day sequences (08 vs 09-10 split) | If still < 5, demote RQ4 H4 to single-session ECE-drift study, document as scope reduction |
| R-7 (kept) | Voxel-size confound contaminates mIoU vs ConvBKI | M | M | Ablation A-6 if time allows; report all numbers also at fixed 0.25 m | n/a |
| R-8 (kept) | Cylinder3D licence (or its weight release) prevents code release | L | M | RangeNet++ (BSD) as backup; train from scratch on 4060 (~5 days) | n/a |
| R-9 (kept, sharpened) | 8-page IROS overflow given 4 RQs and 5 datasets | M | M | Move ablation tables, per-class IoU, KITTI-360 per-sequence numbers to supplementary; keep main paper at 4 RQs but 3 hero claims | Drop A-6/A-7 from main, defend "single-overlap pair" demo of RQ4 if needed |
| R-10 (kept) | Reviewer demands real-robot deployment | M | M | Cite scope as "methodological + public-dataset + Jetson Orin NX benchmark"; commit to follow-up demo paper | If desk-rejected, the Jetson numbers become the primary deployability evidence |
| **R-11 (new)** | **RTX 4060 16 GB VRAM insufficient for Cylinder3D + 20-D Dirichlet head + voxel-hash + KITTI-360 batch** | **H** | M | Batch size = 1 with gradient accumulation; freeze Cylinder3D backbone (fine-tune last layer only, see §3.5); offload submap RAM to host; checkpoint per submap | If still OOM on KITTI-360, swap Cylinder3D for SalsaNext (lighter) and re-time |
| **R-12 (new)** | **ConvBKI authors' reimpl fails to reproduce published numbers on our 4060** | M | **H** | Allocate full W2 + W3 to ConvBKI reproduction; pin CUDA + PyTorch versions per their README; reach out to authors if blocked; cross-check with published seq-08 mIoU within ±2 | If reproduction fails after 2 weeks, document the discrepancy openly and use their reported numbers as a quoted-only comparison (clearly labelled in tables) |
| **R-13 (new)** | **KITTI-360 multi-session revisit length / overlap is too short to demonstrate stale-voxel decay** | M | M | Pre-compute overlap matrix in W3; if insufficient, augment with synthetic temporal-gap injection on single sessions (delete random voxel evidence between artificial "sessions") | Demote to "synthetic-revisit study only" and shrink H4 from precision-recall to AUROC of stale-voxel ranking |
| **R-14 (new)** | **EDL Dirichlet head training is unstable / collapses to uniform Dirichlet** | M | **H** | Use Sensoy 2018 KL-annealing schedule; gradient-clip; monitor evidence sum trajectory; warm-start from softmax pretrain | If unstable after 1 week of tuning, switch to posterior network (PostNet, Charpentier 2020) which has the same evidential semantics but more stable training |

**H/M/L summary (v2):** **H: 3** (R-4, R-11, R-12 — algorithmic-fail + hardware-fail + baseline-fail are the three structural risks); **M: 8** (R-2, R-3, R-6, R-7, R-9, R-10, R-13, R-14); **L: 1** (R-8). Net: same total risk count as v1 (1 retired R-1/R-5 pair, two new R-11/R-12/R-13/R-14 quartet, with R-14 absorbing the R-1 evidential-training fragility).

(v1 risks R-1 (K-Radar pseudo-GT) and R-5 (4D-radar head) are **retired** — see §0.5. R-2/R-3/R-4/R-6/R-7/R-8/R-9/R-10 carried over with minimal edits.)

---

## §7 Timeline (≈ 36 weeks, IROS 2027 deadline ≈ 2027-03)

Reference today = 2026-05-28; IROS 2027 paper deadline ≈ 2027-03 (one-month buffer for submission cycle); usable working horizon ≈ 36 weeks.

| Phase | Weeks | Calendar (approx.) | Focus | Concrete output / milestone |
|-------|-------|---------------------|-------|-----------------------------|
| **P1 Bring-up** | W1-W3 | 2026-06 → 2026-06-mid | R2-reimpl + Cylinder3D fine-tune + KITTI-360 revisit-overlap matrix | W1: R2-reimpl runs; W2: Cylinder3D EDL last-layer fine-tune complete on 4060; **W3: KITTI-360 revisit matrix delivered (R-13 check); first Go/No-Go (§9 G-1, G-3)** |
| **P2 Algorithm** | W4-W6 | 2026-07 → 2026-07-mid | M1 evidential head integrated; A-1 ablation skeleton | W4: M1 working end-to-end on seq 08; **W4 second Go/No-Go (§9 G-2 evidential head ≥ +1 mIoU)**; W5: ConvBKI reimpl validated (R-12 check); W6: A-1 numbers in hand |
| **P3 Systems** | W7-W12 | 2026-08 → 2026-09 | M2 loop+fusion; M3 decay; A-2 / A-3 / A-5 ablations | W7-W8: M2 single-session loop closure on SemanticKITTI; W9-W10: M3 decay integrated; W11-W12: A-2 + A-3 + A-5 numbers; **W12 third Go/No-Go (§9 G-4 M2 loop precision ≥ 70 %)** |
| **P4 Robustness** | W13-W18 | 2026-10 → 2026-11 | Robo3D corruption sweep + RQ2 numbers + open-set backup if needed | W13-W15: Robo3D pipeline reimpl + run; W16-W17: RQ2 H2 numbers; W18: open-set backup eval if RQ2 H2 not green; **W18 fourth Go/No-Go (§9 G-5)** |
| **P5 Lifelong + Downstream** | W19-W24 | 2026-12 → 2027-01 | KITTI-360 multi-session RQ4; SemanticSpray RQ3; Jetson Orin NX deploy | W19-W22: RQ4 H4 numbers on KITTI-360; W23: RQ3 H3 numbers on SemanticSpray; W24: Jetson Orin NX latency + memory benchmark |
| **P6 Writing** | W25-W30 | 2027-01 → 2027-02-mid | `/ars-full` first draft, figures, tables, supplementary | W25-W28: draft 1 from outline_v2; W29: internal reviewer-sim (`academic-paper-reviewer`); W30: revision pass |
| **P7 Submission** | W31-W36 | 2027-02-mid → 2027-03 | Reviewer-sim second pass + format-convert + camera-ready prep + buffer | W31-W33: second reviewer-sim + revision; W34: format-convert to IROS LaTeX; W35-W36: buffer / camera-ready / submission |

**First milestone (W1 deliverable):** R2-reimpl baseline running end-to-end on SemanticKITTI seq 08, producing at least a (possibly poor) mIoU number, by end of **W1 = 2026-06-04**. This is the earliest Go/No-Go warning indicator: if even the baseline cannot stand up in W1, the entire schedule slips.

---

## §8 Open Questions for the User (v2, 3 questions)

v1's five questions were all radar / K-Radar / RA-L / FusionPortable / annotation — all rendered moot by the v2 reset. Three new questions:

1. **RQ2 choice confirmation.** §2 recommends **Robo3D corruption robustness** for RQ2; alternative is open-set vacuity. Recommended for reasons in §2 (established benchmark, zero annotation, deepens §1.4 thesis). **Do you accept the Robo3D recommendation, or do you prefer the open-set formulation?** (Both are technically defensible; user judgement decides which story is closer to your taste.)

2. **Khronos head-to-head scope.** Khronos [A2] is the most-feared reviewer threat. v2 plans a **closed-set RQ1-only** comparison (we do not try to beat its dynamic-scene 4D factorisation). **Are you comfortable explicitly declaring orthogonality** ("Khronos factorises time, we factorise uncertainty") in the related work and discussion, **or do you want us to attempt a dynamic-scene comparison** (which would cost ~3 more weeks on a 4060)?

3. **Jetson Orin NX deployment depth.** v2 budgets a single deployment **benchmark** (latency + memory + ≥ 5 Hz target on Orin NX). **Do you want us to additionally collect a short real-time demo video** (e.g., live-replay of SemanticKITTI seq 08 streaming into the Jetson, ~30 s of online mapping)? This is "free" engineering-wise (the system already runs) but consumes ~2 extra calendar days for capture + edit. If yes, slot in W24.

---

## §9 Go / No-Go Criteria (v2, 5 conditions)

All five must be green to proceed past P3 (Systems phase end, W12); any red triggers documented fallbacks. Each is checkpointed at the date shown in §7.

1. **G-1 — Baselines reproducible (W3).** R2-reimpl produces an end-to-end mIoU on SemanticKITTI seq 08 by end of W3, within ±2 mIoU of published Cylinder3D numbers (since R2 itself reports no mIoU, Cylinder3D is the indirect target). If not, slip the schedule one week and re-baseline; if still failing W4, switch backbone to RangeNet++ (BSD-licensed, simpler to reimpl).

2. **G-2 — Evidential head working (W4).** M1 Dirichlet evidence module shows **≥ +1 mIoU** and reduced ECE on at least SemanticKITTI seq 08 by end of W4 (sanity check that Eqs. M1.1-M1.3 are correctly implemented). If **< 0 mIoU delta**, retreat to §0 fallback (calibration-only RA-L paper).

3. **G-3 — KITTI-360 revisit usable (W3).** Pre-computed odometry-overlap matrix delivers ≥ 5 revisit pairs with ≥ 30 m sustained overlap by end of W3. If < 5, augment with synthetic-revisit injection (R-13 mitigation) and document as a scope adjustment in the final paper; if even synthetic injection fails to produce a useful lifelong story by W18, demote RQ4 to a single-session ECE-drift micro-study.

4. **G-4 — M2 loop closure precision (W12).** Confidence-aware loop closure achieves **≥ 70 % precision at 50 % recall** on SemanticKITTI seq 08 self-loop pairs by end of W12. If not, drop the entropy-channel descriptor variant (A-3) from main and use h_class only; the paper still ships but C2 systems claim shrinks.

5. **G-5 — RQ2 corruption story green by W18.** Either Robo3D RQ2 H2 numbers hit the targets (mean mIoU drop ≤ 50 % of R2-reimpl, vacuity AUROC ≥ 0.80) by end of W18, **or** the open-set vacuity backup hits AUROC ≥ 0.80 by end of W18. If neither, RQ2 is demoted to an honest negative result section (still publishable — calibration on clean data plus negative on corruption is fine for IROS), and C1's "one vacuity, three jobs" framing tightens to "one vacuity, two jobs" (open-set OOD + decay only).

If any of G-2 or G-4 reds, the **§0 fallback to a calibration-only RA-L paper** engages. G-1 / G-3 / G-5 reds shrink scope but do not abort.

---

*End of Research Plan v2. Companion: `paper_outline_v2.md` in the same directory. v1 frozen at `research_plan_v1.md` and `paper_outline_v1.md`.*
