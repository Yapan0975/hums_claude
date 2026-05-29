---
version: v3 (2026-05-28) — patches: RQ2→open-set, +Khronos dynamic comparison, +Jetson demo video
v2 archived as: research_plan_v2.md (same directory)
v1 archived as: research_plan_v1.md (same directory)
v3 axis (unchanged from v2): EvidLife-Map  =  candidate X (EvidVox)  +  candidate Y (Lifelong + Map Decay)
v3 venue: IROS 2027 (deadline ≈ 2027-03; ≈ 9 months horizon)
v3 compute: 1× RTX 4060 (16 GB) + 1× Jetson Orin NX (deployment benchmark + 30 s demo video)
v3 datasets: SemanticKITTI / nuScenes-LiDARSeg / KITTI-360 / SemanticKITTI dynamic split / SemanticSpray (+ Robo3D as fallback for RQ2)
---

# Research Plan (v3)

**Working title (primary):** *EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay and Open-Set Vacuity*

**Tagline (one line):** A LiDAR-only online metric-semantic mapping system that (i) replaces the unspecified per-voxel Bayes filter of R2-class systems with a closed-form Dirichlet-evidential posterior carrying explicit unknown mass, (ii) maintains the map across sessions through confidence-aware loop closure and voxel-hash submap fusion, and (iii) ages stale voxels via a vacuity-driven decay rule so the same representation answers three reviewer-credible questions — calibration, open-set, lifelong — without inflating the modality stack.

**Author / lead:** (TBD by user)
**Target venue:** IROS 2027 (8 pages double-column, conference systems track).
**Prepared:** 2026-05-28
**Stage:** Stage-1 architect output (v3 patch) of the `academic-pipeline` skill; precedes any `/ars-full` execution.
**Upstream inputs:**
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\r2_audit.md` (R2 6-dim technical audit)
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md` (54-entry literature scan)
- `D:\_7_sci\semantic_mapping\_Online_Metric_Semantic_Mapping_for_Autonomous.txt` (R2 cleaned full text)
- `D:\_7_sci\semantic_mapping\_new_paper\drafts\research_plan_v2.md` (frozen v2, for diff)
- `D:\_7_sci\semantic_mapping\_new_paper\drafts\research_plan_v1.md` (frozen v1, for archaeology)

---

## §0 Direction Decision — v2 reset retained in v3 (mandatory before §1)

v1 selected candidate **W = EviRad-Map** (LiDAR + 4D-radar evidential fusion) as the primary axis, with candidate **X = EvidVox** (LiDAR-only evidential) as fallback. The v1 §9 Go/No-Go listed **G-3** ("K-Radar pseudo-GT validated by 50-frame manual sparse-GT sanity sample ≥ 75 % agreement, by end of Week 3") as the controlling gate.

**v2 trigger.** The user's resource brief (2026-05-28) sets the manual K-Radar voxel annotation budget to **zero person-days**. With no manual sparse-GT, G-3 cannot be evaluated at all; the entire pseudo-GT defence collapses to "trust a pretrained Cylinder3D on out-of-domain frames", which is precisely the audit weakness (E1/E3) we promised to fix. v1 §6 R-1 therefore deterministically fires and v1 §0 fallback engages.

**v2 primary axis (retained in v3):** **EvidLife-Map** = candidate X (EvidVox) **combined with** candidate Y (Lifelong + Loop + Map Decay from v1 §0 Top-2 axis). This restores the "two strong contributions" balance that pure X alone would not carry — a system paper needs both an algorithmic and a systems contribution to fill 8 IROS pages credibly without radar.

| Axis | X. Evidential Voxel Fusion (R2 Top-1) | Y. Lifelong + Loop + Decay (R2 Top-2) | **X+Y. EvidLife-Map (v3)** |
|---|---|---|---|
| Novelty | 3 — crowded by ConvBKI [C9] / LatentBKI [D10] | 2 — Khronos [A2] owns dynamic-scene 4D MSM | **4** — Dirichlet-vacuity-as-decay-signal is the bridge nobody yet ships in one system |
| Venue-fit IROS 8 p | 3 — feels like a method paper, hard to fill 8 p | 4 — IROS systems track loves it | **5** — algorithm + system, fits IROS double column comfortably |
| R2 audit delta covered | S1/S2/S5 | T1/T2/T3 + T4 | **S1/S2/S4/S5 + T1/T2/T3/T4** (8 of 30 audit findings closed) |
| SOTA differentiation | Must beat ConvBKI in mIoU | Uphill vs Khronos | **v3: dynamic-scene head-to-head vs Khronos accepted**; we fight on **calibration + lifelong + dynamic** axis, where Khronos/Clio/ConvBKI each cover only one |
| Data availability (no manual) | 5 — SemanticKITTI / nuScenes-LiDARSeg dense GT | 4 — KITTI-360 has multi-session revisits | **5** — every dataset is fully public and pre-labelled |
| Engineering effort on 1× 4060 | 2 — Dirichlet head + Bayes update | 4 — submap + loop + decay | **3** — large; v3 added Khronos dynamic comparison consumes ~3 extra calendar weeks; ablation grid further compressed (§4.4) |
| Reviewer risk | Low | Med (Khronos head-to-head) | **Low-Med** — v3 accepts the dynamic-scene head-to-head rather than declaring orthogonality only; this defangs the most likely reviewer attack |
| Compute fit RTX 4060 | 5 — fits | 4 — long-sequence eval tolerable | **3** — tight; A-6 dropped first, A-7 promoted to firm because open-set is now RQ2 lead |

**Decision (v3 primary, unchanged from v2):** **EvidLife-Map** — three integrated modules (M1 Dirichlet-evidential per-voxel posterior; M2 confidence-aware loop closure + multi-session voxel-hash submap fusion; M3 vacuity-driven voxel decay + map staleness ageing) trained and evaluated entirely on public LiDAR-only datasets, deployable on Jetson Orin NX with a published 30-second demo video.

**Fallback (single layer only, no further nesting).** If by Week 4 the Dirichlet evidential head fails to beat the argmax-Bayes baseline by ≥ +1 mIoU on SemanticKITTI seq 08, retreat to a **calibration-only paper**: drop M2 and M3 to brief sub-sections, keep M1 as the headline, target **RA-L** instead of IROS (4-page brevity hides the missing systems story). This is documented in §9 G-2.

---

## §0.5 v1 → v2 Change Log (preserved unchanged)

| Dim | v1 | v2 | Reason |
|---|---|---|---|
| **Primary axis** | **EviRad-Map** (LiDAR + 4D-radar evidential fusion) | **EvidLife-Map** (LiDAR-only evidential + lifelong + decay) | User: zero person-days for K-Radar manual annotation → v1 G-3 unverifiable → v1 fallback engaged |
| **Title** | EviRad-Map: Evidential LiDAR + 4D-Radar Online MSM for Weather-Robust Robot Navigation | EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay | follows axis |
| **Datasets** | SemanticKITTI / nuScenes-LiDARSeg / **K-Radar** (+ secondary ACDC, SemanticSpray, Boreas reserve) | SemanticKITTI / nuScenes-LiDARSeg / **KITTI-360** / **Robo3D-corrupted SemanticKITTI** / SemanticSpray | K-Radar dropped (no manual GT); KITTI-360 added for lifelong revisits; Robo3D added for corruption robustness (replaces weather); ACDC dropped (image-only, RGB head no longer central); FusionPortable explicitly excluded (user: clean public only) |
| **RQ2** | "Does 4D-radar fusion close the weather gap?" | "Does vacuity detect open-set / corruption-unknown voxels reliably under Robo3D-style point-cloud corruptions?" | Radar branch gone; need a defensible RQ2 of equal ambition that runs on a 4060 |
| **RQ4** | Lifelong as appendix-grade stretch | Lifelong **promoted to first-class RQ4 in main paper** | IROS 8 p can carry it; v1 RA-L tightness no longer applies |
| **Modules** | 4 modules (LiDAR head, radar head, evidential fusion, trust schedule, decay, loop) | 3 modules: M1 evidential, M2 loop+submap-fusion, M3 vacuity-decay | leaner, no radar, no trust schedule |
| **Baselines** | R2-reimpl + NvBlox-vanilla + ConvBKI + OpenVox + Khronos + L4DR stretch | R2-reimpl + NvBlox-vanilla + ConvBKI + **Kimera-Semantics** + **Khronos** (closed-set RQ1 only) + **Clio-LiDAR-stub** (open-set RQ2 only) | Removed L4DR (no radar); added Kimera-Semantics as canonical Voxblox-class baseline; added Clio LiDAR stub for open-set comparison |
| **Ablation matrix** | 7 ablation variables (A-1…A-7) on 4090 | **5-7 ablation variables** but each on 2 sequences only on 4060; A-2 (radar ±) removed; A-3 (trust ±) removed; new A-1' (decay schedule shape), A-2' (loop descriptor) added | 4060 wall-clock ≈ 2.5× 4090 per epoch; ablation grid had to shrink |
| **Compute** | 1× **RTX 4090 (24 GB)** + Jetson Orin NX deployment check | 1× **RTX 4060 (16 GB)** + Jetson Orin NX deployment check | User hardware reality; halves batch sizes; doubles training wall clock |
| **Venue** | RA-L primary (4 p) + IROS fallback | **IROS 2027 primary** (8 p) directly; RA-L only as fallback if M2/M3 fail (§9 G-2) | User preference for IROS 8 p over RA-L 4 p; ample room for lifelong story |
| **Timeline** | 10 weeks to RA-L | ~36 weeks (≈ 9 months) to IROS 2027 deadline ≈ 2027-03 | Real horizon; allows 4060 slow training plus reviewer-sim cycle |
| **Risk register** | R-1 (K-Radar GT), R-5 (radar head) high | R-1/R-5 **retired**; new R-11 (4060 VRAM), R-12 (ConvBKI reimpl), R-13 (KITTI-360 revisit length), R-14 (evidential head instability) | Risks remapped to actual stack |

---

## §0.6 v2 → v3 Change Log (new in v3)

| Dim | v2 | v3 | Reason |
|---|---|---|---|
| **RQ2 lead** | **Robo3D corruption robustness** (recommended) with open-set as backup | **Open-set vacuity** (SemanticKITTI known/unknown split) as primary; Robo3D demoted to fallback | User decision 2026-05-28: prefers open-set story over corruption; ties more tightly to §1.4 "one vacuity, three jobs" thesis; AUROC-of-vacuity for OOD detection is the canonical EDL evaluation |
| **Khronos comparison** | Closed-set RQ1-only; declared *orthogonality* in related work | **Adds dynamic-scene experimental head-to-head** on SemanticKITTI dynamic split (+3 weeks of work) | User decision 2026-05-28: text-only orthogonality declaration is reviewer-vulnerable; actual numbers on dynamic frames forecloses the most likely attack |
| **Jetson Orin NX** | Benchmark table only (latency + memory) | Benchmark table **plus 30-second demo video** in supplementary (+2 days) | User decision 2026-05-28: demo video makes C4 deployability claim visceral; "free" given the system already runs |
| **Timeline** | 36 weeks; P5 = W19-W24 | **39 weeks**; P5 extended to W18-W26 to absorb Khronos dynamic experiment; P6/P7 push back accordingly; W24 + W25 get the demo capture and edit | The +3 weeks from Khronos dynamic + 2 days from Jetson video reduce the W34-W36 buffer from 3 weeks down to ~1 week |
| **Ablations** | A-7 (open-set channel) listed as kept-if-time | A-7 **promoted to firm**, A-6 (voxel size) becomes the drop-first stretch | RQ2 lead is now open-set, so A-7 directly answers the lead RQ |
| **Risk register** | R-15 not present | **R-15** (open-set split definition controversial) and **R-16** (SemanticKITTI lacks taxonomic diversity for "unknown") added; **R-17** (Khronos reproduction failure for dynamic comparison) added; Robo3D-specific risk subsumed into fallback path | Open-set evaluation has its own methodological controversies; Khronos reproduction is now a load-bearing experiment |
| **Method §3** | M1/M2/M3 only | Adds **§3.7 Dynamic-Object Handling Discussion** before §3.8 deltas | Required to set up the Khronos dynamic head-to-head as a fair comparison rather than apples-to-oranges |
| **Go/No-Go** | 5 gates | **6 gates** — adds G-6 (Khronos baseline ready by W20) | Khronos reproduction is now a load-bearing experiment; needs an explicit checkpoint |

---

## §1 Problem Statement & Motivation (v3)

### 1.1 Scenario
A ground robot operates outdoors over weeks of repeated traversal of the same area (campus, urban block, industrial site). It must build, online and on-board (Jetson-class GPU), a dense voxel map labelled with closed-set semantics plus an explicit "unknown" channel, and must keep that map correct across (i) **single-session noise** (LiDAR drop-out, sensor occlusion), (ii) **open-set encounters** (object categories absent from the training taxonomy — construction debris, exotic vegetation, unmodelled vehicle types), and (iii) **inter-session change** (revisits weeks later with moved objects, construction, new vegetation). Camera and radar are *not* assumed; the system is LiDAR-only with optional RGB-derived priors at training time only.

### 1.2 R2 baseline summary and the four reviewer-credible weaknesses
R2 [A1] (Jiao et al., HKUST) couples LVIO with an NvBlox TSDF backbone and a confidence-aware HRNet [P1] segmenter, fused per-voxel through an **unspecified** iterative Bayes filter. The R2 audit, distilled to the four cripple-points an even mildly-attentive RA-L/IROS reviewer will raise:

1. **Methodological under-specification.** R2's Bayes update formula, prior, and confidence-into-fusion pathway are all missing (audit S1/S2/S4). Reproducible? No.
2. **No quantitative evidence.** Zero ablation, zero baseline number, zero mIoU, zero F-score; evaluation only on two self-recorded sequences (audit E1/E2/E3/E5).
3. **No open-set / OOD handling.** "Unlabeled = untraversable" (R2 p.3, audit S5); pessimistic and unusable for active exploration in environments where genuinely-novel categories appear (the open-set / OOD-detection-in-mapping gap, v3 RQ2 lead).
4. **No long-term / loop-closure / decay mechanism.** Lifelong drift, stale voxels, re-visits all unaddressed (audit T1/T2/T3/T4).

v1 sought to fix (1)+(3) algorithmically and add a fifth axis (multi-modal weather robustness). v3 (= v2 retained) drops the modality axis and instead pairs (1)+(3) algorithmic fix with a (4) systems fix — which is exactly the audit's largest cluster of unfilled weaknesses, and is achievable on a 4060.

### 1.3 Differentiation vs current SOTA (v3 re-framed)
The literature scan flags three SOTA threats; v3 declares its relationship to each:
- **Khronos [A2/D1]** factorises *short-term motion vs long-term change in time*; v3 factorises *evidence strength vs vacuity in observation* and uses time only as a decay scalar in M3. The decay rule (Eq. M3.1) is dual to Khronos's "long-term change detector" — it requires no explicit change-detection network, only a Dirichlet-conjugate exponential. **v3 escalation:** rather than only declaring this orthogonality in prose, we run a head-to-head experiment on a SemanticKITTI dynamic split (§3.7, §4.2 baseline, §4.4 Table V) so reviewers see numbers, not adjectives.
- **Clio [A3/D2]** compresses *given a task*; v3 produces a *task-agnostic, uncertainty-aware substrate* that Clio could be re-implemented on top of. The Clio LiDAR stub (one of our open-set baselines, §4.2) is the explicit comparison.
- **ConvBKI [C9] / LatentBKI [D10]** use kernel Bayesian inference for *spatial smoothing*; v3 uses **EDL Dirichlet** for closed-form per-voxel epistemic uncertainty *without* spatial-kernel smoothing — which yields a sharper, more honest vacuity signal for the M3 decay rule and for the open-set OOD-detection task (RQ2 lead in v3), at the price of giving up smoothing's mIoU bump (which the M2 loop closure re-projection partially restores).

The novelty story for v3 is **not** "first to fuse modality X+Y" (v1 story); it is **first to wire one statistical quantity (Dirichlet vacuity) into three jobs simultaneously**: (a) **open-set / OOD detection** (v3 RQ2 lead), (b) loop-closure descriptor entropy term, (c) lifelong decay trigger. That single quantity threading three modules is what reviewers will remember.

### 1.4 Thesis statement (v3)
> *A metric-semantic voxel map becomes simultaneously more calibrated, more open-set-aware, and more lifelong-maintainable when its per-voxel posterior is a closed-form Dirichlet-evidential distribution whose vacuity mass is reused as (i) the **open-set / OOD score for unknown-category voxels**, (ii) the loop-closure descriptor's entropy channel, and (iii) the conjugate decay trigger that ages stale voxels — eliminating the three independent ad-hoc heuristics that prior systems (R2, ConvBKI, Khronos) use one each.*

---

## §2 Research Questions & Hypotheses (v3)

### RQ1 (preserved from v1/v2) — Does Dirichlet-evidential per-voxel fusion outperform argmax-Bayes / S-BKI-style kernel Bayesian inference on closed-set dense semantic mapping?
- **H1.** On SemanticKITTI sequences 08, 11-21, an EDL-Dirichlet voxel update yields **≥ +2 mIoU** and **≥ −20 % ECE** versus an R2-reimplemented argmax-Bayes baseline at equal voxel size (0.25 m) and equal compute. (mIoU bar lowered from v1's +3 because we no longer have radar to push the upper end; ECE bar held.)
- **Boundary.** Holds for closed-set 19-class setting; for open-set we expect ECE win to grow and mIoU possibly to shrink because "unknown" steals mass from rare classes — quantified in RQ2.

### RQ2 (v3 — open-set vacuity as lead; Robo3D corruption as fallback)

**v3 primary formulation (open-set vacuity):**
> *Does the per-voxel Dirichlet vacuity score reliably separate voxels whose observed point evidence comes from a held-out unknown category from voxels of known categories, while the closed-set mIoU on the remaining known categories stays within striking distance of a fully-supervised baseline?*

- **Open-set split protocol.** We adopt **SemanticKITTI 19-class as the supervision label set**, with **N classes withheld as "unknown" at training time** (provisional choice: N = 5, withholding `bicyclist`, `motorcyclist`, `truck`, `other-vehicle`, `other-ground` — a mix of dynamic-rare and static-rare classes that span the LiDAR appearance distribution rather than clustering at one end). The remaining 14 classes form the closed-set supervision space. At evaluation, points belonging to a withheld class are treated as "unknown" GT; the model must assign them high vacuity. Pre-registered in supplementary; alternative N = 3 (drop only `bicyclist`, `motorcyclist`, `other-vehicle`) reported as robustness check.
- **Reverse cross-domain check (nuScenes-LiDARSeg).** Train on SemanticKITTI 14-known split; evaluate vacuity-as-OOD on nuScenes-LiDARSeg classes that do not overlap with the SemanticKITTI training set (e.g., `construction_vehicle`, `barrier`, `traffic_cone`). Provides a cross-dataset OOD cross-check that does not depend on our own split choice.
- **H2 (v3 primary).** On SemanticKITTI 14-known / 5-unknown split, EvidLife-Map's vacuity achieves **voxel-level AUROC ≥ 0.80** and **AUPR ≥ 0.60** for known-vs-unknown discrimination (averaged over withheld classes), while **closed-set mIoU on the 14 known classes drops by ≤ 1.5 mIoU** versus the 19-class supervised baseline. On the nuScenes reverse cross-domain check, AUROC ≥ 0.75 (lower bar because of domain shift).
- **Boundary.** Open-set evaluation depends on the known/unknown split choice; we mitigate by reporting both N = 5 and N = 3 splits and by adding the independent nuScenes reverse-check. We do not claim taxonomic universality; the claim is "vacuity is a usable OOD score on these splits".

**Why open-set as RQ2 lead (user 2026-05-28 decision).**
1. Open-set / OOD detection directly instantiates the §1.4 "one vacuity, three jobs" thesis without an extra detour through corruption labels.
2. AUROC-of-vacuity for OOD is the canonical EDL evaluation pattern (Sensoy et al., NeurIPS-18); a paper that claims EDL-on-voxels and *doesn't* report it would invite a reviewer "why not".
3. Avoids the synthetic-corruption-vs-real-weather trap — open-set is meaningfully real even on clean data.
4. Sets up a clean ablation A-7 (± dedicated unknown channel) that is now firm in v3 §4.4.

**Fallback formulation (Robo3D corruption robustness), kept on standby if open-set split is blocked at W5:**
> *Does the Dirichlet vacuity score degrade gracefully across the 8-corruption × 5-severity Robo3D-SemanticKITTI grid where ConvBKI's kernel-smoothed confidence stays artificially high?* — H2-fallback: mean mIoU drop ≤ 50 % of R2-reimpl; vacuity-as-corruption-detector AUROC ≥ 0.80. Identical metric family to H2 (AUROC-based), so analysis pipeline is reusable.

**Decision rule.** Go with **open-set vacuity** (H2 primary above). If the open-set split protocol is contested by W5 internal review (e.g., the N = 5 split turns out trivially separable because all 5 withheld classes look obviously different in LiDAR — which §6 R-16 calls out), switch to Robo3D corruption fallback within the same week without timeline slip.

### RQ3 (kept conceptually from v1, descoped in v2, unchanged in v3) — Does propagating evidential uncertainty into downstream traversability improve navigation safety on public closed-loop replay scenarios?
- **H3.** On SemanticSpray-style passive-replay traversability evaluation, an uncertainty-gated traversability head produces **≥ +10 pp safe-region recall** and **≥ −30 % false-traversable rate** versus an R2-style hard-rule baseline, at equal coverage.
- **Boundary.** Passive replay (no actuated navigation) — we explicitly do not claim closed-loop navigation success in v3.

### RQ4 (promoted from v1 appendix to main in v2, unchanged in v3) — How does the evidential map scale and degrade gracefully across multi-session revisits?
- **H4.** Across **KITTI-360 sequences 00, 02, 04, 05, 06, 07, 09, 10** (≥ 5 revisit pairs constructed from spatial overlap), the M2 confidence-aware loop closure + M3 voxel-hash submap fusion achieves: (i) **stale-voxel removal precision ≥ 80 %** over the second-pass trajectory after evidence decay (M3.1), (ii) **inter-session map ECE drift ≤ 1.5×** the single-session ECE, (iii) **multi-session memory footprint ≤ 1.7×** the single-session footprint (sub-linear in session count thanks to voxel-hash deduplication). All three measured on the KITTI-360 evaluation split.
- **Boundary.** Single-robot multi-session only; not claimed for collaborative multi-robot map merging (Hydra-Multi [A5] territory). Revisit pairs constructed by us via odometry-overlap (documented in §4.3); reviewers can audit the overlap script.

### RQ5 (new in v3 — dynamic-scene head-to-head vs Khronos)
> *On a SemanticKITTI dynamic-object-heavy subset, does EvidLife-Map — without an explicit 4D motion model — achieve dynamic-object mIoU and static-recall within a defensible margin of Khronos's explicit short-/long-term factorisation, by virtue of vacuity naturally flagging unstable evidence on moving objects?*

- **H5.** On SemanticKITTI sequences with high fraction of dynamic-object frames (provisionally: portions of seq 00, 04, 05, 07 marked dynamic via the SemanticKITTI dynamic-label flag), EvidLife-Map achieves **dynamic-object mIoU within 3 mIoU of Khronos** and **static-region recall ≥ Khronos − 1 pp**, while **latency stays at ≥ 5 Hz on Jetson Orin NX** (Khronos is not designed for Orin NX deployment). The framing: we cede dynamic-scene mIoU supremacy but win on integrated calibration + latency.
- **Boundary.** Not a claim that vacuity replaces explicit 4D motion modelling; the claim is that for online MSM with deployment constraints, the vacuity-as-instability-detector is competitive enough to make Khronos's full short-/long-term machinery a higher-cost choice in our operating regime. If Khronos reproduction fails (§6 R-17), fall back to published-numbers comparison with an honest disclaimer.

---

## §3 Technical Approach (v3)

### 3.1 System architecture (ASCII, v3 three modules — unchanged structurally from v2)

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
              |  SemanticKITTI; closed-set 14   |
              |  + 1 "unknown" channel = 15-D   |
              |  Dirichlet evidence vector e_L  |
              |  under v3 open-set RQ2 split;   |
              |  19+1 = 20-D under closed-set   |
              |  RQ1 / RQ4 / RQ5)               |
              +---------------+-----------------+
                              |
                              v
   +--------------------------+------------------------+
   | §3.2  M1  EVIDENTIAL PER-VOXEL POSTERIOR          |
   | per-voxel alpha_v in R^{C+1}                      |
   | vacuity m_u(v) = (C+1) / sum(alpha_v)             |
   | Eqs. M1.1-M1.3 (closed form, no kernel smoothing) |
   | NOTE: vacuity acts as natural OOD score (RQ2)     |
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

### 3.2 M1 — Evidential per-voxel posterior (inherits v2 M1)
Replace R2's unspecified Bayes filter (audit S1/S2) by a Dirichlet-evidential posterior. Per voxel `v`, maintain accumulated evidence `α_v ∈ R^{C+1}` where C = 19 SemanticKITTI classes (closed-set RQ1/RQ4/RQ5) or C = 14 (open-set RQ2 lead) and the last channel is the open-set "unknown" channel. **Vacuity `m_u(v) = (C+1) / Σ α_v` acts as a natural OOD score** (the v3 RQ2 lead consumes this directly). New observations contribute evidence `e_v(z) = softplus(logit(z))` (no per-modality trust schedule; v3 has only one modality). Eqs.:

- **(M1.1)** Evidence accumulation: `α_v^{t+1} = α_v^{t} + e_v^{t+1}`.
- **(M1.2)** Posterior class mean: `E[p_c | α_v] = α_{v,c} / Σ_k α_{v,k}`.
- **(M1.3)** Vacuity (open-set / OOD score, reused in M2 and M3): `m_u(v) = (C+1) / Σ_k α_{v,k}`.

EDL Dirichlet (Sensoy et al., NeurIPS-18) rather than S-BKI / ConvBKI kernel-Bayesian inference, because (i) we want per-voxel epistemic uncertainty *without* the spatial-kernel coupling that makes vacuity contaminated by neighbour evidence — the M3 decay rule and the RQ2 OOD score both need an honest per-voxel vacuity, and (ii) closed-form posterior keeps a 4060 viable.

### 3.3 M2 — Confidence-aware loop closure + multi-session voxel-hash submap fusion (unchanged from v2)
Submaps sealed every 50 m or 30 s of trajectory. Per-submap descriptor:
- **(M2.1)** `d(S) = ( h_class(S),  h_entropy(S) )` where `h_class(S)` is the L1-normalised class histogram over confident voxels (vacuity below median), and `h_entropy(S)` is the histogram of per-voxel posterior entropy bucketed to 10 bins. Matching via cosine on `h_class` + Earth-Mover-Distance on `h_entropy`; verified by semantic-ICP restricted to class-consistent confident voxels.

Once a loop is verified, pose graph optimisation re-projects each voxel's `α_v` consistently across the loop; in submap overlap regions, `α_v` from different sessions are merged by **conjugate addition** `α_v ← α_v^{(s1)} + α_v^{(s2)}` (closed-form Dirichlet posterior over independent observations), so fusion is parameter-free.

Inter-session storage: voxel-hash key = `(int(x/v), int(y/v), int(z/v))`, value = `α_v`. New sessions write into the same hash; deduplication is implicit.

Inspired by: Kimera-Multi [C7] (pose graph + multi-robot), Hydra [A4] (per-submap descriptor), SA-LOAM [B4.4] (semantic-aided LCD), PlaneSDF [B4.2] (cross-session change detection). Distinct from all four because the *descriptor entropy channel* and the *parameter-free Dirichlet conjugate fusion* are both new.

### 3.4 M3 — Vacuity-driven voxel decay + map staleness ageing (unchanged from v2)
Conjugate exponential decay on `α_v`, but with the decay time-constant `τ` itself a function of vacuity:

- **(M3.1)** `α_v^{t+Δ} = ((α_v^{t} − 1) · exp(−Δ / τ(m_u))) + 1`, where `τ(m_u) = τ_max · (1 − m_u) + τ_min · m_u`, with `τ_max ≈ 3600 s` (one hour) for confidently-known voxels (low vacuity, ages slowly = persistent map) and `τ_min ≈ 60 s` (one minute) for high-vacuity voxels (ages fast = transient / unknown / dynamic stuff). This single equation is the v3 thesis (§1.4) made concrete: vacuity *is* the staleness signal.

The decay preserves the posterior mean (M1.2) while inflating vacuity over time, so stale voxels become re-writable as new evidence arrives. Submap-level ageing: a whole submap whose median vacuity exceeds a threshold is flagged for re-observation by an exploration policy (out of scope for this paper but the hook is present).

Distinct from v1 §3.4 (which used a fixed `τ`) and from Voxblox / NvBlox (which use monotonically-growing weight with no decay at all, audit T2).

### 3.5 Implementation Details (terse, IROS budget)
- LiDAR semantic head: Cylinder3D, pretrained weights from authors' release; we fine-tune only the last layer to a (C+1)-D output (19+1 = 20-D under closed-set RQ1/RQ4/RQ5; 14+1 = 15-D under open-set RQ2) using EDL loss (Sensoy 2018) on SemanticKITTI train split. Pretrain on 4060 ≈ 2 days for the last-layer fine-tune (entire backbone frozen). **Note:** two separate fine-tunes are needed — one 20-D head for closed-set runs, one 15-D head for open-set RQ2.
- Backbone: NvBlox [C10] forked; `α_v` (20 × float32 = 80 B, or 15 × float32 = 60 B) replaces the 19-class label probability vector. Memory per voxel grows from ~100 B (R2) to ~180 B (v3 closed-set) or ~160 B (v3 open-set); §4.3 budgets this.
- LVIO state estimator: reused unchanged (any open-source LIO; we use the FAST-LIO2 release on KITTI-360, R3LIVE on SemanticKITTI single-session, doc'd in §4.6).
- Runs target: ≥ 10 Hz on RTX 4060, ≥ 5 Hz on Jetson Orin NX. Numbers to be measured; deployment **benchmark + 30-s demo video** (v3 §4.G).

### 3.6 Key equations (closed-form, all derived in `/ars-full`)
- **Eq. M1.1** evidence accumulation
- **Eq. M1.2** posterior class mean
- **Eq. M1.3** vacuity (open-set / OOD score, used directly by RQ2)
- **Eq. M2.1** submap descriptor (class histogram + entropy histogram)
- **Eq. M3.1** vacuity-conditioned conjugate decay
Five equations total; v1/v2 had five too. Same equation count, different semantics.

### 3.7 Dynamic-Object Handling Discussion (NEW in v3 — sets up the Khronos comparison)
EvidLife-Map does **not** explicitly model 4D space-time as Khronos [A2] does. Khronos's contribution is a learned short-term motion segmenter combined with a long-term change detector; together they let it factorise dynamic objects from a persistent map.

v3 argues that **vacuity naturally degrades on moving-object voxels** for two structural reasons. (a) A moving object presents inconsistent evidence to the same voxel across consecutive frames (now-occupied, now-free, now-different-class), so per-voxel evidence accumulation under Eq. M1.1 collects conflicting `e_v` contributions, raising `Σ α_v` slowly relative to the inter-class disagreement — equivalently, vacuity stays elevated while the posterior class mean (M1.2) gets noisy. (b) The M3 decay rule (M3.1) under a high-vacuity regime sets `τ → τ_min ≈ 60 s`, so the dynamic voxel's evidence is actively flushed before it can crystallise into a wrong-persistent label.

The result: EvidLife-Map handles dynamic objects *implicitly* through (a) + (b), where Khronos handles them *explicitly* through a dedicated network. This is a fair-comparison framing — we are not claiming EvidLife-Map's implicit mechanism dominates Khronos's explicit mechanism on dynamic-object mIoU; we are claiming it is competitive enough to make the explicit machinery a deployment-cost trade-off rather than a capability necessity. The §4.4 Table V experiment (new in v3) measures this competitiveness directly.

This framing also addresses the "why don't you just bolt Khronos onto your system?" reviewer question: an explicit short-term motion segmenter would *override* vacuity's natural-instability signal in M3, doing the same job twice with two different parameter sets. We deliberately keep the implicit channel only, document the trade-off, and let numbers decide.

### 3.8 Methodological deltas, one line each (v3 set; renumbered from v2 §3.7)
- vs **R2 [A1]:** specifies the Bayes filter (Eqs M1.1-M1.3 closed form); adds vacuity, adds decay, adds loop+fusion; adds explicit unknown channel.
- vs **Khronos [A2]:** Khronos factorises short-/long-term in time via change-detection network; we let vacuity *be* the change signal via M3.1 (no separate network). **v3 escalation: now head-to-head experimentally on dynamic split (§4.4 Table V), not just argued in prose.**
- vs **Clio [A3]:** Clio compresses given a task; we keep a task-agnostic uncertainty-aware substrate. Clio could sit on top.
- vs **ConvBKI [C9]:** kernel-Bayesian spatial smoothing; we use EDL Dirichlet — sharper vacuity (no neighbour contamination), needed by M3.1 and by RQ2 OOD.
- vs **LatentBKI [D10]:** open-vocab BKI; we are closed-set + open-set unknown channel, so we cleanly separate "known classes" and "unknown mass" rather than embedding everything in a CLIP-feature manifold.
- vs **OpenVox [A8/D8]:** Bernoulli per-instance; we are Dirichlet over all classes including unknown, and we add lifelong M2+M3.
- vs **S-BKI [C8]:** same spirit (probabilistic semantic voxel) but with explicit vacuity for decay; no kernel smoothing.
- vs **Voxblox / Voxblox++ [C1, C3]:** Voxblox-class does TSDF only; we add semantic Dirichlet + lifelong.
- vs **Kimera-Semantics [C6]:** Kimera does scene-graph + multi-robot; we focus on per-voxel calibration + single-robot multi-session lifelong.

---

## §4 Experimental Plan (v3)

### 4.1 Datasets (final shortlist, 6 datasets — all public, all pre-labelled, zero manual budget)

| # | Dataset | Role in v3 | Why kept / why added |
|---|---------|------------|----------------------|
| D-a | **SemanticKITTI** (Behley 2019; community Cylinder3D split) | Primary closed-set dense GT for LiDAR mapping mIoU and ECE (RQ1); **base substrate for v3 open-set 14/5 split (RQ2)**; base substrate for dynamic split (RQ5) | Standard, reproducible, every cited competitor has results on it |
| D-b | **nuScenes-LiDARSeg** (Caesar 2020; LiDAR seg labels released 2021) | Cross-domain mIoU/ECE check (RQ1 generalisation); **reverse cross-domain OOD cross-check for v3 RQ2** (train on SemanticKITTI 14-class, evaluate vacuity on nuScenes-only classes) | The only mainstream AD dataset with LiDAR semantic GT *and* a different class taxonomy, perfect for the cross-dataset OOD sanity check |
| D-c | **KITTI-360** (Liao 2022; multi-session dense LiDAR semantic seg, 19 classes) | Primary lifelong / multi-session evaluation arena (RQ4) | Multi-session structure (revisits across drives), large enough for the M2 loop closure tests; has dense semantic GT |
| D-d | **SemanticKITTI dynamic split** (constructed from SemanticKITTI dynamic-label flag on seq 00, 04, 05, 07; frames with high fraction of `moving-*` class instances) | **Dynamic-scene head-to-head vs Khronos (RQ5, v3 new)** | SemanticKITTI's per-point dynamic labels (moving-car, moving-person, etc.) make this a public, reproducible, no-extra-annotation construction |
| D-e | **SemanticSpray** [B3.9] (RA-L-24 wet-road LiDAR seg) | Passive-replay traversability arena (RQ3) | Real-world LiDAR-only, no radar dependence, no manual annotation needed (authors released labels) |
| D-f | **Robo3D-corrupted SemanticKITTI** (Kong 2023, ICCV-23) | **Fallback substrate for RQ2 if open-set split is blocked at W5** (not primary in v3) | Demoted from v2 RQ2 lead; retained as backup so the pipeline switch costs zero datasets |

**Open-set split protocol (v3 RQ2 primary, new sub-section).**
- **Primary split:** SemanticKITTI 19 → 14 known + 5 unknown. Withheld classes: `bicyclist`, `motorcyclist`, `truck`, `other-vehicle`, `other-ground`. Selection rationale: (i) spans static + dynamic; (ii) spans vehicle + ground; (iii) all five have ≥ 500 GT points per validation sequence so AUROC is statistically meaningful; (iv) leaves the 14 known classes with adequate per-class population for closed-set mIoU. Listed in pre-registration table in supplementary.
- **Robustness split:** SemanticKITTI 19 → 16 known + 3 unknown (drop only `bicyclist`, `motorcyclist`, `other-vehicle`). Stricter test (fewer unknowns to find); reported alongside primary as robustness.
- **Reverse cross-domain split:** train on SemanticKITTI 14 known; evaluate vacuity on nuScenes-LiDARSeg points whose class is in {`construction_vehicle`, `barrier`, `traffic_cone`, `pushable_pullable`} — none of which have a SemanticKITTI analogue. AUROC measures cross-dataset OOD generalisation, independent of the SemanticKITTI split.
- All split definitions, evaluation scripts, and AUROC/AUPR computation code released in supplementary.

**SemanticKITTI dynamic split protocol (v3 RQ5, new sub-section).**
- **Construction:** From SemanticKITTI seq 00, 04, 05, 07, select all frames where ≥ 5 % of GT points carry a `moving-*` class label (SemanticKITTI's official dynamic-label set: `moving-car`, `moving-person`, `moving-bicyclist`, `moving-motorcyclist`, `moving-bus`, `moving-truck`, `moving-other-vehicle`, `moving-on-rails`). Provisional yield: ~1500-2000 dynamic-heavy frames across the 4 sequences (to be verified W18).
- **Metrics specific to RQ5:** dynamic-object mIoU (restricted to `moving-*` classes), static-region recall (non-moving GT correctly preserved), end-to-end latency.
- **Khronos comparison:** see §4.2 baseline #5 below; if Khronos reproduction fails, fall back to published numbers per §6 R-17.

**Datasets explicitly NOT used and why:**
- **K-Radar** — dropped (user: zero annotation budget for dense voxel pseudo-GT validation).
- **FusionPortable** — dropped (user instruction: clean public datasets only).
- **Boreas [B3.4]** — would be ideal for lifelong + multi-season but the LiDAR semantic GT is sparse; KITTI-360 covers the multi-session need with dense GT.
- **ACDC [B3.3]** — image-only adverse-condition seg; v3 has no central RGB semantic head, so ACDC contributes nothing.
- **CADC [B3.5]** — adverse-weather LiDAR detection, no dense semantic GT.
- **Replica / TUM / ScanNet** — indoor, off-topic.

### 4.2 Baselines (6 total, IROS 8 p budget — Khronos role escalated in v3)
1. **R2-reimpl.** Faithful Python/CUDA reimplementation of R2 [A1] (NvBlox + Cylinder3D + argmax-Bayes). No code released by R2, so we must build it ourselves; this is the v1 baseline-1, kept.
2. **NvBlox-vanilla + per-frame argmax.** Lower bound (no temporal fusion).
3. **ConvBKI [C9].** Authors' code; canonical probabilistic semantic voxel competitor (lit_scan reviewer-threat #3). **R-12 risks reimpl failure** — see §6.
4. **Kimera-Semantics [C6].** Authors' release; canonical Voxblox-class semantic baseline; covers the "did you cite the seminal MIT-SPARK ancestor" reviewer demand. New in v2.
5. **Khronos [A2]** — **v3 escalation: now used for both (i) RQ1 closed-set mIoU on SemanticKITTI AND (ii) RQ5 dynamic-scene head-to-head on the dynamic split.** Target source: the arXiv 2402.13817 release if available; otherwise we use published numbers for the comparison cells we cannot reproduce, with relative-improvement framing and explicit disclaimer in §V (see §6 R-17 mitigation). Khronos is the load-bearing reviewer-threat baseline in v3.
6. **Clio-LiDAR-stub** [A3] — Clio's LiDAR pathway used as the open-set RQ2 secondary baseline; if Clio's release is RGB-D-only, we approximate with the OpenScene [B1.3] LiDAR distillation.

Stretch baseline (not Go criterion): **OpenVox [A8/D8]** if their code drops by W6.

### 4.3 Metrics
| Tier | Metric | Used in RQ | Notes |
|------|--------|-----------|-------|
| Mapping | mIoU, per-class IoU, F@5 cm reconstruction, voxel coverage, GPU memory peak, end-to-end latency (mean + 99-pct) | RQ1, RQ4, RQ5 | closes audit E4/E5 |
| Uncertainty (closed-set) | Expected Calibration Error (ECE), Brier score | RQ1, RQ4 | distinguishes us from R2 and ConvBKI |
| Open-set (RQ2 primary) | **AUROC** of vacuity ranking unknown voxels above known; **AUPR** for the unknown-positive class; closed-set mIoU on the known subset | **RQ2 lead** | v3 primary metric family for the lead RQ |
| Corruption (fallback) | per-corruption mIoU drop on Robo3D-SemanticKITTI; mean Corruption Error (mCE); voxel-level AUROC of vacuity-as-corruption-detector | RQ2 fallback only | analysis pipeline overlaps with the open-set AUROC pipeline (same AUROC code) |
| Dynamic (RQ5 new) | dynamic-object mIoU (restricted to `moving-*` classes), static-region recall, end-to-end latency on Jetson Orin NX | **RQ5 (new)** | head-to-head vs Khronos |
| Traversability | safe-region recall, false-traversable rate, deferral rate | RQ3 | passive replay on SemanticSpray |
| Lifelong | stale-voxel removal precision/recall over multi-session trajectory; ECE drift across sessions; map size growth | RQ4 | KITTI-360 multi-session split |

**Pseudo-GT?** Not needed in v3. Every dataset above has dense semantic GT (D-a/D-b/D-c/D-d/D-e) or is a fallback (D-f). The v1 K-Radar pseudo-GT bridge is deleted.

### 4.4 Ablations (v3 — A-7 promoted to firm, A-6 becomes drop-first stretch)
| # | Variable | Question answered | Compute cost on 4060 (est.) | Firmness |
|---|---|---|---|---|
| A-1 | ± Dirichlet evidential head (vs argmax-Bayes / vs softmax-Bayes) | RQ1: does evidential improve mIoU + ECE? | 3 × 2 days = 6 days | firm |
| A-2 | ± vacuity-conditioned decay τ(m_u) (vs fixed-τ / vs no-decay) | RQ4 + M3 isolation: does vacuity-coupled decay beat fixed decay? | 3 × 1.5 days = 4.5 days | firm |
| A-3 | ± entropy channel in submap descriptor (h_class only vs h_class + h_entropy) | RQ4 + M2 isolation: does the entropy channel improve loop precision? | 2 × 1.5 days = 3 days | firm |
| A-4 | ± vacuity-aware traversability (vs hard threshold) | RQ3: does propagating uncertainty improve safety metrics? | 2 × 0.5 days = 1 day | firm |
| A-5 | ± conjugate Dirichlet fusion across sessions (vs naive overwrite) | RQ4 + M2 isolation: does the parameter-free conjugate fusion help? | 2 × 1 day = 2 days | firm |
| **A-7** | **± openset channel in M1 (15-D vs 14-D under v3 open-set split)** | **RQ2 lead: does the dedicated unknown channel matter, or does vacuity alone suffice?** | 2 × 2 days = 4 days | **firm in v3** (was stretch in v2) |
| A-6 (drop-first stretch in v3) | ± voxel size (0.10 m vs 0.25 m) | defensive: are mIoU gains a voxel-size artefact? | 2 × 2 days = 4 days | drop first if pressed |

**Total firm ablations (A-1…A-5 + A-7):** ≈ 20.5 GPU-days on 4060. **Including A-6:** ≈ 24.5 days. Plus full main eval runs (≈ 10 days) plus the v3-new Khronos dynamic run (≈ 3 days), total ≈ 38 GPU-days for ablations + main results. Fits in the W6-W18 window of §7 with overhead for re-runs.

**4060 ablation budget verdict (v3).** Run **6 ablations (A-1…A-5 + A-7) firmly + A-6 only if all gates green by W14**. A-6 drops first because the voxel-size confound is a rebuttal-only question; A-7 is now firm because the open-set channel is the methodological choice the lead RQ depends on.

### 4.5 Compute budget (v3 reality on 4060 + Orin NX, with +3 weeks Khronos and +2 days Jetson video)
- **Training and main eval:** single workstation, 1× **RTX 4060 (16 GB)**.
- **Deployment benchmark + demo video:** 1× Jetson Orin NX (16 GB) — §III.E latency + memory + the **v3-new 30-second supplementary video** capture (W24-W25).
- **Estimated wall-clock budget across the 39 weeks (v3 expanded from 36):**
  - Cylinder3D last-layer EDL fine-tune on SemanticKITTI (×2: 20-D closed and 15-D open-set): ~2 days × 2 = 4 days.
  - R2-reimpl bring-up: ~5 days.
  - ConvBKI baseline reproduction: ~3 days.
  - Kimera-Semantics baseline: ~2 days.
  - Khronos closed-set RQ1 run on SemanticKITTI: ~2 days.
  - **Khronos dynamic-split RQ5 run (v3 new): ~3 days reproduction + ~2 days experiment = 5 days.**
  - Main mapping eval over SemanticKITTI seq 08+11-21 per system: ~1 day × 6 systems = 6 days.
  - **Open-set RQ2 eval (v3 lead, replaces Robo3D sweep in primary path): ~4 days** (two splits + nuScenes reverse-check across our system + 3 baselines).
  - KITTI-360 multi-session eval (8 sequences, ~5 revisit pairs): ~4 days.
  - SemanticSpray passive replay (RQ3): ~1 day.
  - Ablations: ≈ 20.5 GPU-days firm + 4 stretch (§4.4 above).
  - Jetson Orin NX deployment latency / memory check + 30 s demo video capture and edit: ~4 days (was 2 in v2; +2 days for video as per user spec).
  - **Total: ≈ 64 GPU-days of 4060 time** (was 60 in v2). At 5-6 productive GPU-days per calendar week, that is ~13-15 weeks of pure runtime, fits the W4-W26 window of §7's 39-week IROS schedule.

### 4.6 Embedded Real-Time Demo (NEW in v3)
- **Platform:** Jetson Orin NX (16 GB), TensorRT FP16 build of the EvidLife-Map inference path; the M1 head and the M3 decay run on GPU, M2 loop closure runs on CPU.
- **Data source:** SemanticKITTI seq 08 offline replay over rosbag at native 10 Hz (or SemanticSpray rosbag if the wet-road condition is more visually informative; choice in W24).
- **Duration:** 30 s continuous (300 frames @ 10 Hz).
- **Output:** real-time metric-semantic voxel map with the **traversability overlay (RQ3 output)** and a vacuity heat-map overlay showing high-vacuity voxels in a distinct colour; recorded as an MP4 with annotated overlay timestamp.
- **Use in paper:** referenced from §IV.G (new sub-section "F. Embedded Demo") and submitted as supplementary `evidlife_demo.mp4`; reviewers can see the system actually running rather than reading a latency number. **Promotes C4 deployability claim from "benchmark only" to "benchmark + visible demo".**

### 4.7 Reproducibility hygiene (closes audit E7) — renumbered from v2 §4.6
- Code + Docker + ROS 2 launch + EDL fine-tune script (both 20-D and 15-D variants) + **open-set split definition file** (v3 new) + **dynamic split selection script** (v3 new) + Robo3D corruption applier (kept for fallback path) + KITTI-360 revisit-pair builder script + per-experiment seed list + hyperparameter table + **Jetson Orin NX demo capture script** (v3 new).
- LVIO choice documented per dataset: FAST-LIO2 on KITTI-360 (multi-session friendly), R3LIVE on SemanticKITTI single-session, public LIO config on SemanticSpray.

---

## §5 Expected Contributions (v3, 4 contributions)

- **C1 (Algorithm).** A Dirichlet-evidential per-voxel posterior whose **single vacuity scalar simultaneously serves three downstream jobs** — **(i) open-set / OOD detection** for unknown-category voxels (v3 RQ2 lead; Eq. M1.3), (ii) submap descriptor entropy channel (Eq. M2.1), and (iii) lifelong decay trigger (Eq. M3.1). Strictly generalises R2's unspecified Bayes filter (audit S1/S2 closed) and adds the open-set channel (audit S5 closed). The "one scalar, three jobs" framing — **with the (i) job now anchored on open-set OOD rather than corruption** — is the v3 novelty hook.

- **C2 (System).** A complete online metric-semantic mapping system EvidLife-Map with confidence-aware loop closure, parameter-free conjugate Dirichlet inter-session submap fusion, and vacuity-conditioned voxel decay — closing audit T1/T2/T3/T4 in one paper. **Dynamic-scene head-to-head against Khronos demonstrates that the implicit vacuity-instability mechanism is competitive with explicit 4D modelling at far lower latency on Jetson Orin NX** (v3 RQ5; see C3 (v)). Deployable on Jetson Orin NX (§III.E target ≥ 5 Hz, **plus 30-s supplementary demo video**). First system to ship the lifelong + open-set + calibration + dynamic-competitive quartet together on a public LiDAR-only stack.

- **C3 (Empirical).** Across SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, SemanticKITTI dynamic split, and SemanticSpray: (i) **≥ +2 mIoU and ≥ −20 % ECE** over a faithful R2-reimpl on SemanticKITTI (RQ1 H1); (ii) **AUROC ≥ 0.80 and AUPR ≥ 0.60** for vacuity-as-OOD-detector on the SemanticKITTI 14/5 open-set split, with closed-set mIoU dropping by ≤ 1.5 (RQ2 H2 primary); (iii) **≥ +10 pp safe-region recall** on SemanticSpray passive replay (RQ3 H3); (iv) **≥ 80 % stale-voxel removal precision and ≤ 1.5× ECE drift** on KITTI-360 multi-session (RQ4 H4); **(v) dynamic-object mIoU within 3 of Khronos and static recall within 1 pp, at ≥ 5 Hz on Jetson Orin NX (RQ5 H5).** All numbers replaced by measured ones before submission; targets set as the §9 Go/No-Go thresholds.

- **C4 (Reproducibility + Deployability).** Code + Docker + ROS 2 launch + Jetson Orin NX deployment benchmark **plus 30-s online-mapping demo video in supplementary material** — directly attacks R2 audit E7 ("复现性零保障"). **The deployability claim, previously optional in v2, is now firm in v3** ("first lifelong evidential MSM that runs at ≥ 5 Hz on a 16 GB Orin NX, demonstrated on supplementary video"); hardware-grounded and visually verifiable.

---

## §6 Risk Register (v3)

| ID | Risk | Prob | Impact | Mitigation | Trigger to fall back |
|----|------|------|--------|------------|---------------------|
| R-2 (kept) | Khronos [A2] releases a multi-session / lifelong extension before IROS 2027 submission | M | H | Track arXiv weekly via post-research literature monitor; pre-register our differentiation as "vacuity-coupled decay, no separate change-detection network" | If Khronos-v2 lands on lifelong before us, sharpen claim to the "one vacuity, three jobs" calibration story |
| R-3 (kept) | GS-LIVO [B2.7] HKUST sibling lab releases semantic-GSplat extension | M | M | Differentiate on representation (voxel-Dirichlet vs Gaussian) and target (lifelong vs photo-realistic single-session) | If GS-LIVO-semantic ships, sharpen our "Jetson Orin NX deployable lifelong with demo video" angle |
| R-4 (kept) | Dirichlet evidential fusion fails to beat ConvBKI in mIoU | M | H | Calibration (ECE / AUROC unknown) is a separate axis; we win on calibration + lifelong even if mIoU ties | If ECE also ties, retreat per §0 fallback to calibration-only paper at RA-L (drop M2/M3 to brief sub-sections) |
| R-6 (kept) | KITTI-360 multi-session revisit overlap insufficient for RQ4 | M | M | Pre-compute overlap matrix in W3 (deliverable W3.b); if pairs < 5, augment with SemanticKITTI cross-day sequences | If still < 5, demote RQ4 H4 to single-session ECE-drift study, document as scope reduction |
| R-7 (kept) | Voxel-size confound contaminates mIoU vs ConvBKI | M | M | Ablation A-6 (now drop-first stretch in v3) if time allows; report all numbers also at fixed 0.25 m | n/a |
| R-8 (kept) | Cylinder3D licence (or its weight release) prevents code release | L | M | RangeNet++ (BSD) as backup; train from scratch on 4060 (~5 days) | n/a |
| R-9 (kept, sharpened) | 8-page IROS overflow given 5 RQs (one added in v3) and 5 datasets | **H** (up from M in v2) | M | Move ablation tables, per-class IoU, KITTI-360 per-sequence numbers, and the new RQ5 dynamic detail to supplementary; keep main paper at 5 RQs but 3 hero claims; demo video lives in supplementary (saves 0 page; it's a file attachment) | Drop A-6 from main, defend "two-pair" RQ4 demo, push RQ5 details to supplementary table |
| R-10 (kept) | Reviewer demands real-robot deployment | M | M | Cite scope as "methodological + public-dataset + Jetson Orin NX benchmark + 30-s demo video"; the demo video softens this risk | If desk-rejected, the Jetson video + numbers become the primary deployability evidence |
| R-11 (kept) | RTX 4060 16 GB VRAM insufficient for Cylinder3D + 20-D Dirichlet head + voxel-hash + KITTI-360 batch | H | M | Batch size = 1 with gradient accumulation; freeze Cylinder3D backbone (fine-tune last layer only); offload submap RAM to host; checkpoint per submap | If still OOM on KITTI-360, swap Cylinder3D for SalsaNext (lighter) and re-time |
| R-12 (kept) | ConvBKI authors' reimpl fails to reproduce published numbers on our 4060 | M | H | Allocate full W2 + W3 to ConvBKI reproduction; pin CUDA + PyTorch versions; cross-check with published seq-08 mIoU within ±2 | If reproduction fails after 2 weeks, document discrepancy and use published numbers in a quoted-only table |
| R-13 (kept) | KITTI-360 multi-session revisit length / overlap is too short to demonstrate stale-voxel decay | M | M | Pre-compute overlap matrix in W3; augment with synthetic temporal-gap injection on single sessions | Demote to "synthetic-revisit study only" and shrink H4 from precision-recall to AUROC of stale-voxel ranking |
| R-14 (kept) | EDL Dirichlet head training is unstable / collapses to uniform Dirichlet | M | H | Use Sensoy 2018 KL-annealing schedule; gradient-clip; monitor evidence sum trajectory; warm-start from softmax pretrain | If unstable after 1 week of tuning, switch to posterior network (PostNet, Charpentier 2020) |
| **R-15 (NEW in v3)** | **Open-set 14/5 split definition is methodologically controversial** (reviewers argue we cherry-picked easy unknowns, or that the split leaks training-time information through related-class features) | **M** | **H** | Pre-register the split (table in supplementary) before any RQ2 experiment runs; report both N=5 primary and N=3 robustness splits; **add the nuScenes reverse-cross-domain check as an independent validation that does not depend on our split choice**; release the split definition file in supplementary | If a reviewer's preferred split is starkly different, retreat to the "AUROC across multiple splits" framing and report mean ± std |
| **R-16 (NEW in v3)** | **SemanticKITTI lacks taxonomic diversity for a meaningful 5-class unknown** — the withheld 5 classes are too similar in LiDAR appearance to known classes (or too dissimilar, making the task trivial) | M | M | Validate split quality by W5 with a 2-NN baseline check on raw features; the **nuScenes reverse cross-domain check provides a backup OOD evaluation independent of SemanticKITTI's taxonomy**; if both splits are problematic, the **Robo3D fallback** (D-f dataset retained) covers the same metric family (AUROC of vacuity for anomaly detection) | If both SemanticKITTI splits and nuScenes cross-domain fail to produce informative AUROC by W7, switch RQ2 to Robo3D corruption fallback (H2-fallback in §2) within the same week |
| **R-17 (NEW in v3)** | **Khronos reproduction fails for the dynamic-split RQ5 experiment** — authors' release is incomplete, dependency-broken, or the dynamic-split protocol cannot be matched | **H** | **H** | Allocate W18-W20 firmly to Khronos bring-up (gated by G-6); pin docker environment matching Khronos arXiv 2402.13817 release; reach out to Khronos authors at W17; **as fallback, use Khronos's published dynamic-scene numbers and report a quoted-comparison Table V with a relative-improvement framing and explicit disclaimer in §V about the comparison being approximate** | If full reproduction fails by W20, ship Table V as a "published numbers vs ours" hybrid table with disclaimer; the RQ5 H5 hypothesis becomes "competitive with published Khronos numbers", which is a weaker claim but still publishable |

**H/M/L summary (v3):** **H: 5** (R-4, R-9, R-11, R-12, R-15, R-17 — algorithmic-fail + page-overflow + hardware-fail + baseline-fail + open-set-validity + Khronos-repro are the structural risks; v3 raised R-9 to H because the new RQ5 + demo video tighten the page budget); **M: 8** (R-2, R-3, R-6, R-7, R-10, R-13, R-14, R-16); **L: 1** (R-8). **Net 14 risks** (v2 had 12: 3 H + 8 M + 1 L). v3 added 3 (R-15, R-16, R-17), zero retired, and re-graded R-9 up from M to H.

(v1 risks R-1/R-5 already retired in v2. v2 risks R-2..R-14 all carried forward to v3 with minor wording updates noted above.)

---

## §7 Timeline (≈ 39 weeks, IROS 2027 deadline ≈ 2027-03)

Reference today = 2026-05-28; IROS 2027 paper deadline ≈ 2027-03 (formerly 4-week buffer in v2; **v3 buffer reduced to ≈ 1 week** because P5 absorbed +3 weeks for Khronos dynamic and +2 days for Jetson video).

| Phase | Weeks | Calendar (approx.) | Focus | Concrete output / milestone |
|-------|-------|---------------------|-------|-----------------------------|
| **P1 Bring-up** | W1-W3 | 2026-06 → 2026-06-mid | R2-reimpl + Cylinder3D fine-tune (×2: 20-D and 15-D heads) + KITTI-360 revisit-overlap matrix + open-set split pre-registration | W1: R2-reimpl runs; W2: Cylinder3D EDL last-layer fine-tune complete (20-D head); **W3: KITTI-360 revisit matrix delivered (R-13 check); v3-new: open-set 14/5 split definition pre-registered (R-15 mitigation); first Go/No-Go (§9 G-1, G-3)** |
| **P2 Algorithm** | W4-W6 | 2026-07 → 2026-07-mid | M1 evidential head integrated; A-1 ablation skeleton; v3-new: 15-D open-set head fine-tune + initial AUROC sanity check | W4: M1 working end-to-end on seq 08; **W4 second Go/No-Go (§9 G-2 evidential head ≥ +1 mIoU)**; W5: ConvBKI reimpl validated (R-12 check); **v3-new W5-W7 open-set RQ2 sanity (R-16 check, must yield non-trivial AUROC)**; W6: A-1 numbers |
| **P3 Systems** | W7-W12 | 2026-08 → 2026-09 | M2 loop+fusion; M3 decay; A-2 / A-3 / A-5 ablations | W7-W8: M2 single-session loop closure on SemanticKITTI; W9-W10: M3 decay integrated; W11-W12: A-2 + A-3 + A-5 numbers; **W12 third Go/No-Go (§9 G-4 M2 loop precision ≥ 70 %)** |
| **P4 Open-set + Robustness** | W13-W17 | 2026-10 → 2026-11-mid | Open-set RQ2 lead + nuScenes reverse-check + A-7 ablation; Robo3D fallback only if open-set fails | W13-W15: open-set RQ2 H2 numbers (primary 14/5 + robustness 16/3 splits); W16: nuScenes reverse-cross-domain OOD; W17: A-7 (± unknown channel); **W17 fourth Go/No-Go (§9 G-5 open-set AUROC ≥ 0.80 OR Robo3D fallback ready)** |
| **P5 Lifelong + Dynamic + Downstream + Deploy** (extended in v3 from W19-W24 to W18-W26, +3 weeks for Khronos dynamic, +2 days for Jetson video) | W18-W26 | 2026-11-mid → 2027-01 | KITTI-360 multi-session RQ4; **v3-new: Khronos reproduction + RQ5 dynamic split experiment**; SemanticSpray RQ3; Jetson Orin NX deploy benchmark; **v3-new: 30-s Jetson demo video capture + edit** | W18-W20: **v3-new Khronos bring-up (G-6 checkpoint W20)**; W21-W22: RQ5 H5 dynamic numbers; W23-W24: RQ4 H4 KITTI-360 numbers; **W24: Jetson Orin NX benchmark + 30-s demo video capture (2 days)**; **W25: video edit + supplementary annotation overlay (~2 days)**; W26: RQ3 H3 SemanticSpray + buffer |
| **P6 Writing** | W27-W33 | 2027-01 → 2027-02-mid | `/ars-full` first draft, figures, tables, supplementary including demo video | W27-W30: draft 1 from outline_v3; W31: internal reviewer-sim (`academic-paper-reviewer`); W32-W33: revision pass |
| **P7 Submission** | W34-W39 | 2027-02-mid → 2027-03 | Reviewer-sim second pass + format-convert + camera-ready prep + buffer (≈ 1 week, v3 reduced from v2's ≈ 3 weeks) | W34-W36: second reviewer-sim + revision; W37: format-convert to IROS LaTeX; W38-W39: buffer / camera-ready / submission |

**Total: 39 weeks** (v2: 36 weeks). IROS 2027 deadline ≈ 2027-03; W39 ≈ 2027-03-mid. **Submission window is feasible but tight: the buffer drops from ~3 weeks in v2 to ~1 week in v3.** No further additions to P5 should be accepted without re-baselining.

**First milestone (W1 deliverable):** R2-reimpl baseline running end-to-end on SemanticKITTI seq 08, producing at least a (possibly poor) mIoU number, by end of **W1 = 2026-06-04**. Unchanged from v2.

---

## §8 Open Questions for the User (v3)

v2 had three open questions on (RQ2 lead, Khronos comparison scope, Jetson deployment depth). **All three were resolved by user decision 2026-05-28 and are now closed in v3:**

- ✅ **v2 Q1 (RQ2 lead): resolved → open-set vacuity** (Robo3D demoted to fallback).
- ✅ **v2 Q2 (Khronos scope): resolved → add dynamic-scene experimental head-to-head** (+3 weeks accepted).
- ✅ **v2 Q3 (Jetson depth): resolved → benchmark + 30-s demo video** (+2 days accepted).

**v3 leaves no new open questions to the user.** All scope decisions for the v3 plan are committed. The next user touch-point should be after the W3 Go/No-Go (R2-reimpl baseline + KITTI-360 revisit matrix + open-set split pre-registration) or at any earlier red-flag.

(If the user wishes to revisit any of the three closed questions, plan returns to v2-style 3-question state and a v4 patch is warranted. No anticipated open question identified by the architect at v3 commit time.)

---

## §9 Go / No-Go Criteria (v3, 6 conditions)

All six must be green to proceed past P5 (Lifelong + Dynamic phase end, W26); any red triggers documented fallbacks. Each is checkpointed at the date shown in §7.

1. **G-1 — Baselines reproducible (W3).** R2-reimpl produces an end-to-end mIoU on SemanticKITTI seq 08 by end of W3, within ±2 mIoU of published Cylinder3D numbers. If not, slip the schedule one week and re-baseline; if still failing W4, switch backbone to RangeNet++.

2. **G-2 — Evidential head working (W4).** M1 Dirichlet evidence module shows **≥ +1 mIoU** and reduced ECE on at least SemanticKITTI seq 08 by end of W4. If **< 0 mIoU delta**, retreat to §0 fallback (calibration-only RA-L paper).

3. **G-3 — KITTI-360 revisit usable (W3).** Pre-computed odometry-overlap matrix delivers ≥ 5 revisit pairs with ≥ 30 m sustained overlap by end of W3. If < 5, augment per R-13 mitigation; if even synthetic injection fails by W18, demote RQ4 to a single-session ECE-drift micro-study.

4. **G-4 — M2 loop closure precision (W12).** Confidence-aware loop closure achieves **≥ 70 % precision at 50 % recall** on SemanticKITTI seq 08 self-loop pairs by end of W12. If not, drop the entropy-channel descriptor variant (A-3) from main and use h_class only.

5. **G-5 — RQ2 lead story green by W17 (revised in v3 from v2 W18; primary metric changed).** Either: (a) **open-set vacuity AUROC ≥ 0.80** on the SemanticKITTI 14/5 primary split AND **AUPR ≥ 0.60**, with closed-set mIoU on the 14 known classes within 1.5 mIoU of the fully-supervised baseline, by end of W17 — OR — (b) Robo3D fallback path: vacuity-as-corruption-detector AUROC ≥ 0.80 averaged across corruptions by end of W17. If neither, RQ2 is demoted to an honest negative result section (still publishable for IROS), and C1's "one vacuity, three jobs" framing tightens to "one vacuity, two jobs" (entropy descriptor + decay only).

6. **G-6 — Khronos baseline ready by W20 (NEW in v3).** Khronos [A2] reproduction runs end-to-end on SemanticKITTI seq 08, producing a published-paper-comparable closed-set mIoU number by end of **W20**. If full reproduction fails: ship Table V as a "published numbers vs ours" hybrid per R-17 mitigation; the RQ5 H5 claim weakens to "competitive with published Khronos numbers" but RQ5 is not aborted. Hard abort of RQ5 only if Khronos published numbers are also not available for the dynamic-split frames (unlikely but documented).

If any of G-2 or G-4 reds, the **§0 fallback to a calibration-only RA-L paper** engages. G-1 / G-3 / G-5 / G-6 reds shrink scope but do not abort.

---

*End of Research Plan v3. Companion: `paper_outline_v2.md` (filename retained for v3 contents; v2 contents archived at `paper_outline_v2_archived.md`). v2 plan frozen at `research_plan_v2.md`. v1 plan frozen at `research_plan_v1.md`.*
