---
version: v4 (2026-05-29) — patches: 5090-server compute, W2 done, W3 preliminary verification PASS, Cylinder3D backbone blocker called out, 9-month re-anchor
v3 archived as: research_plan_v3_archive.md (same directory)
v2 archived as: research_plan_v2.md (same directory)
v1 archived as: research_plan_v1.md (same directory)
v4 axis (unchanged from v2/v3): EvidLife-Map  =  candidate X (EvidVox)  +  candidate Y (Lifelong + Map Decay)
v4 venue: IROS 2027 (deadline ≈ 2027-03; ≈ 9 months horizon from 2026-05-29)
v4 compute: **local 5060 Laptop (8 GB, dev/smoke)** + **5090 server `server@100.64.0.5` via Tailscale (4 × RTX 5090 32 GB, sm_120, CUDA 13.0, 80 cores, 251 GB RAM, 879 GB disk)** + Jetson Orin NX (deployment + 30 s demo video). **Cloud-train budget cancelled** (free local server replaces it).
v4 datasets: SemanticKITTI / nuScenes-LiDARSeg / KITTI-360 / SemanticKITTI dynamic split / SemanticSpray (+ Robo3D as fallback for RQ2)
---

# Research Plan (v4)

**Working title (primary):** *EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay and Open-Set Vacuity*

**Tagline (one line):** A LiDAR-only online metric-semantic mapping system that (i) replaces the unspecified per-voxel Bayes filter of R2-class systems with a closed-form Dirichlet-evidential posterior carrying explicit unknown mass, (ii) maintains the map across sessions through confidence-aware loop closure and voxel-hash submap fusion, and (iii) ages stale voxels via a vacuity-driven decay rule so the same representation answers three reviewer-credible questions — calibration, open-set, lifelong — without inflating the modality stack.

**Author / lead:** (TBD by user)
**Target venue:** IROS 2027 (8 pages double-column, conference systems track).
**Prepared:** 2026-05-29
**Stage:** Stage-1 architect output (v4 patch) of the `academic-pipeline` skill; precedes any `/ars-full` execution.
**Upstream inputs (carried forward unchanged from v3 unless flagged):**
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\r2_audit.md` (R2 6-dim technical audit)
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md` (54-entry literature scan)
- `D:\_7_sci\semantic_mapping\_Online_Metric_Semantic_Mapping_for_Autonomous.txt` (R2 cleaned full text)
- `D:\_7_sci\semantic_mapping\_new_paper\drafts\research_plan_v3_archive.md` (frozen v3, for diff)
- **NEW v4:** `D:\_7_sci\semantic_mapping\_new_paper\W2-1_status.md` (Cylinder3D + spconv 2.x integration blocker)
- **NEW v4:** `D:\_7_sci\semantic_mapping\_new_paper\artifacts\preliminary_results.md` (W3 preliminary verification)
- **NEW v4:** `D:\_7_sci\semantic_mapping\_new_paper\drafts\paper_v2.md` §IV.0 "Preliminary Status" box (2026-05-29 snapshot)

---

## §0 Direction Decision — v2 reset retained through v3 and v4 (mandatory before §1)

v1 selected candidate **W = EviRad-Map** (LiDAR + 4D-radar evidential fusion) as the primary axis, with candidate **X = EvidVox** (LiDAR-only evidential) as fallback. The v1 §9 Go/No-Go listed **G-3** ("K-Radar pseudo-GT validated by 50-frame manual sparse-GT sanity sample ≥ 75 % agreement, by end of Week 3") as the controlling gate.

**v2 trigger.** The user's resource brief (2026-05-28) sets the manual K-Radar voxel annotation budget to **zero person-days**. With no manual sparse-GT, G-3 cannot be evaluated at all; the entire pseudo-GT defence collapses to "trust a pretrained Cylinder3D on out-of-domain frames", which is precisely the audit weakness (E1/E3) we promised to fix. v1 §6 R-1 therefore deterministically fires and v1 §0 fallback engages.

**v2 primary axis (retained in v3 and v4):** **EvidLife-Map** = candidate X (EvidVox) **combined with** candidate Y (Lifelong + Loop + Map Decay from v1 §0 Top-2 axis). This restores the "two strong contributions" balance that pure X alone would not carry — a system paper needs both an algorithmic and a systems contribution to fill 8 IROS pages credibly without radar.

| Axis | X. Evidential Voxel Fusion (R2 Top-1) | Y. Lifelong + Loop + Decay (R2 Top-2) | **X+Y. EvidLife-Map (v4)** |
|---|---|---|---|
| Novelty | 3 — crowded by ConvBKI [C9] / LatentBKI [D10] | 2 — Khronos [A2] owns dynamic-scene 4D MSM | **4** — Dirichlet-vacuity-as-decay-signal is the bridge nobody yet ships in one system |
| Venue-fit IROS 8 p | 3 — feels like a method paper, hard to fill 8 p | 4 — IROS systems track loves it | **5** — algorithm + system, fits IROS double column comfortably |
| R2 audit delta covered | S1/S2/S5 | T1/T2/T3 + T4 | **S1/S2/S4/S5 + T1/T2/T3/T4** (8 of 30 audit findings closed) |
| SOTA differentiation | Must beat ConvBKI in mIoU | Uphill vs Khronos | **v3+v4: dynamic-scene head-to-head vs Khronos accepted**; we fight on **calibration + lifelong + dynamic** axis, where Khronos/Clio/ConvBKI each cover only one |
| Data availability (no manual) | 5 — SemanticKITTI / nuScenes-LiDARSeg dense GT | 4 — KITTI-360 has multi-session revisits | **5** — every dataset is fully public and pre-labelled |
| Engineering effort | 2 — Dirichlet head + Bayes update | 4 — submap + loop + decay | **v4: 4** — 5090 server lifts the v3 4060-VRAM bottleneck; backbone blocker (W2-1) is the new critical-path risk |
| Reviewer risk | Low | Med (Khronos head-to-head) | **Low-Med** — v3 escalation (dynamic head-to-head) preserved; v4 adds preliminary-scale evidence (W3) that defangs the "vapourware" objection |
| Compute fit | v3: tight on 4060 | — | **v4: comfortable** — 4 × 5090 32 GB on server, 80 cores, 879 GB disk; the v3 ablation budget that was 64 GPU-days on 4060 becomes ~16 wall-clock days on a single 5090 |

**Decision (v4 primary, unchanged from v2/v3):** **EvidLife-Map** — three integrated modules (M1 Dirichlet-evidential per-voxel posterior; M2 confidence-aware loop closure + multi-session voxel-hash submap fusion; M3 vacuity-driven voxel decay + map staleness ageing) trained and evaluated entirely on public LiDAR-only datasets, deployable on Jetson Orin NX with a published 30-second demo video.

**Fallback (single layer only, no further nesting).** If by Week 4 the Dirichlet evidential head fails to beat the argmax-Bayes baseline by ≥ +1 mIoU on SemanticKITTI seq 08 *at full scale on the resolved backbone*, retreat to a **calibration-only paper**: drop M2 and M3 to brief sub-sections, keep M1 as the headline, target **RA-L** instead of IROS (4-page brevity hides the missing systems story). This is documented in §9 G-2. **v4 update on G-2:** the preliminary-scale R2-vs-M1 ordering at matched 0.21 M-param PointNet (M1 wins on mIoU 14.73 % vs R2 1.69 %, ECE 0.171 vs 0.490, latency 3.0 ms vs 663 ms) is *consistent* with the G-2 thesis but does not formally pass it — G-2 demands the same ordering on the Cylinder3D-class backbone at full SemKITTI val. So G-2 is provisionally green and we proceed at risk.

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

## §0.6 v2 → v3 Change Log (preserved unchanged)

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

## §0.7 v3 → v4 Change Log (NEW in v4)

The v3 plan was written 2026-05-28 against an *assumed* RTX 4060 (16 GB) single-workstation compute model and a not-yet-built backbone. In the 24 hours since, four substantive changes have landed: hardware reality, W2 substantive progress, W3 preliminary verification (3 of 5 RQ-thesis pieces validated end-to-end on real KITTI data), and a Cylinder3D backbone blocker. v4 reflects all four.

| Dim | v3 | v4 | Reason |
|-----|-----|-----|--------|
| **Local GPU** | RTX 4060 (16 GB) assumed | **RTX 5060 Laptop (8 GB)** — dev/smoke/debug/Jetson cross-compile/paper writing only | Hardware reality check 2026-05-28; VRAM halved vs v3 assumption |
| **Primary training compute** | Same local 4060 | **5090 server `server@100.64.0.5` via Tailscale: 4 × RTX 5090 32 GB sm_120, CUDA 13.0, 80 cores, 251 GB RAM, 879 GB disk; workspace `~/Documents/yping/mapping/code/`** | User secured free server access; replaces all training compute paths |
| **Cloud budget** | v3 had no cloud; v4-delta proposed $30-80 autoDL/Vast/RunPod | **Cancelled in v4 main plan** — server access removes the need | The 5090 server is free, 4× the v3 4060 VRAM per card, and 4 cards |
| **Effective wall-clock** | 39 weeks W4-W26 to cover 64 GPU-days on 4060 | Same 64 method-GPU-days → ~16 wall-clock days on a single 5090; 5× speedup makes per-experiment turnaround fast enough to iterate | 5090 sm_120 + cu130 stack roughly 3-4× per-tensor over 4060 + extra VRAM for batch ↑ |
| **W2-2 status** | Pending | **Substantively done**: train sequences 00-07, 09, 10 (19 230 velodyne frames + labels) on server at `/data/shared/SemanticKITTI/dataset/sequences/`; seq 08 first-100 frames done, full 4071 frames pending background transfer | Server rsync run in W2 background |
| **W2-3 status** | Pending | **Done**: flat-tensor evidence accumulator (replaces dict-based v3-era prototype) delivers R2 10× / M1 60× / E2E 13× speedup. Json artifacts under `artifacts/` | W2 engineering output |
| **W2-4 status** | Pending | **Done**: first RQ1 comparison run end-to-end on 100-frame SemKITTI seq 08; paper §III.B thesis (M1 > R2 on mIoU + ECE + latency at matched backbone) validated at preliminary scale | W2 engineering output, gated on W2-3 |
| **W2-1 status (Cylinder3D)** | Pending | **HARD BLOCKER (see §3.9)** — spconv 1.x → 2.x port: structural patches applied (imports, replace_feature, indice_key uniqueness, weight permute) but inference outputs all-NaN. Root cause = internal kernel index iteration order change (D,H,W) ↔ (W,H,D), not fixable by pure tensor permute. Three unblock paths in §3.9; decision deadline P1+0 = 2026-06-01 | Engineering diary `W2-1_status.md`; this is the v4 critical-path risk |
| **W3 preliminary results** | Not present in v3 | **Three of three method-level claims validated at preliminary scale** (PointNetVanilla 0.21 M params, 30 ep, 80 frames seq 08 train + 20 frames val): (W3-A) EDL PointNet trained, best ep-3 val_miou 0.111; (W3-B) M3 vacuity-decay top10%/bottom10% loss-ratio = **2.64×** on 747 047 voxels (paper §III.D Eq 11 PASS); (W3-C) M1 open-set vacuity AUROC = **0.8082** on 14/5 SemKITTI split (RQ2 H2 ≥ 0.80 PASS); (W3-D) RQ4 lifelong sim A/B-half on seq 08 first 100 frames yields stale-voxel removal **P = 0.77, R = 0.52** (paper §IV.F preview). Artifacts: `artifacts/rq1_fair_100frames.json`, `artifacts/m3_validation.json`, `artifacts/m1_openset.json`, `artifacts/preliminary_results.md` | W3 done in one chat session of focused engineering after W2 unblock |
| **"One vacuity, three jobs" matrix** | Asserted only | **2 of 3 legs verified end-to-end on real KITTI data** (open-set OOD ✓, decay clock ✓); 3rd leg (loop closure descriptor with entropy channel) gated on KITTI-360 transfer + revisit-pair eval | Preliminary results artifact §"One vacuity, three jobs validation matrix" |
| **Effective remaining timeline** | 39 weeks W1-W39 from 2026-05-28 baseline | **9-month re-anchored**: today = 2026-05-29; IROS deadline ≈ 2027-03; ~36-39 weeks runway. **P1 (preliminary done in 1 week wall-clock vs v3 W1-W3) gives back ~2 weeks of buffer**. New phasing (§7): P1 (now → +2 wk) unblock Cylinder3D + scale RQ1/RQ2 to full train; P2 (+2-4 wk) RQ4 KITTI-360; P3 (+4-6 wk) RQ3 + RQ5; P4 (+6-9 wk wall-clock; ≡ +6-9 months calendar) ablations/robustness/Jetson; P5 (+9 months wall-clock through 2027-03) writing + submission | W3 preliminary done early; backbone unblock is now the gating step |
| **G-1 status** | Pending W3 checkpoint | **Provisionally green at preliminary scale** (W3 done); awaiting full-Cylinder3D verification once W2-1 unblocked | Preliminary RQ1 ordering validated |
| **G-2 status** | Pending W4 | **Provisionally green** (M1 > R2 on all three of mIoU, ECE, latency at matched backbone) but formal pass deferred to full Cylinder3D run | Preliminary trajectory consistent with H1; G-2 final = full-backbone re-run |
| **G-3 status** | Pending W3 | **Pending** — gated on KITTI-360 download + revisit-overlap matrix; no preliminary equivalent | Multi-session dataset not yet on server |
| **G-5 status** | Pending W17 | **Provisionally green at preliminary scale** (AUROC = 0.8082 on 14/5 split); full-scale rerun required | Preliminary RQ2 PASS |
| **G-4 / G-6** | Pending W12 / W20 | **Unchanged from v3** — still pending | M2 loop closure and Khronos reproduction not yet attempted |
| **Risks** | 14 (5H + 8M + 1L) | **16** (5H + 9M + 2L) — adds R-21 (spconv 1.x/2.x value-semantic gap; M-H) and R-22 (PVKD code quality unknown; M); demotes R-12 (ConvBKI reproduction) from H to M (cheap on 5090) and R-14 (EDL instability) is **CONFIRMED** at preliminary scale (epoch-1 collapse observed; KL warm-restart mitigation required) | New risks from backbone blocker; old risks burn down on server compute |
| **Method §3** | M1/M2/M3 + §3.7 dynamic | Adds **§3.9 Backbone Resolution Path** explaining Path 1/2/3 trade-offs and the deadline | Backbone blocker is now load-bearing |
| **Cloud-train section** | Not in v3 main | **Removed entirely** | Server access cancels cloud need |
| **Compute budget §4.5** | 64 GPU-days on single 4060 | **~16 wall-clock days on single 5090; 4× cards available for parallel ablations** | Server math |
| **Open Questions §8** | Closed (all resolved 2026-05-28) | **Two new open questions** (§8): backbone path selection (Path 1 vs 2 vs 3) and EDL stability fix priority | Backbone blocker requires user decision; KL annealing schedule rework is non-trivial |

---

## §1 Problem Statement & Motivation (v4 — text unchanged from v3 except §1.5 added)

### 1.1 Scenario
A ground robot operates outdoors over weeks of repeated traversal of the same area (campus, urban block, industrial site). It must build, online and on-board (Jetson-class GPU), a dense voxel map labelled with closed-set semantics plus an explicit "unknown" channel, and must keep that map correct across (i) **single-session noise** (LiDAR drop-out, sensor occlusion), (ii) **open-set encounters** (object categories absent from the training taxonomy — construction debris, exotic vegetation, unmodelled vehicle types), and (iii) **inter-session change** (revisits weeks later with moved objects, construction, new vegetation). Camera and radar are *not* assumed; the system is LiDAR-only with optional RGB-derived priors at training time only.

### 1.2 R2 baseline summary and the four reviewer-credible weaknesses
R2 [A1] (Jiao et al., HKUST) couples LVIO with an NvBlox TSDF backbone and a confidence-aware HRNet [P1] segmenter, fused per-voxel through an **unspecified** iterative Bayes filter. The R2 audit, distilled to the four cripple-points an even mildly-attentive RA-L/IROS reviewer will raise:

1. **Methodological under-specification.** R2's Bayes update formula, prior, and confidence-into-fusion pathway are all missing (audit S1/S2/S4). Reproducible? No.
2. **No quantitative evidence.** Zero ablation, zero baseline number, zero mIoU, zero F-score; evaluation only on two self-recorded sequences (audit E1/E2/E3/E5).
3. **No open-set / OOD handling.** "Unlabeled = untraversable" (R2 p.3, audit S5); pessimistic and unusable for active exploration in environments where genuinely-novel categories appear (the open-set / OOD-detection-in-mapping gap, v4 RQ2 lead).
4. **No long-term / loop-closure / decay mechanism.** Lifelong drift, stale voxels, re-visits all unaddressed (audit T1/T2/T3/T4).

v1 sought to fix (1)+(3) algorithmically and add a fifth axis (multi-modal weather robustness). v4 (= v3 retained = v2 retained) drops the modality axis and instead pairs (1)+(3) algorithmic fix with a (4) systems fix — which is exactly the audit's largest cluster of unfilled weaknesses, and **as of v4 is preliminarily validated on real SemKITTI data for the algorithm-half** (W3-A through W3-D).

### 1.3 Differentiation vs current SOTA (v4 — text unchanged from v3)
The literature scan flags three SOTA threats; v4 declares its relationship to each:
- **Khronos [A2/D1]** factorises *short-term motion vs long-term change in time*; v4 factorises *evidence strength vs vacuity in observation* and uses time only as a decay scalar in M3. The decay rule (Eq. M3.1) is dual to Khronos's "long-term change detector" — it requires no explicit change-detection network, only a Dirichlet-conjugate exponential. **v3+v4 escalation:** rather than only declaring this orthogonality in prose, we run a head-to-head experiment on a SemanticKITTI dynamic split (§3.7, §4.2 baseline, §4.4 Table V) so reviewers see numbers, not adjectives.
- **Clio [A3/D2]** compresses *given a task*; v4 produces a *task-agnostic, uncertainty-aware substrate* that Clio could be re-implemented on top of. The Clio LiDAR stub (one of our open-set baselines, §4.2) is the explicit comparison.
- **ConvBKI [C9] / LatentBKI [D10]** use kernel Bayesian inference for *spatial smoothing*; v4 uses **EDL Dirichlet** for closed-form per-voxel epistemic uncertainty *without* spatial-kernel smoothing — which yields a sharper, more honest vacuity signal for the M3 decay rule and for the open-set OOD-detection task (RQ2 lead in v4), at the price of giving up smoothing's mIoU bump (which the M2 loop closure re-projection partially restores).

The novelty story for v4 is **not** "first to fuse modality X+Y" (v1 story); it is **first to wire one statistical quantity (Dirichlet vacuity) into three jobs simultaneously**: (a) **open-set / OOD detection** (v4 RQ2 lead — preliminarily validated W3-C, AUROC 0.8082), (b) loop-closure descriptor entropy term, (c) lifelong decay trigger (preliminarily validated W3-B, 2.64× decay ratio). That single quantity threading three modules is what reviewers will remember.

### 1.4 Thesis statement (v4 — preserved verbatim from v3)
> *A metric-semantic voxel map becomes simultaneously more calibrated, more open-set-aware, and more lifelong-maintainable when its per-voxel posterior is a closed-form Dirichlet-evidential distribution whose vacuity mass is reused as (i) the **open-set / OOD score for unknown-category voxels**, (ii) the loop-closure descriptor's entropy channel, and (iii) the conjugate decay trigger that ages stale voxels — eliminating the three independent ad-hoc heuristics that prior systems (R2, ConvBKI, Khronos) use one each.*

### 1.5 Preliminary verification evidence (NEW in v4 — supports §1.4)
Three of the three method-level pieces of the §1.4 thesis are validated end-to-end on real SemanticKITTI data at *preliminary backbone capacity* (PointNetVanilla 0.21 M params, ≈ 1/250 the capacity of the planned Cylinder3D backbone). The validations are reported in `artifacts/preliminary_results.md` and summarised in paper_v2.md §IV.0:

- **Leg (i) — open-set OOD score.** Vacuity AUROC = **0.8082** on a SemanticKITTI 14-known / 5-unknown split (withheld classes = bicyclist, motorcyclist, truck, other-vehicle, other-ground), evaluated on seq 08 val frames 80-99 (37 838 unknown + 2 055 160 known points). Meets RQ2 H2 ≥ 0.80 threshold at preliminary scale. Best AUROC reached at training epoch 1 with degradation thereafter due to KL annealing pushing the head toward uniform Dirichlet — this is the R-14 instability, now CONFIRMED, requiring late-stage KL warm-restart for full-backbone training.
- **Leg (ii) — loop-closure descriptor entropy channel.** Gated on KITTI-360 transfer (no server-side data yet); explicit `⏸️ pending` mark in the matrix.
- **Leg (iii) — vacuity-driven decay clock.** Top-10 %-vacuity voxels lose **42.3 %** of evidence mass over Δt = 600 s; bottom-10 %-vacuity voxels lose **16.0 %**; ratio **2.64×**. Confirms the directional claim of paper Eq. 11 on 747 047 unique voxels at 0.25 m voxel size over seq 08 first 50 frames.

**Provisional RQ4 lifelong evidence:** seq 08 A/B-half split (first 50 frames as "session 1", next 50 as "session 2", artificial removal of evidence between halves) yields stale-voxel removal **P = 0.77, R = 0.52**. This is a preliminary-scale, single-session, intra-sequence stand-in for the true multi-session KITTI-360 evaluation; it indicates the M2+M3 stack can detect stale voxels but the recall is currently below the v3-stated H4 target of "≥ 80 % precision" by ~3 pp on precision and substantially below the implicit recall expectation. Interpretation: encouraging directional signal, not yet a PASS.

What this preliminary verification accomplishes for the plan: the three-of-three matrix at small scale converts §1.4 from a "we believe" claim to a "we have shown on real data, at small scale" claim. The remaining IROS-publishable evidence requires (a) Cylinder3D-class backbone resolution and (b) full SemKITTI val + KITTI-360 multi-session arena — both gated on the §3.9 backbone-resolution path and the KITTI-360 transfer.

---

## §2 Research Questions & Hypotheses (v4 — RQ text unchanged from v3 except preliminary validation footnotes)

### RQ1 — Does Dirichlet-evidential per-voxel fusion outperform argmax-Bayes / S-BKI-style kernel Bayesian inference on closed-set dense semantic mapping?
- **H1.** On SemanticKITTI sequences 08, 11-21, an EDL-Dirichlet voxel update yields **≥ +2 mIoU** and **≥ −20 % ECE** versus an R2-reimplemented argmax-Bayes baseline at equal voxel size (0.25 m) and equal compute. (mIoU bar lowered from v1's +3 because we no longer have radar to push the upper end; ECE bar held.)
- **Boundary.** Holds for closed-set 19-class setting; for open-set we expect ECE win to grow and mIoU possibly to shrink because "unknown" steals mass from rare classes — quantified in RQ2.
- **v4 preliminary state.** At matched 0.21 M-param PointNet backbone on 100 SemKITTI seq 08 frames, M1 yields mIoU 14.73 %, ECE 0.171, latency 3.0 ms/frame vs R2's mIoU 1.69 %, ECE 0.490, latency 663 ms/frame. The *relative* ordering predicted by H1 holds; absolute mIoU is far below ConvBKI's published 77.7 % because the backbone is ≈ 1/250 capacity. Full-scale Cylinder3D rerun pending backbone resolution (§3.9). Json: `artifacts/rq1_fair_100frames.json`.

### RQ2 — Open-set vacuity as lead (Robo3D corruption as fallback)

**v4 primary formulation (open-set vacuity, unchanged from v3):**
> *Does the per-voxel Dirichlet vacuity score reliably separate voxels whose observed point evidence comes from a held-out unknown category from voxels of known categories, while the closed-set mIoU on the remaining known categories stays within striking distance of a fully-supervised baseline?*

- **Open-set split protocol.** SemanticKITTI 19 → 14 known + 5 unknown. Withheld: `bicyclist`, `motorcyclist`, `truck`, `other-vehicle`, `other-ground`. Robustness split: 16 known / 3 unknown (drop only `bicyclist`, `motorcyclist`, `other-vehicle`). Reverse cross-domain on nuScenes-LiDARSeg classes that do not overlap. Pre-registered split file in supplementary.
- **H2 (v4 primary, unchanged).** On the 14/5 split, voxel-level AUROC ≥ 0.80 and AUPR ≥ 0.60 for known-vs-unknown discrimination, with closed-set mIoU on the 14 known classes dropping by ≤ 1.5 vs the 19-class supervised baseline. On nuScenes reverse cross-domain, AUROC ≥ 0.75.
- **v4 preliminary state.** AUROC = **0.8082** on the 14/5 split at preliminary backbone, seq 08 val frames 80-99 (37 838 unknown + 2 055 160 known points). **Meets the H2 ≥ 0.80 threshold at preliminary scale.** AUPR not yet computed at the preliminary split; full-backbone rerun will report AUROC + AUPR + closed-set mIoU on the same harness. Caveat (R-14 CONFIRMED): best AUROC reached at training epoch 1, after which KL annealing pushes the head toward uniform Dirichlet and AUROC degrades to ~0.67; this is a tuning issue, not a capability issue, and requires the warm-restart schedule before full-backbone training. Json: `artifacts/m1_openset.json`.

**Fallback formulation (Robo3D corruption robustness, unchanged from v3).** AUROC of vacuity ≥ 0.80 across corruptions; H2-fallback retained on standby if open-set split is contested at W5.

### RQ3 — Does propagating evidential uncertainty into downstream traversability improve navigation safety on public closed-loop replay scenarios? (Unchanged from v3.)
- **H3.** On SemanticSpray-style passive-replay traversability evaluation, an uncertainty-gated traversability head produces ≥ +10 pp safe-region recall and ≥ −30 % false-traversable rate vs R2-style hard-rule baseline, at equal coverage.
- **v4 preliminary state.** Not yet attempted.

### RQ4 — How does the evidential map scale and degrade gracefully across multi-session revisits? (Unchanged from v3.)
- **H4.** Across KITTI-360 sequences 00, 02, 04, 05, 06, 07, 09, 10 (≥ 5 revisit pairs from spatial overlap): (i) stale-voxel removal precision ≥ 80 % over the second-pass trajectory after evidence decay; (ii) inter-session map ECE drift ≤ 1.5× the single-session ECE; (iii) multi-session memory footprint ≤ 1.7× the single-session footprint.
- **v4 preliminary state.** A *single-session intra-sequence A/B-half stand-in* on seq 08 first 100 frames yields stale-voxel removal **P = 0.77, R = 0.52**. Below H4's 80 % precision target and well below an implicit recall expectation. Interpretation: directional evidence that M2+M3 detect stale voxels, not yet a PASS. The true H4 evaluation requires KITTI-360 multi-session data (not yet on server) and the M2 loop closure module (not yet implemented). Use this preliminary number as a *lower-bound sanity check* only.

### RQ5 — Dynamic-scene head-to-head vs Khronos. (Unchanged from v3.)
- **H5.** On SemanticKITTI dynamic-heavy frames, dynamic-object mIoU within 3 mIoU of Khronos, static-region recall ≥ Khronos − 1 pp, latency ≥ 5 Hz on Jetson Orin NX.
- **v4 preliminary state.** Not yet attempted; awaits both Khronos reproduction and full-backbone EvidLife-Map.

---

## §3 Technical Approach (v4)

### 3.1 System architecture (ASCII, unchanged from v3 structurally)

```
                    +-------------------+
                    | LiDAR (OS1-128 /  |
                    | Velodyne HDL-64)  |
                    +---------+---------+
                              |
                              v
              +---------------+----------------+
              | LiDAR Semantic Head             |
              | TARGET: Cylinder3D (55.85 M     |
              |   params), pretrained SemKITTI; |
              |   STATUS v4: BLOCKED on spconv  |
              |   1.x→2.x port (§3.9)           |
              | PRELIM: PointNetVanilla         |
              |   0.21 M params (W3 verification)|
              | EDL Dirichlet head:             |
              |   (C+1)-dim with C=19 closed-set|
              |   or C=14 open-set RQ2          |
              +---------------+-----------------+
                              |
                              v
   +--------------------------+------------------------+
   | §3.2  M1  EVIDENTIAL PER-VOXEL POSTERIOR          |
   | per-voxel alpha_v in R^{C+1}                      |
   | vacuity m_u(v) = (C+1) / sum(alpha_v)             |
   | Eqs. M1.1-M1.3 (closed form, no kernel smoothing) |
   | NOTE: vacuity acts as natural OOD score (RQ2)     |
   | NOTE v4: flat-tensor accumulator (W2-3 done)      |
   |   delivers R2 10× / M1 60× / E2E 13× speedup      |
   |   over dict-based prototype                       |
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
| descriptor   |  | NOTE v4 (W3-B): 2.64× decay ratio        |
| d(S) =       |  |   top10% vs bottom10% verified           |
| (h_class,    |  |   on 747k voxels seq 08 first 50 frames  |
| h_entropy)   |  +--------------------+---------------------+
+------+-------+                       |
       |           +-----------+-----------+
       v           |
+------+-------+   v
| pose graph   |   +-------+---------------+
| optimise +   |   | downstream queries:   |
| reproject    |   | (a) uncertainty-aware |
| alpha_v      |   |     traversability    |
+--------------+   | (b) language / task   |
                   |     layer (Clio-      |
                   |     compatible)       |
                   +-----------------------+
```

### 3.2 M1 — Evidential per-voxel posterior (inherits v3 M1, with v4 implementation note)
Replace R2's unspecified Bayes filter (audit S1/S2) by a Dirichlet-evidential posterior. Per voxel `v`, maintain accumulated evidence `α_v ∈ R^{C+1}` where C = 19 SemanticKITTI classes (closed-set RQ1/RQ4/RQ5) or C = 14 (open-set RQ2 lead) and the last channel is the open-set "unknown" channel. **Vacuity `m_u(v) = (C+1) / Σ α_v` acts as a natural OOD score** (the v4 RQ2 lead consumes this directly, preliminarily verified W3-C AUROC = 0.8082). New observations contribute evidence `e_v(z) = softplus(logit(z))` (no per-modality trust schedule; v4 has only one modality). Eqs.:

- **(M1.1)** Evidence accumulation: `α_v^{t+1} = α_v^{t} + e_v^{t+1}`.
- **(M1.2)** Posterior class mean: `E[p_c | α_v] = α_{v,c} / Σ_k α_{v,k}`.
- **(M1.3)** Vacuity (open-set / OOD score, reused in M2 and M3): `m_u(v) = (C+1) / Σ_k α_{v,k}`.

EDL Dirichlet (Sensoy et al., NeurIPS-18) rather than S-BKI / ConvBKI kernel-Bayesian inference, because (i) we want per-voxel epistemic uncertainty *without* the spatial-kernel coupling that makes vacuity contaminated by neighbour evidence — the M3 decay rule and the RQ2 OOD score both need an honest per-voxel vacuity, and (ii) closed-form posterior keeps even an 8 GB 5060 viable for the dev path and trivialises on the 5090 server.

**v4 implementation note (W2-3 done).** The evidence accumulator is now a flat-tensor allocator that pre-allocates `α_v` in CUDA memory and indexes via a dense hash, replacing the dict-of-tensors prototype. Empirical: R2 baseline 10× faster, M1 60× faster, end-to-end pipeline 13× faster. This brings R2 frame latency from 663 ms (prototype) to ≈ 66 ms and M1 from 3 ms (already fast in prototype because of evidential closed-form) to functionally instantaneous. The 60× M1 speedup matters more for full-train scale than for the 100-frame preliminary number; it is the structural fix that makes full SemKITTI val tractable on a single 5090.

### 3.3 M2 — Confidence-aware loop closure + multi-session voxel-hash submap fusion (unchanged from v3)
Submaps sealed every 50 m or 30 s of trajectory. Per-submap descriptor:
- **(M2.1)** `d(S) = ( h_class(S),  h_entropy(S) )` where `h_class(S)` is the L1-normalised class histogram over confident voxels (vacuity below median), and `h_entropy(S)` is the histogram of per-voxel posterior entropy bucketed to 10 bins. Matching via cosine on `h_class` + Earth-Mover-Distance on `h_entropy`; verified by semantic-ICP restricted to class-consistent confident voxels.

Once a loop is verified, pose graph optimisation re-projects each voxel's `α_v` consistently across the loop; in submap overlap regions, `α_v` from different sessions are merged by **conjugate addition** `α_v ← α_v^{(s1)} + α_v^{(s2)}` (closed-form Dirichlet posterior over independent observations), so fusion is parameter-free.

Inter-session storage: voxel-hash key = `(int(x/v), int(y/v), int(z/v))`, value = `α_v`. New sessions write into the same hash; deduplication is implicit.

Inspired by: Kimera-Multi [C7] (pose graph + multi-robot), Hydra [A4] (per-submap descriptor), SA-LOAM [B4.4] (semantic-aided LCD), PlaneSDF [B4.2] (cross-session change detection). Distinct from all four because the *descriptor entropy channel* and the *parameter-free Dirichlet conjugate fusion* are both new.

### 3.4 M3 — Vacuity-driven voxel decay + map staleness ageing (unchanged from v3, preliminarily validated W3-B)
Conjugate exponential decay on `α_v`, but with the decay time-constant `τ` itself a function of vacuity:

- **(M3.1)** `α_v^{t+Δ} = ((α_v^{t} − 1) · exp(−Δ / τ(m_u))) + 1`, where `τ(m_u) = τ_max · (1 − m_u) + τ_min · m_u`, with `τ_max ≈ 3600 s` (one hour) for confidently-known voxels (low vacuity, ages slowly = persistent map) and `τ_min ≈ 60 s` (one minute) for high-vacuity voxels (ages fast = transient / unknown / dynamic stuff). This single equation is the v4 thesis (§1.4) made concrete: vacuity *is* the staleness signal.

The decay preserves the posterior mean (M1.2) while inflating vacuity over time, so stale voxels become re-writable as new evidence arrives. Submap-level ageing: a whole submap whose median vacuity exceeds a threshold is flagged for re-observation by an exploration policy (out of scope for this paper but the hook is present).

**v4 preliminary state (W3-B).** On 747 047 unique voxels at 0.25 m voxel size over seq 08 first 50 frames with Δt = 600 s, τ_min = 60 s, τ_max = 3600 s: bottom-10 %-vacuity voxels lose 16.0 % of evidence mass; top-10 %-vacuity voxels lose 42.3 %; ratio 2.64×. The same vacuity scalar that drives the M1 posterior update *also* drives the M3 decay rate at the rate predicted by Eq. M3.1. Json: `artifacts/m3_validation.json`.

Distinct from v1 §3.4 (which used a fixed `τ`) and from Voxblox / NvBlox (which use monotonically-growing weight with no decay at all, audit T2).

### 3.5 Implementation Details (terse, IROS budget; v4 backbone reality)
- **LiDAR semantic head — target:** Cylinder3D 55.85 M params, pretrained weights from authors' release. Fine-tune only the last layer to a (C+1)-D output (20-D closed-set; 15-D open-set) using EDL loss (Sensoy 2018) on SemanticKITTI train split. Two separate fine-tunes (closed-set and open-set heads). On the 5090 server, a last-layer fine-tune is roughly ½ day per head with backbone frozen.
- **LiDAR semantic head — preliminary stand-in:** PointNetVanilla 0.21 M params used in W3 to demonstrate method-level claims under matched-backbone conditions. NOT the IROS submission backbone.
- **LiDAR semantic head — v4 BLOCKER:** the planned Cylinder3D pretrained weights ship in spconv 1.x format; spconv 2.x (the only stack that supports sm_120 on cu130) has a kernel index iteration order change that gives all-NaN logits after weight permutation. Three unblock paths documented in §3.9; deadline 2026-06-01 (P1 +3 days).
- Backbone: NvBlox [C10] forked; `α_v` (20 × float32 = 80 B, or 15 × float32 = 60 B) replaces the 19-class label probability vector. Memory per voxel grows from ~100 B (R2) to ~180 B (v4 closed-set) or ~160 B (v4 open-set); §4.3 budgets this. 879 GB server disk and 251 GB RAM make voxel-hash growth from KITTI-360 multi-session a non-issue.
- LVIO state estimator: reused unchanged (any open-source LIO; we use the FAST-LIO2 release on KITTI-360, R3LIVE on SemanticKITTI single-session, doc'd in §4.6).
- Runs target: ≥ 10 Hz on 5090 dev, ≥ 5 Hz on Jetson Orin NX. Numbers to be measured at full backbone; deployment **benchmark + 30-s demo video** (v4 §4.6).

### 3.6 Key equations (closed-form, all derived in `/ars-full`)
- **Eq. M1.1** evidence accumulation
- **Eq. M1.2** posterior class mean
- **Eq. M1.3** vacuity (open-set / OOD score, used directly by RQ2)
- **Eq. M2.1** submap descriptor (class histogram + entropy histogram)
- **Eq. M3.1** vacuity-conditioned conjugate decay
Five equations total; v1/v2/v3 had five too. Same equation count, different semantics. **Eq. M1.3 verified W3-C (AUROC 0.8082); Eq. M3.1 verified W3-B (2.64× decay ratio).**

### 3.7 Dynamic-Object Handling Discussion (preserved from v3; sets up the Khronos comparison)
EvidLife-Map does **not** explicitly model 4D space-time as Khronos [A2] does. Khronos's contribution is a learned short-term motion segmenter combined with a long-term change detector; together they let it factorise dynamic objects from a persistent map.

v4 argues that **vacuity naturally degrades on moving-object voxels** for two structural reasons. (a) A moving object presents inconsistent evidence to the same voxel across consecutive frames (now-occupied, now-free, now-different-class), so per-voxel evidence accumulation under Eq. M1.1 collects conflicting `e_v` contributions, raising `Σ α_v` slowly relative to the inter-class disagreement — equivalently, vacuity stays elevated while the posterior class mean (M1.2) gets noisy. (b) The M3 decay rule (M3.1) under a high-vacuity regime sets `τ → τ_min ≈ 60 s`, so the dynamic voxel's evidence is actively flushed before it can crystallise into a wrong-persistent label.

The result: EvidLife-Map handles dynamic objects *implicitly* through (a) + (b), where Khronos handles them *explicitly* through a dedicated network. This is a fair-comparison framing — we are not claiming EvidLife-Map's implicit mechanism dominates Khronos's explicit mechanism on dynamic-object mIoU; we are claiming it is competitive enough to make the explicit machinery a deployment-cost trade-off rather than a capability necessity. The §4.4 Table V experiment measures this competitiveness directly.

This framing also addresses the "why don't you just bolt Khronos onto your system?" reviewer question: an explicit short-term motion segmenter would *override* vacuity's natural-instability signal in M3, doing the same job twice with two different parameter sets. We deliberately keep the implicit channel only, document the trade-off, and let numbers decide.

### 3.8 Methodological deltas, one line each (preserved from v3)
- vs **R2 [A1]:** specifies the Bayes filter (Eqs M1.1-M1.3 closed form); adds vacuity, adds decay, adds loop+fusion; adds explicit unknown channel. **Preliminary: M1 > R2 on mIoU 14.73 / 1.69 %, ECE 0.171 / 0.490, latency 3.0 / 663 ms at matched 0.21 M-param PointNet (W3-A vs W3-baseline).**
- vs **Khronos [A2]:** Khronos factorises short-/long-term in time via change-detection network; we let vacuity *be* the change signal via M3.1 (no separate network). **v3+v4 escalation: head-to-head experimentally on dynamic split (§4.4 Table V), not just argued in prose.**
- vs **Clio [A3]:** Clio compresses given a task; we keep a task-agnostic uncertainty-aware substrate. Clio could sit on top.
- vs **ConvBKI [C9]:** kernel-Bayesian spatial smoothing; we use EDL Dirichlet — sharper vacuity (no neighbour contamination), needed by M3.1 (preliminarily verified) and by RQ2 OOD (preliminarily verified).
- vs **LatentBKI [D10]:** open-vocab BKI; we are closed-set + open-set unknown channel, so we cleanly separate "known classes" and "unknown mass" rather than embedding everything in a CLIP-feature manifold.
- vs **OpenVox [A8/D8]:** Bernoulli per-instance; we are Dirichlet over all classes including unknown, and we add lifelong M2+M3.
- vs **S-BKI [C8]:** same spirit (probabilistic semantic voxel) but with explicit vacuity for decay; no kernel smoothing.
- vs **Voxblox / Voxblox++ [C1, C3]:** Voxblox-class does TSDF only; we add semantic Dirichlet + lifelong.
- vs **Kimera-Semantics [C6]:** Kimera does scene-graph + multi-robot; we focus on per-voxel calibration + single-robot multi-session lifelong.

### 3.9 Backbone Resolution Path (NEW in v4 — the W2-1 blocker)

The planned Cylinder3D backbone (55.85 M params, sparse-convolution voxel-cylinder architecture, the de-facto SOTA for SemanticKITTI LiDAR semantic segmentation as of 2024) ships its pretrained weights in **spconv 1.x** format. On the 5090 server (sm_120 + CUDA 13.0), only **spconv 2.x** (specifically `spconv-cu126 2.3.8`) compiles and runs. W2-1 engineering has applied a structural port (`W2-1_status.md` lines 5-14, 27-65):

- Import patches (`import spconv.pytorch as spconv`)
- 36 `.features = X` → `.replace_feature(X)` rewrites
- 15 `indice_key` uniqueness patches (spconv 2.x enforces same-kernel-size-per-indice-key invariant)
- Weight permutation (kD,kH,kW,in,out) → (out,kD,kH,kW,in) for 48 conv layers
- torch_scatter.scatter_max → torch.scatter_reduce_
- 9-dim voxel input (dxyz_pol + xyz_pol + xy_cart + intensity)
- Volume bounds aligned to semantickitti.yaml

After these structural patches, model loads (0 missing, 0 unexpected keys, 55.85 M params) and forward pass runs without API error. **But the output logits are all-NaN across all 46k non-zero voxels.** Root cause: spconv 1.x → 2.x changed the *kernel index iteration order* in addition to the weight layout. The shape is correct after permute(4,0,1,2,3) but the *value semantics* of each kernel element are different — same shape, different mapping of voxel-offsets to weight elements. A pure tensor permutation cannot fix this; the change is in C++ kernel-iteration code, not in the tensor-shape contract.

**Three unblock paths (decision required by 2026-06-01, P1 +3 days; v4-revised 2026-05-29):**

- **Path 1 — Build spconv 1.x from source against torch 2.11+cu130.** Effort: 2-4 hours engineering. Risk: torch 2.11 internals may have moved beyond what spconv 1.x's CUDA-C++ code expects (new dispatcher signatures, removed at::cuda::* APIs). If the build succeeds, the pretrained weights load cleanly with no further patching and we get full Cylinder3D fidelity. If it fails (60 % probability based on the rough analogues of other spconv 1.x revivals on torch 2.x), we lose 4 hours. **Risk graded M-H (new R-21 in §6).**
- ~~**Path 2 — Adopt PVKD.**~~ **RULED OUT** (inspection 2026-05-29, supplementary `backbone_path_findings.md`): PVKD's `requirements.txt` pins `spconv==1.2.1`, so it inherits the exact same value-semantic kernel-index gap as the original Cylinder3D checkpoint. PVKD's contribution is *knowledge distillation* of Cylinder3D into smaller variants (1.0x → 71.8 mIoU, 1.5x → 72.4 mIoU), not a spconv 2.x port. Adopting PVKD does not avoid Path 1.
- **Path 3 — Switch to MinkUNet / WaffleIron / RandLA-Net.** Effort: 2-3 days to swap backbone. Risk: mIoU drop of approximately 5-10 (estimate) versus Cylinder3D published 65-68 mIoU on SemKITTI test, leaving us in the 55-60 mIoU band. Reproducibility benefit: every alternative listed is pure-PyTorch (no spconv), so the IROS supplementary becomes trivially reproducible on any sm_120 box. **Risk graded L on engineering, M on mIoU position.**
- **Path 4 (NEW) — Extend our existing PointNet2Lite to a small RandLA-Net-class backbone.** Effort: 1-2 days (existing `pointnet2_lite.py` provides one kNN-context block; add a second / third dilated block with attentive pooling). Currently 17.92 % mIoU at 0.28 M params on the multi-seq audit (§IV.0); target 25-35 % mIoU at ~1-2 M params. Pure-PyTorch (no spconv), so reproducibility is trivial on sm_120. **Risk graded L on engineering; mIoU floor below Cylinder3D-class but matches the §IV.0 backbone-floor narrative the paper already uses.**

**v4-revised preferred order: Path 4 (extend lite) → Path 1 (spconv 1.x source build) → Path 3 (alt sparse-conv backbone).** Rationale: Path 4 is the highest-leverage now (we already have PointNet2Lite producing 17.92 % mIoU; extending one stage is mechanical and gets us a defensible 25-35 % mIoU backbone with no external dependencies). Path 1 in parallel as the high-fidelity continuation if it cooperates with torch 2.11+cu130. Path 3 as the heavyweight fallback if both stall.

**Open question to user (§8 Q1):** select preferred path order, or delegate to architect.

**Decision deadline:** **2026-06-01** (P1 day 3). If a path is not selected by this date, the architect default-selects Path 2; if PVKD inspection on 2026-06-01 fails (code stale, dependencies broken, licence-incompatible), the architect default-falls to Path 1; if Path 1 fails by 2026-06-03, the architect default-falls to Path 3 and adjusts §5 C3 mIoU targets per Path-3-haircut estimates.

This sequencing is intentionally biased toward action over deliberation: every day of P1 not spent on a real backbone is a day deducted from the §7 P4 buffer.

---

## §4 Experimental Plan (v4)

### 4.1 Datasets (final shortlist, 6 datasets — unchanged from v3, with v4 server-side status)

| # | Dataset | Role in v4 | v4 server-side status | Why kept / why added |
|---|---------|------------|------------------------|----------------------|
| D-a | **SemanticKITTI** (Behley 2019; community Cylinder3D split) | Primary closed-set dense GT for LiDAR mapping mIoU and ECE (RQ1); base substrate for v4 open-set 14/5 split (RQ2); base substrate for dynamic split (RQ5) | **Train sequences 00-07, 09, 10 on server `/data/shared/SemanticKITTI/dataset/sequences/`: 19 230 velodyne frames + labels.** Seq 08 (val): first 100 of 4071 frames done; full transfer in progress | Standard, reproducible, every cited competitor has results on it |
| D-b | **nuScenes-LiDARSeg** (Caesar 2020; LiDAR seg labels released 2021) | Cross-domain mIoU/ECE check (RQ1); reverse cross-domain OOD cross-check for RQ2 | Not on server yet; transfer planned P1 W2 | Only mainstream AD dataset with LiDAR semantic GT + different taxonomy |
| D-c | **KITTI-360** (Liao 2022; multi-session dense LiDAR semantic seg, 19 classes) | Primary lifelong / multi-session evaluation arena (RQ4) | Not on server yet; transfer planned P1 W1 (gating step for RQ4 + leg ii of "three jobs" matrix) | Multi-session structure with dense semantic GT |
| D-d | **SemanticKITTI dynamic split** (constructed from SemanticKITTI dynamic-label flag on seq 00, 04, 05, 07; frames with high fraction of `moving-*` class instances) | Dynamic-scene head-to-head vs Khronos (RQ5) | Inherits D-a transfer; selection script to be run when seq 00, 04, 05, 07 train labels are server-side (✓ as of v4) | SemanticKITTI's per-point dynamic labels make this reproducible without extra annotation |
| D-e | **SemanticSpray** [B3.9] (RA-L-24 wet-road LiDAR seg) | Passive-replay traversability arena (RQ3) | Not on server yet; transfer planned P3 | Real-world LiDAR-only, no radar dependence, no manual annotation needed |
| D-f | **Robo3D-corrupted SemanticKITTI** (Kong 2023, ICCV-23) | Fallback substrate for RQ2 if open-set split is blocked (not primary in v4) | Generation script only; will run from D-a when needed | Fallback path retained |

**Open-set split protocol (v4 RQ2 primary, preliminarily validated W3-C).**
- **Primary split:** SemanticKITTI 19 → 14 known + 5 unknown. Withheld: `bicyclist`, `motorcyclist`, `truck`, `other-vehicle`, `other-ground`. Selection rationale: spans static + dynamic, vehicle + ground, ≥ 500 GT points per validation sequence each. **Preliminary AUROC 0.8082 at this split (W3-C).**
- **Robustness split:** SemanticKITTI 19 → 16 known + 3 unknown (drop only `bicyclist`, `motorcyclist`, `other-vehicle`).
- **Reverse cross-domain split:** train on SemanticKITTI 14 known; evaluate vacuity on nuScenes-LiDARSeg classes that do not overlap.

**SemanticKITTI dynamic split protocol (v4 RQ5, unchanged from v3).**
- Construction: from seq 00, 04, 05, 07, select all frames where ≥ 5 % of GT points carry a `moving-*` class label. Provisional yield ~1500-2000 dynamic-heavy frames.
- Metrics: dynamic-object mIoU, static-region recall, end-to-end latency.
- Khronos comparison per §4.2 baseline #5; fallback to published numbers per R-17.

**Datasets explicitly NOT used and why (unchanged from v3):**
- **K-Radar** — dropped (zero annotation budget).
- **FusionPortable** — dropped (user instruction).
- **Boreas [B3.4]** — sparse semantic GT; KITTI-360 covers the multi-session need.
- **ACDC [B3.3]** — image-only.
- **CADC [B3.5]** — no dense semantic GT.
- **Replica / TUM / ScanNet** — indoor.

### 4.2 Baselines (6 total — unchanged from v3 except for v4 compute footnote)

1. **R2-reimpl.** Faithful Python/CUDA reimplementation of R2 [A1] (NvBlox + Cylinder3D + argmax-Bayes). v4 status: prototype on PointNetVanilla backbone executes (W2-4 done); full Cylinder3D version gated on §3.9 backbone resolution.
2. **NvBlox-vanilla + per-frame argmax.** Lower bound (no temporal fusion).
3. **ConvBKI [C9].** Authors' code; canonical probabilistic semantic voxel competitor (lit_scan reviewer-threat #3). **v4: R-12 demoted from H to M** because the 5090 makes reproduction cheap (≈ 4 GPU-hr instead of multi-day on 4060).
4. **Kimera-Semantics [C6].** Authors' release; canonical Voxblox-class baseline.
5. **Khronos [A2]** — used for both (i) RQ1 closed-set mIoU on SemanticKITTI AND (ii) RQ5 dynamic-scene head-to-head on the dynamic split. **v4: R-17 risk unchanged** (Khronos reproduction is its own undertaking, server compute does not eliminate code-quality unknowns).
6. **Clio-LiDAR-stub** [A3].

Stretch baseline (not Go criterion): **OpenVox [A8/D8]** if their code drops by P3.

### 4.3 Metrics (unchanged from v3)

| Tier | Metric | Used in RQ | Notes |
|------|--------|-----------|-------|
| Mapping | mIoU, per-class IoU, F@5 cm reconstruction, voxel coverage, GPU memory peak, end-to-end latency (mean + 99-pct) | RQ1, RQ4, RQ5 | closes audit E4/E5 |
| Uncertainty (closed-set) | Expected Calibration Error (ECE), Brier score | RQ1, RQ4 | distinguishes us from R2 and ConvBKI |
| Open-set (RQ2 primary) | AUROC of vacuity, AUPR for unknown-positive class, closed-set mIoU on known subset | **RQ2 lead** | preliminary AUROC 0.8082 verified W3-C |
| Corruption (fallback) | per-corruption mIoU drop, mCE, voxel-level AUROC of vacuity | RQ2 fallback only | analysis pipeline shares AUROC code with primary |
| Dynamic (RQ5) | dynamic-object mIoU, static-region recall, Jetson latency | RQ5 | head-to-head vs Khronos |
| Traversability | safe-region recall, false-traversable rate, deferral rate | RQ3 | passive replay on SemanticSpray |
| Lifelong | stale-voxel removal precision/recall, ECE drift, map size growth | RQ4 | KITTI-360 multi-session split; preliminary intra-seq A/B-half stand-in P=0.77 R=0.52 |

**Pseudo-GT?** Not needed in v4 (same as v3). Every dataset has dense semantic GT or is a fallback.

### 4.4 Ablations (v4 — unchanged from v3 except for "all firm" rationale)

| # | Variable | Question answered | Compute cost on 5090 (est.) | Firmness |
|---|---|---|---|---|
| A-1 | ± Dirichlet evidential head (vs argmax-Bayes / vs softmax-Bayes) | RQ1: does evidential improve mIoU + ECE? | 3 × 0.5 day = 1.5 day | firm |
| A-2 | ± vacuity-conditioned decay τ(m_u) (vs fixed-τ / vs no-decay) | RQ4 + M3 isolation | 3 × 0.4 day = 1.2 day | firm |
| A-3 | ± entropy channel in submap descriptor | RQ4 + M2 isolation | 2 × 0.4 day = 0.8 day | firm |
| A-4 | ± vacuity-aware traversability | RQ3 | 2 × 0.1 day = 0.2 day | firm |
| A-5 | ± conjugate Dirichlet fusion across sessions | RQ4 + M2 isolation | 2 × 0.25 day = 0.5 day | firm |
| A-7 | ± openset channel in M1 (15-D vs 14-D under open-set split) | RQ2 lead | 2 × 0.5 day = 1 day | firm |
| A-6 | ± voxel size (0.10 m vs 0.25 m) | defensive | 2 × 0.5 day = 1 day | **firm in v4** (cloud-cost reason for dropping in v4-delta gone; 5090 makes the run cheap) |

**Total ablations (A-1…A-7) on 5090:** ≈ 6.2 wall-clock days. **v4 promotes A-6 back to firm** because the v4-delta cloud-cost rationale for dropping it disappears with free server access.

**Plus main eval + Khronos + open-set:** an estimated ~10 wall-clock days on a single 5090 for all baselines + primary RQ runs. With 4 cards, parallel ablation execution can compress this further; the relevant metric is wall-clock to first-complete-table, not raw GPU-days.

### 4.5 Compute budget (v4 reality)

- **Dev / smoke / debug / Jetson cross-compile / paper writing:** local 5060 Laptop (8 GB). Batch ≤ 1 fp16 for sanity checks only. NOT used for any reported number in the IROS paper.
- **Training and main eval:** **5090 server `server@100.64.0.5` (Tailscale): 4 × RTX 5090 32 GB sm_120, CUDA 13.0, 80 cores, 251 GB RAM, 879 GB disk; workspace `~/Documents/yping/mapping/code/`.** Single 5090 ≈ 4× a 4060 on transformer-style ops, ≈ 3× on sparse-conv ops; the v3 ablation grid's 64 GPU-days on 4060 becomes ≈ 16 wall-clock days on one 5090. With 4 cards, ablations parallelise to ≈ 4-5 wall-clock days.
- **Deployment benchmark + demo video:** 1× Jetson Orin NX (16 GB) — §III.E latency + memory + 30-second supplementary video capture.
- **Estimated wall-clock budget across the 9-month re-anchored timeline (v4):**
  - Cylinder3D backbone resolution (§3.9, Path 2 expected): 1-2 days.
  - Cylinder3D last-layer EDL fine-tune on SemanticKITTI (×2: 20-D closed and 15-D open-set): ≈ 0.5 day × 2 = 1 day.
  - R2-reimpl full bring-up on Cylinder3D: 1-2 days (W2-4 prototype already validated, full version is the last-layer swap).
  - ConvBKI baseline reproduction: ≈ 0.5-1 day (5090 makes it cheap; R-12 demoted to M).
  - Kimera-Semantics baseline: 1 day.
  - Khronos closed-set RQ1 run on SemanticKITTI: ≈ 1 day on 5090.
  - Khronos dynamic-split RQ5 run: 2-3 days (reproduction + experiment).
  - Main mapping eval over SemanticKITTI seq 08+11-21 per system: ≈ 0.25 day × 6 systems = 1.5 days.
  - Open-set RQ2 eval (v4 lead): ≈ 1 day (two splits + nuScenes reverse-check across our system + 3 baselines).
  - KITTI-360 multi-session eval (8 sequences, ≥ 5 revisit pairs): ≈ 2 days.
  - SemanticSpray passive replay (RQ3): ≈ 0.5 day.
  - Ablations: ≈ 6 wall-clock days firm (§4.4 above), parallelisable across 4 cards.
  - Jetson Orin NX deployment latency / memory check + 30 s demo video capture and edit: ~4 days (Jetson is single-device, not parallelisable).
  - **Total: ≈ 21-25 wall-clock days of server time** (vs v3's 64 GPU-days on 4060). At 5 productive days per calendar week, that's ~5 weeks of pure runtime + 1 week buffer for re-runs, fitting comfortably in the P1-P4 window of §7 (≈ 9-12 weeks).

### 4.6 Embedded Real-Time Demo (preserved from v3)

- **Platform:** Jetson Orin NX (16 GB), TensorRT FP16 build of the EvidLife-Map inference path; the M1 head and the M3 decay run on GPU, M2 loop closure runs on CPU.
- **Data source:** SemanticKITTI seq 08 offline replay over rosbag at native 10 Hz (or SemanticSpray rosbag if the wet-road condition is more visually informative; choice in P4).
- **Duration:** 30 s continuous (300 frames @ 10 Hz).
- **Output:** real-time metric-semantic voxel map with the traversability overlay (RQ3 output) and a vacuity heat-map overlay showing high-vacuity voxels in a distinct colour; recorded as an MP4 with annotated overlay timestamp.
- **Use in paper:** referenced from §IV.G (sub-section "F. Embedded Demo") and submitted as supplementary `evidlife_demo.mp4`. Promotes C4 deployability claim from "benchmark only" to "benchmark + visible demo".

### 4.7 Reproducibility hygiene (closes audit E7)

- Code + Docker + ROS 2 launch + EDL fine-tune script (both 20-D and 15-D variants) + open-set split definition file + dynamic split selection script + Robo3D corruption applier (kept for fallback path) + KITTI-360 revisit-pair builder script + per-experiment seed list + hyperparameter table + Jetson Orin NX demo capture script.
- LVIO choice documented per dataset: FAST-LIO2 on KITTI-360, R3LIVE on SemanticKITTI single-session, public LIO config on SemanticSpray.
- **v4-new:** the W2-1 backbone-resolution path (§3.9) selected and its consequences for reproducibility (e.g., spconv 1.x source build instructions; PVKD repo pin; alternative-backbone training script) are part of the supplementary `reproduce.yaml`.

---

## §5 Expected Contributions (v4, 4 contributions — text essentially unchanged from v3, with v4 preliminary-evidence footnote)

- **C1 (Algorithm).** A Dirichlet-evidential per-voxel posterior whose single vacuity scalar simultaneously serves three downstream jobs — (i) open-set / OOD detection for unknown-category voxels (v4 RQ2 lead; Eq. M1.3; **preliminarily verified W3-C, AUROC 0.8082**), (ii) submap descriptor entropy channel (Eq. M2.1), and (iii) lifelong decay trigger (Eq. M3.1; **preliminarily verified W3-B, 2.64× decay ratio**). Strictly generalises R2's unspecified Bayes filter (audit S1/S2 closed) and adds the open-set channel (audit S5 closed). The "one scalar, three jobs" framing — with two of three jobs preliminarily validated on real KITTI data — is the v4 novelty hook.

- **C2 (System).** A complete online metric-semantic mapping system EvidLife-Map with confidence-aware loop closure, parameter-free conjugate Dirichlet inter-session submap fusion, and vacuity-conditioned voxel decay — closing audit T1/T2/T3/T4 in one paper. Dynamic-scene head-to-head against Khronos demonstrates that the implicit vacuity-instability mechanism is competitive with explicit 4D modelling at far lower latency on Jetson Orin NX (v4 RQ5; see C3 (v)). Deployable on Jetson Orin NX (§III.E target ≥ 5 Hz, plus 30-s supplementary demo video). First system to ship the lifelong + open-set + calibration + dynamic-competitive quartet together on a public LiDAR-only stack.

- **C3 (Empirical).** Across SemanticKITTI, nuScenes-LiDARSeg, KITTI-360, SemanticKITTI dynamic split, and SemanticSpray: (i) ≥ +2 mIoU and ≥ −20 % ECE over a faithful R2-reimpl on SemanticKITTI (RQ1 H1; **preliminary ordering ✓ at 0.21 M-param backbone**); (ii) AUROC ≥ 0.80 and AUPR ≥ 0.60 for vacuity-as-OOD-detector on the SemanticKITTI 14/5 open-set split, with closed-set mIoU dropping by ≤ 1.5 (RQ2 H2 primary; **preliminary AUROC 0.8082 ✓ at 0.21 M-param backbone**); (iii) ≥ +10 pp safe-region recall on SemanticSpray passive replay (RQ3 H3); (iv) ≥ 80 % stale-voxel removal precision and ≤ 1.5× ECE drift on KITTI-360 multi-session (RQ4 H4; preliminary intra-seq A/B-half stand-in: P=0.77, R=0.52 — directional evidence, not a PASS); (v) dynamic-object mIoU within 3 of Khronos and static recall within 1 pp, at ≥ 5 Hz on Jetson Orin NX (RQ5 H5). All numbers replaced by measured ones before submission; targets set as the §9 Go/No-Go thresholds. **Preliminary numbers in `artifacts/preliminary_results.md` will be replaced by full-backbone numbers from §3.9-resolved Cylinder3D runs.**

- **C4 (Reproducibility + Deployability).** Code + Docker + ROS 2 launch + Jetson Orin NX deployment benchmark plus 30-s online-mapping demo video in supplementary material — directly attacks R2 audit E7. The deployability claim, previously optional in v2, is firm in v3/v4 ("first lifelong evidential MSM that runs at ≥ 5 Hz on a 16 GB Orin NX, demonstrated on supplementary video"); hardware-grounded and visually verifiable. **v4-new:** supplementary also includes the §3.9 backbone-resolution path documentation, so reviewers can reproduce on any sm_120 host without re-encountering the spconv 1.x ↔ 2.x stuck.

---

## §6 Risk Register (v4 — 16 risks)

| ID | Risk | Prob | Impact | Mitigation | Trigger to fall back |
|----|------|------|--------|------------|---------------------|
| R-2 (kept) | Khronos releases a multi-session / lifelong extension before IROS 2027 submission | M | H | Track arXiv weekly via post-research literature monitor; pre-register our differentiation as "vacuity-coupled decay, no separate change-detection network" | If Khronos-v2 lands on lifelong before us, sharpen claim to the "one vacuity, three jobs" calibration story |
| R-3 (kept) | GS-LIVO [B2.7] HKUST sibling lab releases semantic-GSplat extension | M | M | Differentiate on representation (voxel-Dirichlet vs Gaussian) and target (lifelong vs photo-realistic single-session) | If GS-LIVO-semantic ships, sharpen our "Jetson Orin NX deployable lifelong with demo video" angle |
| R-4 (kept) | Dirichlet evidential fusion fails to beat ConvBKI in mIoU at full backbone scale | M | H | Calibration (ECE / AUROC unknown) is a separate axis; we win on calibration + lifelong even if mIoU ties. **v4 preliminary: M1 > R2 ordering holds at preliminary scale, but ConvBKI is a stronger comparator than R2-reimpl; full-scale rerun still required.** | If ECE also ties at full scale, retreat per §0 fallback to calibration-only paper at RA-L |
| R-6 (kept) | KITTI-360 multi-session revisit overlap insufficient for RQ4 | M | M | Pre-compute overlap matrix in P1 (gated on dataset transfer); if pairs < 5, augment with SemanticKITTI cross-day sequences | If still < 5, demote RQ4 H4 to single-session ECE-drift study |
| R-7 (kept) | Voxel-size confound contaminates mIoU vs ConvBKI | M | M | Ablation A-6 (firm in v4) covers this | n/a |
| R-8 (kept) | Cylinder3D licence prevents code release | L | M | RangeNet++ or PVKD (§3.9 Path 2) as backup; train from scratch on 5090 (~3 days) | n/a |
| R-9 (kept, lowered) | 8-page IROS overflow given 5 RQs and 5 datasets | M (down from H in v3) | M | Move ablation tables and KITTI-360 per-sequence numbers to supplementary; demo video lives in supplementary | Drop A-6 from main, defend "two-pair" RQ4 demo, push RQ5 details to supplementary table. **v4 lowers R-9 to M because the preliminary results give us a more compact §IV.0 "Preliminary Verification" pre-table that absorbs some space pressure rather than adding to it.** |
| R-10 (kept) | Reviewer demands real-robot deployment | M | M | Cite scope as "methodological + public-dataset + Jetson Orin NX benchmark + 30-s demo video" | If desk-rejected, the Jetson video + numbers become the primary deployability evidence |
| R-11 (REWRITTEN in v4) | **5060 Laptop 8 GB local-VRAM is insufficient for end-to-end SemKITTI training at batch ≥ 1 even with fp16** | H (up from H in v3) | **L (down from M in v3)** | Local 5060 is dev-only in v4; training runs on 5090 server; impact is bounded to "we cannot debug on flight, only on the bench" | Server access loss (unlikely; user has continuous Tailscale link) would require fallback to commercial cloud per pre-v4-delta plan |
| R-12 (DEMOTED in v4) | ConvBKI authors' reimpl fails to reproduce published numbers | M | **M (down from H in v3)** | 5090 server makes ConvBKI reproduction cheap; allocate ~1-2 days; pin CUDA + PyTorch versions; cross-check with published seq-08 mIoU within ±2 | If reproduction fails after 1 week of focused effort, use published numbers in a quoted-only table |
| R-13 (kept) | KITTI-360 multi-session revisit length / overlap is too short to demonstrate stale-voxel decay | M | M | Pre-compute overlap matrix when KITTI-360 arrives on server; augment with synthetic temporal-gap injection on single sessions | Demote to "synthetic-revisit study only" and shrink H4 from precision-recall to AUROC of stale-voxel ranking |
| R-14 (CONFIRMED in v4) | **EDL Dirichlet head training is unstable / collapses to uniform Dirichlet** — **OBSERVED at preliminary scale (best AUROC at ep 1, degrades to ~0.67 by ep 30)** | **CONFIRMED** | **H** | Use late-stage KL warm-restart schedule (re-introduce KL after annealing-to-zero, with a smaller plateau target); gradient-clip; monitor evidence-sum trajectory; warm-start from softmax pretrain; best-ckpt selection by val AUROC rather than val loss | If full-backbone training also collapses irrecoverably, switch to posterior network (PostNet, Charpentier 2020). **v4: this risk is now confirmed at preliminary scale and is the primary EDL methodological risk to engineer around in P2.** |
| R-15 (kept) | Open-set 14/5 split definition is methodologically controversial | M | H | Pre-register the split before any RQ2 experiment runs; report both N=5 primary and N=3 robustness; add nuScenes reverse-cross-domain check; release split file in supplementary. **v4: the preliminary AUROC 0.8082 at this split is one evidence point that the split is not trivially separable; the W3-C run did *not* observe degenerate AUROC=1.0 or AUROC<0.5.** | If reviewer's preferred split is starkly different, retreat to "AUROC across multiple splits, mean ± std" framing |
| R-16 (kept) | SemanticKITTI lacks taxonomic diversity for a meaningful 5-class unknown | M | M | Validate split quality by P2 with a 2-NN baseline check on raw features; nuScenes reverse cross-domain check provides backup; Robo3D fallback (D-f) retained | If both SemanticKITTI splits and nuScenes cross-domain fail to produce informative AUROC by P3, switch RQ2 to Robo3D corruption fallback |
| R-17 (kept) | Khronos reproduction fails for the dynamic-split RQ5 experiment | H | H | Allocate P3 firmly to Khronos bring-up (gated by G-6); pin docker environment; reach out to Khronos authors; fallback to published numbers in Table V hybrid | If full reproduction fails, ship Table V as "published numbers vs ours" hybrid; H5 weakens to "competitive with published Khronos numbers" |
| **R-21 (NEW in v4)** | **spconv 1.x ↔ 2.x value-semantic gap (kernel index iteration order change) blocks the planned Cylinder3D backbone.** Path 1 (source-build spconv 1.x against torch 2.11+cu130) may fail vs new PyTorch internals (estimated 60 % failure probability) | **H** | **H** | §3.9 documents three unblock paths; default order Path 2 → 1 → 3; decision deadline 2026-06-01; if all three fail, default-fall to MinkUNet and adjust §5 C3 mIoU targets per Path-3 haircut | Path 3 (alt-backbone) is the deterministic-success path with a 5-10 mIoU haircut; it is guaranteed unblockable, so this risk is *time-bounded* not catastrophic |
| **R-22 (NEW in v4)** | **PVKD (§3.9 Path 2) code quality, training-script completeness, or licence terms turn out to be incompatible with our reproducibility commitment** | M | M | Inspect repo in P1 (≤ 1 day budget); if any of (code stale, deps broken, licence-incompatible), fall to Path 1; if Path 1 also fails, fall to Path 3 | Path 2 failure cascades to Path 1 then Path 3 per §3.9 sequencing |

**H/M/L summary (v4):** **H: 5** (R-4, R-14, R-15, R-17, R-21 — algorithmic-fail + EDL stability + open-set validity + Khronos-repro + backbone-blocker are now the structural risks; v4 raised R-21 to H, R-14 to confirmed-H, but lowered R-9 to M and R-11 impact to L). **M: 9** (R-2, R-3, R-6, R-7, R-9, R-10, R-12, R-13, R-16, R-22 — that is 10 entries; revised: R-9 and R-11 each one of these tiers; correct M count: R-2, R-3, R-6, R-7, R-9, R-10, R-12, R-13, R-16, R-22 = 10). **L: 2** (R-8, R-11-impact). Adjusted summary: **5 H + 10 M + 1 L = 16 total.** Counting note: R-11's probability is H but impact is L; it lives in the "L impact" bucket for prioritisation purposes. (v3 had 14 risks: 5 H + 8 M + 1 L.)

(v1 risks R-1/R-5 retired in v2. v2 risks R-2..R-14 carried into v3 with wording updates. v3 risks R-15/R-16/R-17 carried into v4. v4 adds R-21/R-22 and confirms R-14.)

---

## §7 Timeline (re-anchored, ≈ 9 months from 2026-05-29 to IROS 2027 deadline ≈ 2027-03)

Reference today = **2026-05-29**; IROS 2027 paper deadline ≈ 2027-03; runway ≈ 39 weeks. The v3 W1-W39 schedule was anchored to 2026-05-28 and assumed a 4060-class compute model; v4 re-anchors to 2026-05-29, replaces 4060 with 5090 server, and absorbs the W3 preliminary verification done in 1 wall-clock day. The result is a shorter critical path (compute is faster) and an earlier first-real-results milestone (W3 work landed *before* the v3 W3 gate). v4 reflects this with a phase-based plan rather than a week-based plan, because the 5090 turnaround makes per-experiment scheduling more elastic than v3 assumed.

| Phase | Wall-clock from 2026-05-29 | Calendar (approx.) | Focus | Concrete output / milestone |
|-------|----------------------------|---------------------|-------|-----------------------------|
| **P1 Backbone Unblock + Full-Scale RQ1/RQ2** | now → +2 weeks (through 2026-06-12) | 2026-05-29 → 2026-06-12 | §3.9 Cylinder3D backbone resolution (Path 2 default deadline 2026-06-01; Path 1 cascade by 2026-06-03; Path 3 cascade by 2026-06-05); EDL R-14 KL warm-restart schedule implemented; scale-up full-train EDL Cylinder3D head; re-run RQ1 (closed-set mIoU + ECE) and RQ2 (open-set AUROC + AUPR + closed-set mIoU on knowns) at full SemKITTI val on 5090 server; KITTI-360 dataset transfer to server kicked off in parallel | P1 deliverable: full-backbone RQ1 + RQ2 tables, replacing the preliminary numbers in paper §IV.0. **G-1 (formal) + G-2 (formal) + G-5 (formal) checkpoints at end of P1.** |
| **P2 RQ4 Lifelong (KITTI-360 + Multi-Session)** | +2 → +4 weeks (through 2026-06-26) | 2026-06-12 → 2026-06-26 | M2 confidence-aware loop closure implementation; M2 + M3 integrated with submap fusion; KITTI-360 revisit-overlap matrix run on transferred data; RQ4 H4 evaluation on ≥ 5 revisit pairs; intra-session A/B-half preliminary stand-in retired in favour of true multi-session numbers | P2 deliverable: RQ4 lifelong table with stale-voxel P/R, ECE drift, memory footprint across multi-session. **G-3 (formal) + G-4 (formal) checkpoints at end of P2.** |
| **P3 RQ3 Traversability + RQ5 Dynamic Head-to-Head** | +4 → +6 weeks (through 2026-07-10) | 2026-06-26 → 2026-07-10 | SemanticSpray RQ3 passive-replay traversability eval; Khronos baseline reproduction (R-17 mitigation; fallback to published numbers if needed); RQ5 dynamic-split head-to-head experiment; SemanticKITTI dynamic-split construction; Clio LiDAR stub for open-set comparison | P3 deliverable: RQ3 + RQ5 tables. **G-6 (formal) checkpoint at end of P3.** |
| **P4 Ablations + Robustness + Jetson Deployment + Demo Video** | +6 → +9 weeks (through 2026-07-31) | 2026-07-10 → 2026-07-31 | All seven ablations (A-1…A-7) firm on 5090 with parallel execution across 4 cards; nuScenes-LiDARSeg cross-domain RQ1 and reverse OOD RQ2 checks; Robo3D fallback path validated and held in reserve; Jetson Orin NX deployment latency + memory benchmark; 30-s demo video capture and edit | P4 deliverable: full ablation tables + cross-domain numbers + Jetson row + demo MP4. End of P4 ≈ +9 weeks from 2026-05-29 = ~2026-07-31; this consumes ~9 weeks of the 39-week runway, leaving ~30 weeks for P5 writing/submission. |
| **P5 Writing + Reviewer-Sim + Submission** | +9 weeks → 2027-03 deadline | 2026-08 → 2027-03 | `/ars-full` first draft starting from existing paper_v2.md (already at §IV with §IV.0 preliminary box); figures, tables, supplementary including demo video; first internal reviewer-sim pass (`academic-paper-reviewer`); revision; second reviewer-sim pass; format-convert to IROS LaTeX; camera-ready prep; final submission ≈ 2027-03 | P5 deliverable: submitted IROS 2027 paper + supplementary + Jetson demo MP4. ~30 weeks of writing/iteration runway is **substantially more than v3's 6+ weeks** because experiments collapse from ~28 weeks (W4-W26 in v3 at 4060 speeds) to ~7 weeks (P1-P4 in v4 at 5090 speeds). The 5090 speedup is the load-bearing change that buys the buffer. |

**Total: ≈ 39 weeks** (same horizon as v3). Distribution shifted: **experiments compressed from W1-W26 (~26 weeks) to P1-P4 (~9 weeks)** thanks to (a) preliminary verification already done, (b) 5090 server compute, (c) flat-tensor evidence accumulator from W2-3 making per-frame processing 13× faster end-to-end. **Writing window expanded from W27-W39 (~12 weeks) to P5 (~30 weeks)**, which absorbs (a) Khronos reproduction risk slip, (b) backbone-blocker recovery if Paths 1/2 fail and Path 3 requires re-validation, (c) reviewer-sim iteration including potential paper restructure if reviewer feedback is severe.

**First v4 milestone (P1 day 3 = 2026-06-01):** §3.9 backbone resolution path selected (Path 1 vs Path 2 vs Path 3) and unblock work started.

**Second v4 milestone (P1 end = 2026-06-12):** full-scale RQ1 + RQ2 tables on Cylinder3D-class backbone, replacing the §IV.0 preliminary numbers in paper_v2.md.

**Risk-adjusted view.** If Path 1 and Path 2 both fail and we cascade to Path 3, P1 slips by ≈ 3-5 days (Path 3 backbone swap takes 2-3 days plus re-training overhead). This still fits inside the 2-week P1 budget. The buffer between P4 end (~2026-07-31) and P5 first reviewer-sim deadline (~late 2026-09) is ample — ~8 weeks of slack against P4 risk.

---

## §8 Open Questions for the User (v4 — two new)

v2 had three open questions on (RQ2 lead, Khronos comparison scope, Jetson deployment depth); all resolved 2026-05-28 and closed in v3. v3 had zero open questions. **v4 reopens two new questions driven by the §3.9 backbone blocker and the R-14 confirmation.**

- **Q1 — Backbone resolution path (deadline 2026-06-01).** §3.9 documents three unblock paths for the Cylinder3D + spconv 1.x ↔ 2.x mismatch:
  - Path 1 (source-build spconv 1.x against torch 2.11+cu130; 2-4 hr engineering; ~60 % failure probability vs torch 2.11 internals).
  - Path 2 (adopt PVKD, the Cylinder3D successor reported to be spconv 2.x-native; 1 day to integrate if repo inspection passes).
  - Path 3 (switch to MinkUNet / WaffleIron / RandLA-Net pure-PyTorch backbone; 2-3 days; ~5-10 mIoU haircut vs Cylinder3D position).
  - **Architect default:** Path 2 → Path 1 → Path 3. If user has no preference by 2026-06-01, architect proceeds with this order.
  - **Asked of user:** confirm Path 2 first, or override to a different sequencing, or specify a hard preference (e.g., "Path 3 immediately because reproducibility outweighs mIoU position").

- **Q2 — EDL stability fix priority (P1 vs P2).** R-14 is confirmed at preliminary scale: best AUROC at epoch 1, degrades to ~0.67 by epoch 30 due to KL annealing pushing the head toward uniform Dirichlet. The required fix is a **late-stage KL warm-restart schedule** (re-introduce KL after annealing to zero, with a smaller plateau target). This is non-trivial — the original Sensoy 2018 schedule does not include warm-restart, and the literature on EDL warm-restart is thin.
  - **Architect default:** prioritise as a P1 sub-task immediately after backbone resolution. The reason: if the full-backbone Cylinder3D run also collapses, we burn P1 wall-clock chasing the wrong root cause; getting the KL warm-restart in *before* the first full-backbone run avoids this.
  - **Asked of user:** confirm P1-priority, or push to P2 if you want full-backbone RQ1 numbers (mIoU + ECE without RQ2-AUROC) out of P1 even if RQ2-AUROC is suboptimal at first.

If neither question is answered by 2026-06-01, the architect proceeds with the defaults: Path 2 → 1 → 3, KL warm-restart engineered in P1 alongside backbone resolution.

---

## §9 Go / No-Go Criteria (v4, 6 conditions — preserved from v3; status updated)

All six must be green to proceed past P4 (Ablations + Deploy phase end); any red triggers documented fallbacks. Each is checkpointed at the phase shown in §7.

1. **G-1 — Baselines reproducible (P1 end).** R2-reimpl produces an end-to-end mIoU on SemanticKITTI seq 08 within ±2 mIoU of published Cylinder3D numbers, on the §3.9-resolved backbone, by end of P1. **v4 provisional status: GREEN at preliminary scale (W2-4 done, M1 vs R2 ordering as predicted at 0.21 M-param PointNet);** formal pass deferred to full-Cylinder3D rerun in P1. If formal fail, slip P1 by 1 week and re-baseline; if still failing P1+1 week, switch backbone per §3.9 Path 3 mandate.

2. **G-2 — Evidential head working (P1 end).** M1 Dirichlet evidence module shows ≥ +1 mIoU and reduced ECE on at least SemanticKITTI seq 08 by end of P1. **v4 provisional status: GREEN at preliminary scale (M1 mIoU 14.73 % vs R2 1.69 %, ECE 0.171 vs 0.490 at matched 0.21 M-param PointNet, W3-A done);** formal pass deferred to full-Cylinder3D rerun. If formal fail at full backbone, retreat to §0 fallback (calibration-only RA-L paper).

3. **G-3 — KITTI-360 revisit usable (P2 start).** Pre-computed odometry-overlap matrix delivers ≥ 5 revisit pairs with ≥ 30 m sustained overlap by end of P1 (gated on KITTI-360 transfer in P1). **v4 status: PENDING** (KITTI-360 not yet on server). If < 5, augment per R-13 mitigation; if even synthetic injection fails by P2, demote RQ4 to single-session ECE-drift micro-study.

4. **G-4 — M2 loop closure precision (P2 end).** Confidence-aware loop closure achieves ≥ 70 % precision at 50 % recall on SemanticKITTI seq 08 self-loop pairs by end of P2. **v4 status: PENDING** (M2 not yet implemented; preliminary RQ4 single-session intra-seq A/B-half stand-in delivers P=0.77, R=0.52, which is consistent with G-4 ≥ 70 % precision but is not the G-4 evaluation — G-4 needs the actual loop-closure pair evaluation, not the stale-voxel removal evaluation). If formal fail, drop the entropy-channel descriptor variant (A-3) from main and use h_class only.

5. **G-5 — RQ2 lead story green (P1 end).** Either: (a) open-set vacuity AUROC ≥ 0.80 on the SemanticKITTI 14/5 primary split AND AUPR ≥ 0.60, with closed-set mIoU on the 14 known classes within 1.5 mIoU of the fully-supervised baseline, by end of P1 — OR — (b) Robo3D fallback path: vacuity-as-corruption-detector AUROC ≥ 0.80 averaged across corruptions by end of P1. **v4 provisional status: GREEN at preliminary scale (AUROC 0.8082 on 14/5 split, W3-C done);** AUPR + closed-set mIoU at full backbone deferred to P1 formal pass. If formal fail, RQ2 demoted to honest negative result section; C1 "one vacuity, three jobs" tightens to "two jobs".

6. **G-6 — Khronos baseline ready (P3 end).** Khronos reproduction runs end-to-end on SemanticKITTI seq 08, producing a published-paper-comparable closed-set mIoU number by end of P3. **v4 status: PENDING.** If full reproduction fails: ship Table V as a "published numbers vs ours" hybrid per R-17 mitigation; RQ5 H5 claim weakens to "competitive with published Khronos numbers" but RQ5 is not aborted. Hard abort of RQ5 only if Khronos published numbers are also not available for the dynamic-split frames (unlikely but documented).

If any of G-2 or G-4 reds at formal pass, the §0 fallback to a calibration-only RA-L paper engages. G-1 / G-3 / G-5 / G-6 reds shrink scope but do not abort.

---

## Closing note on v4 ordering

The single most important call-out v4 makes that v3 did not: **the backbone-resolution path (§3.9) is now the gating step.** Every downstream G-criterion (G-1 closed-set baseline, G-2 evidential head working at full scale, G-5 open-set AUROC at full scale) depends on resolving the spconv 1.x ↔ 2.x value-semantic gap or accepting a Path 3 alternative-backbone with a mIoU haircut. The W3 preliminary results buy us the *direction* of all five method-level claims (RQ1 ordering, RQ2 AUROC, M3 decay ratio, RQ4 directional sanity) but the *position* of those numbers in the IROS table cells requires the resolved backbone. Path 2 (PVKD) is the architect's first recommendation because it is the lowest-risk continuation of the planned Cylinder3D-family lineage.

The secondary v4 call-out: **R-14 EDL instability is no longer a theoretical risk; it is observed.** The full-backbone training pipeline must include a KL warm-restart schedule before we run RQ2 at scale; otherwise we will spend P1 chasing a tuning artefact rather than a method effect.

If both calls (backbone + KL warm-restart) are handled in P1 as planned, the rest of the v4 plan is mechanical execution on the 5090 server with a buffer of approximately 6-8 weeks against the 2027-03 IROS deadline. The horizon is comfortable; the risk concentration is in the first 2 weeks.

---

*End of Research Plan v4. Companion: `paper_outline_v2.md` (filename retained for v3/v4 contents; v2 contents archived at `paper_outline_v2_archived.md`). v3 plan frozen at `research_plan_v3_archive.md`. v2 plan frozen at `research_plan_v2.md`. v1 plan frozen at `research_plan_v1.md`.*

*v4 critical-path summary: P1 backbone unblock (deadline 2026-06-01 for path selection, 2026-06-12 for full-scale RQ1/RQ2) is the single highest-priority deliverable. All other v4 work cascades from this.*
