# Research Plan

**Working title (primary):** *EviRad-Map: Evidential LiDAR + 4D-Radar Online Metric-Semantic Mapping for Weather-Robust Robot Navigation*

**Working title (fallback):** *EvidVox: Evidential Confidence-Weighted Semantic Voxel Fusion for Online Metric-Semantic Mapping*

**Author / lead:** (TBD by user)
**Target venue:** IEEE RA-L (primary), with IROS / ICRA conference option (8-page double-column).
**Prepared:** 2026-05-28
**Stage:** Stage-1 architect output of the `academic-pipeline` skill; precedes any `/ars-full` execution.
**Upstream inputs:**
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\r2_audit.md` (R2 6-dim technical audit)
- `D:\_7_sci\semantic_mapping\_new_paper\artifacts\lit_scan.md` (54-entry literature scan)
- `D:\_7_sci\semantic_mapping\_Online_Metric_Semantic_Mapping_for_Autonomous.txt` (R2 cleaned full text)

---

## §0 Direction Decision Matrix (mandatory before §1)

The R2 audit shortlisted three improvement axes (Top-1 evidential voxel fusion; Top-2 lifelong + loop; Top-3 adaptive multi-res hash SDF). The literature scan flagged a fourth, genuinely-open niche: *no integrated **LiDAR + 4D-Radar semantic mapping** system was located in 54 entries — detection in adverse weather is well covered (L4DR / RaSS / K-Radar), but the volumetric/SDF mapping side remains empty.* We treat that as candidate Z, and a hybrid Z + (X) as candidate W.

| Axis | X. Evidential Voxel Fusion (R2 Top-1) | Y. Lifelong + Loop + Decay (R2 Top-2) | Z. LiDAR + 4D-Radar Semantic Mapping (lit gap) | **W. Hybrid: 4D-Radar + LiDAR Evidential MSM** |
|---|---|---|---|---|
| Novelty (1-5) | 3 — evidential mapping is hot but ConvBKI [C9] / LatentBKI [D10] already crowd the space | 2 — Khronos [A2/D1] just took the 4D-MSM crown; loop-closure papers crowded (B4) | **5** — no LiDAR+4D-radar *mapping* paper found in 2022-2026 scan (lit_scan §Search Failures bullet 3) | **5** — same niche as Z plus a principled evidential contribution to handle radar-LiDAR disagreement |
| Venue-fit (RA-L 8 p) | 4 — fits classic RA-L "method + benchmark" shape | 4 — fits IROS systems track better than RA-L | 4 — RA-L recently opened a weather-robust perception line (SemanticSpray [B3.9]) | **5** — RA-L weather-robust + novel mapping combination is unique |
| R2 incremental delta | 4 — directly fills R2 §II-C-3 (S1/S2/S5) | 3 — fills T1/T2/T3 but rewrites half of R2 | 3 — adds a modality R2 never had | **5** — fills S1/S2/S5 *and* adds a new modality |
| SOTA differentiation | 3 — must beat ConvBKI / OpenVox in mIoU; same problem setting | 2 — Khronos owns dynamic-scene factorisation; uphill compare | **5** — Khronos / GS-LIVO / Clio all use LiDAR+RGB only; no radar | **5** — same as Z |
| Data availability | **5** — SemanticKITTI / nuScenes-LiDARSeg fully labelled | 4 — needs long re-visited sequences (Boreas / KITTI-360) | 2 — K-RADAR [B3.1] has object boxes only, *no dense semantic GT*; major risk | 3 — K-RADAR for radar, nuScenes for dense LiDAR semantics; pseudo-label bridge needed |
| Engineering effort (1=tiny, 5=huge) | 2 — modify Bayes update + add Dirichlet head | 4 — submap fusion + loop + decay are all sizeable | 5 — full multi-modal preproc + sync + calib | **4** — radar-LiDAR sync solved by K-RADAR calibration; evidential head reusable from X |
| Reviewer risk | Low — well-understood territory | Med — direct head-to-head with Khronos is dangerous | High — evaluation legitimacy without semantic radar GT will be challenged | **Med** — same evaluation risk as Z but the evidential angle gives a per-voxel uncertainty story reviewers can attack/defend on its own |
| Short rationale | "Safe, incremental, but easy to scoop by OpenVox v2 / LatentBKI v2 in 2026." | "Strong system paper but Khronos already owns the comparison axis." | "Highest novelty, but evaluation without dense radar GT is fragile." | "Best ratio of novelty (lit gap) to feasibility (evidential head + pseudo-GT bridge make the evaluation defensible)." |
| **Weighted total** | 21 | 18 | 24 | **27** |

**Decision (primary):** **W — EviRad-Map.** A LiDAR + 4D-Radar online metric-semantic mapping system whose semantic fusion layer is Dirichlet-evidential (so radar's coarse, weather-robust evidence and LiDAR's fine, weather-fragile evidence are combined under a uniform uncertainty calculus, with closed-form open-set "I-don't-know" handling).

**Fallback (if at Week 3 the radar-GT bridge fails to validate):** **X — EvidVox.** Drop the radar branch, ship a pure LiDAR/RGB evidential voxel mapping system on SemanticKITTI + nuScenes-LiDARSeg with the same evidential head, the same loop+decay submodule from Y, and a multi-condition robustness study on ACDC [B3.3] / SemanticSpray [B3.9] using LiDAR-only pseudo-degradation.

---

## §1 Problem Statement & Motivation

### 1.1 Scenario
A ground or aerial robot navigates an unstructured outdoor environment (campus, urban street, semi-off-road) under operationally realistic conditions including light rain, fog, low light, and dynamic actors. The robot must build, online and on-board (Jetson-class GPU), a dense 3D metric map enriched with semantic labels suitable for downstream traversability analysis, path planning, and (optionally) language-grounded querying.

### 1.2 R2 baseline summary and limitations
R2 [A1] (Jiao et al., HKUST) couples LVIO [R3LIVE] state estimation with an NvBlox TSDF backbone and a confidence-aware HRNet [P1] segmenter, fused per-voxel through an unspecified iterative Bayes filter; output is a semantic mesh, downstream "road = drivable, sidewalk = not" is hard-coded for traversability. The audit identified four crippling weaknesses for any reviewer competent in the topic:
1. **Methodological under-specification.** The Bayes update formula, prior, and confidence-into-fusion pathway are all missing (audit S1/S2/S4). Reviewers cannot reproduce.
2. **No quantitative evidence.** Zero ablation, zero baseline number, zero mIoU, zero F-score, evaluation only on two self-recorded sequences (audit E1/E2/E3/E5).
3. **No robustness story.** All figures are daytime clear-sky; segmentation under rain/fog/night is untested (audit P4).
4. **No long-term/loop-closure mechanism.** Lifelong drift, stale voxels, and re-visits are unaddressed (audit T1/T2/T3).

R2 is structurally a 5-page workshop snapshot whose journal extension [A1 TASE-24] expands prose but does not close the four gaps above.

### 1.3 Differentiation vs current SOTA
The literature scan flags three threats that any 2026 R2-successor must address:
- **Khronos [A2/D1]** — RSS-24, 4D dynamic-scene MSM. *EviRad-Map's complementarity:* Khronos factorises *time*; we factorise *modality + uncertainty*. Orthogonal axis.
- **GS-LIVO [B2.7/D6]** — T-RO-25, HKUST sibling lab, photo-realistic LIC-Gaussian-Splat mapping on Jetson Orin NX. *Complementarity:* GS-LIVO is photometric and rebuilds Gaussians per region; our representation stays voxelised, focuses on semantic uncertainty and adds radar — not a like-for-like collision.
- **Clio [A3/D2]** — RA-L-24, task-driven open-set scene graph. *Complementarity:* Clio compresses *given a task*; we provide a *task-agnostic, uncertainty-aware substrate* that any task layer (including Clio's) could be built atop.

The 4D-radar gap in lit_scan §Search Failures is the strategic moat: every direct competitor above ingests only LiDAR + RGB(+IMU). None ingest 4D-radar. None expose a Dirichlet-based open-set uncertainty channel at the voxel level.

### 1.4 Thesis statement
> *In adverse, perceptually-degraded outdoor scenes a metric-semantic voxel map becomes both more accurate and more honest when it (i) fuses LiDAR and 4D-radar evidence through a Dirichlet-evidential per-voxel posterior with explicit "unknown" mass, and (ii) routes downstream traversability and re-localisation decisions through that posterior's uncertainty rather than its point estimate.*

---

## §2 Research Questions & Hypotheses

### RQ1 — Does evidential per-voxel fusion (Dirichlet posterior with explicit unknown mass) outperform the iterative point-estimate Bayes filter used by R2 and ConvBKI on dense semantic mapping benchmarks?
- **H1.** On SemanticKITTI sequences 08/11-21, an evidential voxel update yields ≥ +3 mIoU and ≥ −20% expected calibration error (ECE) versus an R2-reimplemented argmax-Bayes baseline at equal voxel size and equal compute.
- **Boundary.** Holds for closed-set 19-class setting; for open-set we expect ECE win to grow, mIoU possibly to shrink because "unknown" steals mass from rare classes.

### RQ2 — Does adding 4D-radar as a second evidential modality close the weather-induced performance gap of a LiDAR-only mapper?
- **H2.** On K-Radar fog/rain/snow splits, LiDAR + 4D-radar evidential fusion preserves ≥ 80% of clear-weather mIoU; LiDAR-only baseline preserves ≤ 60%. (Targets refined after Week-3 pilot.)
- **Boundary.** Requires radar's stochastic noise to be modelled as a *high-entropy Dirichlet evidence vector* rather than a high-confidence wrong label; if not, radar can poison the fusion.

### RQ3 — Does propagating the evidential uncertainty into downstream traversability and re-localisation improve navigation success rate in adverse conditions over R2's hard-coded "road = drivable" rule?
- **H3.** On a closed-loop SemanticSpray-style wet-road eval, an uncertainty-gated traversability head produces ≥ +15 pp navigation success rate and ≥ −50% collision rate versus the R2 rule, at equal path length.
- **Boundary.** Holds when the navigation policy is allowed to *defer / re-plan* on high-uncertainty voxels; fails if the policy is forced to commit.

### RQ4 — How does the evidential, radar-augmented map scale and degrade gracefully in long-term lifelong operation?
- **H4.** With a Dirichlet-conjugate evidence decay schedule plus a semantic-aware submap loop closure (one node per submap, voxel class-histogram descriptor), the map's per-voxel ECE stays bounded under repeated revisits and stale-object removal exceeds 80% precision over 10 km of trajectory.
- **Boundary.** Tested on Boreas [B3.4] / KITTI-360 stretches with ground-truth re-visits; not claimed for unbounded multi-session merging across robots.

---

## §3 Technical Approach

### 3.1 System architecture (ASCII)

```
                    +-------------------+        +----------------------+
                    | LiDAR (OS1-128/    |       | 4D-Radar (K-Radar /  |
                    | Velodyne / OS0-128)|       | Oculii Eagle / TI    |
                    +---------+----------+       | Cascade)             |
                              |                  +----------+-----------+
                              v                             v
                  +-----------+-------------+   +-----------+-------------+
                  | LiDAR Semantic Head      |  | Radar Semantic Head      |
                  | (Cylinder3D / SPVCNN /   |  | (RaSS-distilled / RadarMOS|
                  |  RangeNet++ ; closed-set |  |  ; coarse stuff classes)  |
                  |  Dirichlet evidence e_L) |  |  Dirichlet evidence e_R)  |
                  +-----------+--------------+  +-------------+------------+
                              |                                |
                              +---------+----------+-----------+
                                        |          |
                                        v          v
                          +-------------+----------+-------------+
                          | §3.2 Evidential Voxel Posterior      |
                          | per-voxel Dirichlet posterior alpha  |
                          | with explicit "unknown" mass m_u     |
                          +---------------+----------------------+
                                          |
       +-----------+-------+              v             +-------------------+
       | RGB camera        |   +----------+----------+  | LVIO state est.   |
       | (segmentation     +-->| §3.3 Modality-Aware  |<+ (R3LIVE-class,    |
       |  evidence e_C)    |   |  Trust Allocation   |  | reused unchanged) |
       +-------------------+   |  (weather-aware w_*) |  +-------------------+
                               +----------+-----------+
                                          |
                                          v
                          +---------------+-------------------+
                          | §3.4 Semantic Submap Backbone     |
                          | adaptive multi-res hash voxel SDF |
                          | + Dirichlet evidence + decay      |
                          +---------------+-------------------+
                                          |
              +---------------------------+----------------------------+
              v                           v                            v
   +----------+-----------+   +-----------+----------+   +-------------+--------+
   | §3.5 Uncertainty-    |   | §3.6 Loop-closure &  |   | downstream queries: |
   | aware traversability |   | submap decay         |   | path plan, language |
   +----------------------+   +----------------------+   +----------------------+
```

### 3.2 Module 1 — Evidential per-voxel posterior
Replace R2's unspecified Bayes filter (S1/S2) by a Dirichlet-evidential posterior. Per voxel `v`, maintain accumulated evidence `α_v ∈ R^{C+1}` (last channel = open-set "unknown") and uncertainty mass `m_u = (C+1) / Σ α_v` (vacuity). Each new observation contributes evidence `e_v(z) = w(modality, weather) · softplus(logit(z))`, summed into `α_v`. Posterior mean and variance for class `c` are closed-form, and `m_u` is the open-set / OOD signal that R2 and ConvBKI lack. Inspired by S-BKI / ConvBKI [C8/C9] and LatentBKI [D10] but using EDL-style Dirichlet rather than Bayesian kernel inference, which buys us per-voxel epistemic uncertainty without spatial-kernel smoothing.

### 3.3 Module 2 — Modality-aware trust allocation
The weight `w(modality, weather)` modulates per-modality evidence strength. Weather context comes from (i) point-cloud-derived rain/snow indicator (TripleMixer-style [B3.7] cheap statistics), (ii) RGB image global brightness/contrast, (iii) radar SNR. Trust schedule is learned on K-Radar normal vs adverse splits and is the single source of "weather robustness" in the system. Critically, weights modulate *evidence*, not *posterior label* — so under heavy fog the LiDAR branch contributes near-zero evidence and the radar branch's coarse but available evidence dominates, while the voxel's vacuity rises smoothly rather than the posterior flipping.

### 3.4 Module 3 — Adaptive multi-res hash voxel + decay
A two-level hash voxel structure: 0.05 m near robot pose (±15 m) and 0.25 m far-field, swappable on submap rollover. Each voxel stores SDF, weight, and the Dirichlet evidence vector `α_v`. Evidence decays exponentially with stale time `τ` (audit T2 fix) under a Dirichlet-conjugate rule that preserves the posterior mean but inflates vacuity, so stale voxels become re-writable as new evidence arrives. Submaps are sealed every 50 m or 30 s of trajectory.

### 3.5 Module 4 — Uncertainty-aware traversability and loop closure
Traversability head consumes both the posterior class mean and `m_u`; voxels above an uncertainty threshold are marked "defer / re-plan" rather than "untraversable" (audit S5 fix). Loop closure: per-submap descriptor = (i) voxel class histogram over confident voxels, (ii) Dirichlet entropy histogram. Matching via cosine + entropy-EMD; verified by semantic-ICP over class-consistent voxels (audit T4 fix). Pose graph optimisation re-projects α_v consistently across the loop.

### 3.6 Key equations (placeholders to be derived in `/ars-full`)
- Eq. 1 — Dirichlet evidence accumulation: `α_v^{t+1} = α_v^{t} + e_v^{t+1}`.
- Eq. 2 — Per-voxel vacuity (open-set / OOD score): `m_u(v) = (C+1) / sum(α_v)`.
- Eq. 3 — Modality-weighted evidence: `e_v(z, mod, weather) = w(mod, weather) · softplus(z)`.
- Eq. 4 — Conjugate decay: `α_v^{t+Δ} = ((α_v^{t} − 1) · exp(−Δ/τ)) + 1`.
- Eq. 5 — Loop-closure semantic descriptor: `d(S) = (h_class(S), h_entropy(S))`.

### 3.7 Methodological deltas, one line each
- vs **R2 [A1]:** specifies the Bayes filter (Eq. 1-3 closed form), adds radar modality, adds vacuity, adds decay, adds loop closure.
- vs **Khronos [A2]:** Khronos factorises short-/long-term over time; we factorise modality + uncertainty (orthogonal, can be combined in future).
- vs **GS-LIVO [B2.7]:** GS-LIVO is photometric-Gaussian; we stay voxel-Dirichlet and add radar. Different output products.
- vs **Clio [A3]:** Clio compresses given a task; we keep a task-agnostic uncertainty-aware substrate. Clio could sit on top.
- vs **ConvBKI [C9] / LatentBKI [D10]:** they use Bayesian kernel inference for spatial smoothing; we use EDL Dirichlet for closed-form vacuity, plus a second modality.
- vs **OpenVox [A8/D8]:** OpenVox uses Bernoulli per-instance with Bayes update; we use Dirichlet over all classes including unknown and add radar.
- vs **L4DR [B3.6]:** detection only; we lift to dense voxel mapping.

---

## §4 Experimental Plan

### 4.1 Datasets (final shortlist, 3 datasets)
| # | Dataset | Role | Rationale |
|---|---------|------|-----------|
| D-a | **SemanticKITTI** (Behley et al., 2019; used via Cylinder3D community split) | Primary dense semantic GT for LiDAR mapping mIoU; trains the LiDAR semantic head; provides the closed-set 19-class baseline arena | Standard, reproducible, every cited competitor has results on it. |
| D-b | **nuScenes-LiDARSeg** (also gives RGB + radar) | Cross-domain check (urban USA / Singapore) + provides a *3D radar* (3D not 4D) branch for an honest weaker-radar ablation | The only mainstream AD dataset combining LiDAR semantic GT, RGB and radar in one rig. |
| D-c | **K-Radar [B3.1]** (35K frames, 4D radar + LiDAR, fog/rain/snow scenes) | Primary adverse-weather + 4D-radar evaluation | The only dataset with synchronised 4D-radar + LiDAR + adverse weather; absence of dense semantic GT is acknowledged in §4.3 (pseudo-labelling protocol). |

**Secondary / robustness only (not main results):** ACDC [B3.3] (image-only adverse-condition seg, to pretrain RGB head), SemanticSpray [B3.9] (RA-L-24 wet-road set, used for closed-loop traversability in RQ3).

**Datasets *not* used and why:**
- *Replica / TUM RGB-D / Hilti SLAM Challenge* — indoor, no radar, off-topic.
- *Newer College* — outdoor but no semantic GT, would force pure mapping geometry comparison.
- *KITTI-360* — would be ideal for RQ4 lifelong but the radar branch is absent; held in reserve as Week-9 stretch goal.
- *SCAND* — social navigation focus, no semantic 3D GT.

### 4.2 Baselines (5 total)
1. **R2-reimpl.** Faithful Python/CUDA reimplementation of R2 [A1] (NvBlox + HRNet + argmax-Bayes). Most baselines avoid this; we must do it ourselves since no code is released (R2 audit, "code release: 未提及").
2. **NvBlox-vanilla + per-frame argmax** (lower bound; non-semantic geometry only with naive label majority).
3. **ConvBKI [C9].** Authors' code; the canonical probabilistic semantic voxel competitor (lit_scan reviewer-threat #3).
4. **OpenVox [A8/D8].** Probabilistic voxel + open-vocab; ablates the value of our Dirichlet vs their Bernoulli choice.
5. **Khronos [A2/D1]** *(comparison restricted to RQ1 closed-set mIoU on SemanticKITTI; we do not claim parity on its dynamic-scene 4D factorisation)*.

Stretch baseline (if time allows; not a Go criterion): **L4DR [B3.6]** lifted to per-voxel by majority-voting its detection outputs — purely as a "detection-only is not enough" sanity bar.

### 4.3 Metrics
| Tier | Metric | Notes |
|------|--------|-------|
| **Mapping** | mIoU, per-class IoU, F@5 cm reconstruction, voxel coverage, GPU memory peak, end-to-end latency (mean and 99-pct) | Standard set; closes audit E4/E5. |
| **Uncertainty** | Expected Calibration Error (ECE), Brier score, AUROC for *unknown* class detection | Distinguishes us from R2 and ConvBKI; supports H1. |
| **Robustness** | Δ-mIoU vs clear weather on K-Radar fog/rain/snow; AUC of mIoU-vs-weather-severity curve | Supports H2. |
| **Downstream nav** | Success rate, collision rate, path length ratio, avg deferral rate, on a closed-loop SemanticSpray-style sim | Supports H3. Implemented in Isaac Sim with replayed sensor data. |
| **Lifelong** | Stale-voxel removal precision/recall over 10 km revisited trajectory; ECE drift; map size | Supports H4. |

**Pseudo-GT bridge for K-Radar dense semantics.** K-Radar provides box-level labels; no voxel-level semantics. We construct dense pseudo-labels by (i) running a SemanticKITTI-pretrained Cylinder3D on clear-weather K-Radar frames only, (ii) projecting class labels into K-Radar's LiDAR coordinate frame and voxelising at 0.25 m, (iii) accepting only voxels where the pretrained model's confidence ≥ 0.9 and ≥ 3 frames agree. This is reported transparently as a **pseudo-GT** evaluation; we additionally hold out 50 K-Radar frames for *manual* sparse-voxel labelling (estimated ~2 days of annotation) as a sanity-check sample. Any number reported on K-Radar will be explicitly labelled "pseudo-GT" or "manual sparse".

### 4.4 Ablations (≥ 5, each isolates one design decision)
| # | Variable | Question answered |
|---|---|---|
| A-1 | ± Dirichlet evidential (vs argmax-Bayes / vs categorical Bayes) | RQ1: is evidential worth it on calibration? |
| A-2 | ± radar branch | RQ2: how much does radar help vs clear-weather decay? |
| A-3 | ± modality-aware trust schedule (vs equal weights) | Isolates §3.3 design. |
| A-4 | ± vacuity-aware traversability (vs hard threshold) | RQ3: does propagating uncertainty downstream matter? |
| A-5 | ± conjugate evidence decay | Isolates §3.4 lifelong contribution. |
| A-6 | ± semantic submap loop closure | RQ4 lifelong factor. |
| A-7 (stretch) | ± adaptive multi-res voxel (vs fixed 0.25 m) | Defensive ablation in case reviewers raise voxel-size confound. |

### 4.5 Compute budget
- Training and inference target: **single workstation, 1× RTX 4090 (24 GB)**; deployment target check on Jetson Orin NX (16 GB) to match GS-LIVO [B2.7] claim.
- Estimated wall-clock budget:
  - Semantic head pretrain (Cylinder3D on SemanticKITTI): 2 days × 1× 4090.
  - K-Radar pseudo-GT generation: 1 day.
  - Full mapping system batch eval over SemanticKITTI sequences 08+11-21: 1 day per baseline; 5 baselines + ours = 1 week.
  - Ablation grid (7 ablations × 3 sequences): ~3 days.
  - Closed-loop nav sim (SemanticSpray + Isaac Sim, ~50 episodes per condition): 2 days.

---

## §5 Expected Contributions

- **C1 (System).** First open-source online metric-semantic mapping system to fuse LiDAR and 4D-radar evidence in a single voxel representation, deployable on Jetson Orin NX with submap-level loop closure. *Anchored to lit_scan §Search Failures bullet 3 ("genuine open gap").*
- **C2 (Algorithm).** A Dirichlet-evidential per-voxel fusion rule with explicit vacuity and conjugate decay (Eqs. 1-4), used jointly as the open-set / OOD signal, the modality trust allocator's substrate, and the lifelong stale-voxel detector. Strictly generalises R2's unspecified Bayes filter and ConvBKI's [C9] kernel-Bayes update.
- **C3 (Empirical).** On three public datasets (SemanticKITTI, nuScenes-LiDARSeg, K-Radar) and one closed-loop sim (SemanticSpray + Isaac Sim) we report: (i) ≥ 3 mIoU and ≥ 20 % ECE improvement over a faithful R2 reimplementation on SemanticKITTI, (ii) ≥ 80 % of clear-weather mIoU retained on K-Radar adverse splits where the LiDAR-only baseline retains ≤ 60 %, (iii) ≥ 15 pp closed-loop navigation success-rate uplift on SemanticSpray. All numbers will be replaced by measured ones before submission; the bar sets the Go/No-Go threshold in §9.
- **C4 (Reproducibility).** Code, Docker, ROS 2 launch files, K-Radar pseudo-GT pipeline, and all hyperparameters released — directly attacks R2 audit E7 ("复现性零保障").

---

## §6 Risk Register

| ID | Risk | Prob | Impact | Mitigation | Trigger to fall back |
|----|------|------|--------|------------|---------------------|
| R-1 | K-Radar lacks dense semantic GT; reviewers reject pseudo-GT evaluation | **H** | **H** | Transparent pseudo-GT protocol (§4.3) + 50-frame manual sparse-GT sanity check + agreement-rate stats between pseudo and manual | If sanity check shows pseudo-GT agreement < 75 %, fall back to candidate X (LiDAR-only evidential) |
| R-2 | Khronos [A2] releases a 4D + multi-modal extension before submission | M | H | Track arXiv weekly via post-research literature monitor (deep-research skill mode 7); pre-register our differentiation as "modality + uncertainty axis" not "time axis" | If Khronos-v2 lands on radar before us, pivot main claim to uncertainty calibration story |
| R-3 | GS-LIVO [B2.7] HKUST sibling lab releases semantic-GSplat extension | M | M | Differentiate on representation (voxel vs Gaussian) and modality (radar in/out); cite explicitly | If GS-LIVO-semantic ships, sharpen our "deployable on Jetson with radar" angle |
| R-4 | Dirichlet evidential fusion fails to beat ConvBKI [C9] in mIoU | M | H | Calibration (ECE / AUROC unknown) is a separate axis; we win on uncertainty even if mIoU ties | If ECE also ties, retreat to "modality fusion under uncertainty" story (radar branch becomes lead) |
| R-5 | 4D-radar semantic head trains poorly (no large-scale dense GT exists) | **H** | M | Cross-modal distillation from LiDAR semantic head in clear-weather (RaSS [B5.2] style) | If distilled radar head AUROC < 0.7 for `vehicle`/`road`/`vegetation`, restrict radar evidence to a 4-class "macro" set |
| R-6 | Closed-loop navigation in Isaac Sim differs from real robot behaviour | M | M | Acknowledge as limitation in §V; report SemanticSpray-style passive replay as primary, sim closed-loop as secondary | n/a |
| R-7 | Voxel-size + multi-res confound contaminates mIoU comparison vs ConvBKI | M | M | Ablation A-7; report all numbers also at fixed 0.25 m | n/a |
| R-8 | LiDAR semantic head licence (Cylinder3D / SPVCNN) prevents code release | L | M | Train an in-house head on SemanticKITTI from scratch (RangeNet++ is BSD) | n/a |
| R-9 | 8-page limit overflow given 4 RQs and 3 datasets | M | M | Move ablation tables and per-class IoU to supplementary; keep main paper at 4 RQs but 3 hero claims | Drop RQ4 lifelong to appendix; keep RQ1+2+3 in main |
| R-10 | Reviewer demands real-robot deployment | M | M | Cite scope as "methodological with public-dataset validation"; commit to follow-up demo paper | If RA-L desk-rejects on this, retarget IROS systems track where sim-only is acceptable |

H/M/L summary — **H: 2** (R-1, R-5), **M: 7** (R-2/3/4/6/7/9/10), **L: 1** (R-8).

---

## §7 Timeline (10 weeks to RA-L submission)

| Week | Focus | Concrete output |
|------|-------|-----------------|
| W1 | Reproduce R2 + ConvBKI baselines on SemanticKITTI | Numbers for both on seq 08 |
| W2 | LiDAR semantic head + Dirichlet evidence head integrated | A-1 ablation skeleton runs |
| W3 | K-Radar pseudo-GT pipeline + 50-frame manual sparse GT | Sanity-check agreement reported → **first Go/No-Go checkpoint (§9)** |
| W4 | Radar semantic head via cross-modal distillation | A-2 ablation runs on K-Radar clear split |
| W5 | Modality-aware trust schedule + adverse-weather eval | RQ2 H2 first numbers |
| W6 | Adaptive multi-res hash voxel + conjugate decay | A-5/A-7 ablations |
| W7 | Semantic submap loop closure + lifelong eval | RQ4 H4 numbers (Boreas / KITTI-360 stretch) |
| W8 | Closed-loop nav sim (SemanticSpray + Isaac Sim) | RQ3 H3 numbers |
| W9 | Writing draft 1 (`/ars-full` from outline) | Full draft + figures |
| W10 | Reviewer simulation (`academic-paper-reviewer`) + revise + format-convert + submit | RA-L submission |

---

## §8 Open Questions for the User

These genuinely change the plan; please respond before kick-off.

1. **Compute access.** Do we have ≥ 1× RTX 4090 (or A100) for ~3 weeks of cumulative GPU time, *plus* a Jetson Orin NX for the §C1 deployment check? If only an Orin is available, drop Jetson check from claims.
2. **Annotation budget.** Are ~2 person-days of manual sparse voxel labelling on K-Radar feasible? Without it, R-1 mitigation collapses and we should pre-commit to the fallback (candidate X).
3. **Authorship and venue priority.** Is RA-L the firm target, or are we willing to retarget IROS-26 (deadline shifts ~3 months later) if K-Radar evaluation needs more time? This decides whether to ship Lifelong (RQ4) as main or appendix.
4. **Radar hardware.** K-Radar uses RETINA 4D radar (specific to that dataset). Do we want the system to be radar-vendor-agnostic in the paper claim, or scope it to "demonstrated on K-Radar"? Agnostic claim doubles eval cost.
5. **Real-robot deployment.** User stated "no field collection." Confirmed — but is *replay of pre-recorded user campus sequences* (from R2's FusionPortable) on our pipeline an acceptable mid-pipeline checkpoint, or strictly public datasets only?

---

## §9 Go / No-Go Criteria (before launching `/ars-full`)

All five must be green to proceed; any red triggers the fallback path in §0.

1. **G-1 — Baselines reproducible.** R2-reimpl. and ConvBKI both run end-to-end on SemanticKITTI seq 08 by end of W2, producing within ±1 mIoU of published numbers (where published) or within ±2 mIoU of paper screenshots (R2).
2. **G-2 — Evidential head working.** Dirichlet evidence module shows ≥ +1 mIoU and reduced ECE on at least one SemanticKITTI sequence by end of W2 (sanity check that Eq. 1-3 are correctly implemented).
3. **G-3 — K-Radar pseudo-GT validated.** Manual sparse GT agreement with pseudo-GT ≥ 75 % on the 50-frame sanity sample by end of W3. If < 75 %, **fall back to candidate X** (LiDAR-only evidential mapping; rewrite §1.3, §3, §4.1 to drop radar).
4. **G-4 — Radar branch trains.** Distilled radar semantic head reaches AUROC ≥ 0.7 on K-Radar clear-weather hold-out for the 4 macro classes (vehicle, road, vegetation, building) by end of W4. If not, **restrict radar to occupancy-only evidence** (no class labels; radar contributes only to "occupied vs free" Dirichlet evidence). This is a degraded but still-publishable mode.
5. **G-5 — No scoop.** Weekly arXiv check shows no 2026 paper combining LiDAR + 4D-radar + voxel semantic mapping; if a scoop appears, sharpen contribution to the evidential calibration angle and demote multi-modal claim to "first principled evidential fusion of LiDAR-radar".

---

*End of Research Plan. Companion: `paper_outline_v1.md` in the same directory.*
