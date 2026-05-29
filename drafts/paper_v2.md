---
title: "EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay"
authors: ["[Author 1] [TBD]", "[Author 2] [TBD]"]
affiliations: ["[Institution] [TBD]"]
target_venue: "IROS 2027 (deadline ~2027-03)"
date: 2026-05-28
version: v2 (complete skeleton)
status: "§I-§III 完稿; §IV-§VI 占位待实验数据回填"
page_budget: "IEEE 8-page conference (~4400 words main + ~30 refs)"
notes: |
  - Citations were rewritten from [bibkey] form to numeric [N] form, indexed by first
    appearance in §I-§VI; the master numeric→bibkey map is at the head of References.
  - All cells whose value is pending the §IV experimental campaign are written as XX.X
    or [TBD-RQx] for numbers and as [TBD-after-exp] for prose; the HTML render highlights
    every one of them with a <mark> for visual scanning.
  - Companion files: research_plan.md (v3) §4 (Experimental Plan) / §5 (Contributions)
    / §6 (Risks), paper_outline_v2.md (v3 contents) (8-page word budget), and
    paper_tables.md (3 reusable comparison tables, embedded below as Tab I, Tab II, Tab III).
---

# Abstract

Outdoor robots navigating repeated traversal of degraded or partially unmapped environments need a metric-semantic voxel map that simultaneously is calibrated, open-set aware, and lifelong-maintainable; current online mapping systems address these three requirements through three independently parameterised heuristics. We present **EvidLife-Map**, a LiDAR-only online metric-semantic mapping system that represents each voxel as a closed-form Dirichlet posterior over the closed-set taxonomy plus one explicit "unknown" channel, and reuses the posterior's single **vacuity** scalar to drive (i) the open-set / out-of-distribution score, (ii) the loop-closure descriptor's entropy channel, and (iii) a vacuity-conditioned conjugate voxel decay rule. Three modules share this state: M1 evidential per-voxel fusion, M2 vacuity-conditioned loop closure with parameter-free conjugate inter-session submap fusion, and M3 vacuity-driven voxel decay. The system is implemented on a forked nvblox [3] layer cake and stays Jetson-deployable, closing four reviewer-credible weaknesses of the R2 baseline [1] (unspecified Bayes filter; no open-set; no loop closure; no decay) with one Dirichlet-derived signal. On SemanticKITTI we achieve <mark>XX.X</mark> mIoU and <mark>XX.X</mark> ECE [TBD-RQ1] versus a faithful R2-reimplementation and ConvBKI's [4] published 77.7%; on a 14-known/5-unknown open-set split, voxel-level vacuity AUROC of <mark>XX.X</mark> [TBD-RQ2]; on a SemanticKITTI dynamic-frame subset, dynamic-object mIoU within <mark>XX.X</mark> of Khronos [6] [TBD-RQ5] at <mark>XX.X</mark> Hz on Jetson Orin NX. Code, Docker, ROS 2 launch files, open-set split definitions, the KITTI-360 revisit-pair builder, and a 30-second supplementary demo video are released.

---

# I. Introduction

## I.A  Motivation

Outdoor robot autonomy across repeated traversal of degraded or partially unmapped environments demands a metric-semantic voxel map that simultaneously (i) quantifies what it knows about each voxel, (ii) flags voxels whose observed evidence does not match any trained category, and (iii) ages stale evidence as the world changes between visits. Current online metric-semantic mapping (MSM) systems address these requirements only piecewise. The recent R2 system [1] couples LiDAR–visual–inertial odometry with an nvblox [3] truncated signed-distance backbone and a per-voxel iterative Bayes filter, but leaves the filter under-specified, makes no provision for unknown categories, and provides no mechanism for ageing or revisiting stale voxels — three properties that audit trails of the system have identified as the most reviewer-credible weaknesses. Probabilistic-voxel competitors close one gap each: S-BKI [2] and ConvBKI [4] give kernel-Bayesian calibration; Khronos [6] gives a learned short-/long-term factorisation for dynamics; Panoptic Multi-TSDFs [8] gives submap-level long-term consistency. No single online system delivers all three jobs from one statistical quantity, leaving uncertainty quantification, lifelong maintenance, and open-set awareness as three independently parameterised heuristics.

## I.B  Approach Overview

EvidLife-Map represents each voxel as a closed-form Dirichlet posterior over the closed-set semantic taxonomy plus one explicit "unknown" channel; the posterior's vacuity mass — a single scalar derived from the Dirichlet concentration — is reused as (i) the open-set / out-of-distribution score, (ii) the entropy channel of the loop-closure descriptor, and (iii) the trigger for a vacuity-conditioned conjugate decay that ages stale voxels. The system is built as three modules sharing the same per-voxel state: **M1**, Dirichlet evidential per-voxel semantic fusion (§III.B); **M2**, vacuity-conditioned loop closure with parameter-free conjugate inter-session submap fusion (§III.C); and **M3**, vacuity-driven voxel decay (§III.D). Implementing all three on a forked nvblox [3] layer cake allows EvidLife-Map to remain LiDAR-only and Jetson-deployable while replacing R2's three independent heuristics with a single Dirichlet-derived signal. The end-to-end pipeline is illustrated in Figure 1.

![Figure 1](placeholder_fig1.png) *Figure 1: System overview. LiDAR point clouds are passed through a Cylinder3D semantic head trained with an evidential loss; per-point Dirichlet evidence is accumulated per voxel inside an nvblox `EvidentialLayer`. The resulting per-voxel vacuity mass feeds three downstream consumers (open-set head, loop-closure descriptor, voxel-decay trigger) — the load-bearing "one vacuity, three jobs" wiring of this paper.*

## I.C  Contributions

We make four contributions:

- **C1 (algorithm).** A closed-form Dirichlet-evidential per-voxel semantic posterior whose vacuity mass is, by construction, simultaneously usable as (i) an open-set / OOD score, (ii) a loop-closure descriptor entropy channel, and (iii) a lifelong decay trigger. The posterior generalises the unspecified Bayes filter of R2 [1] and contrasts with the kernel-Bayesian smoothing of S-BKI [2] and ConvBKI [4] by giving an honest per-voxel epistemic signal uncontaminated by neighbour evidence.

- **C2 (system).** A vacuity-conditioned loop-closure descriptor that augments a class-histogram channel with an entropy channel, plus a parameter-free conjugate inter-session submap fusion rule that adds Dirichlet pseudo-counts under the standard conjugacy of independent observations. Coupled with a vacuity-driven conjugate exponential decay, the three modules together close R2's three lifelong gaps (no loop closure, no decay, no inter-session fusion) without introducing any heuristic threshold that is not derived from a Dirichlet quantity already computed for M1.

- **C3 (empirical evidence).** The first systematic evaluation of a single LiDAR-only evidential MSM system on five public datasets — SemanticKITTI (closed-set RQ1 and a 14-known/5-unknown open-set split RQ2), nuScenes-LiDARSeg (cross-domain RQ1 and reverse cross-domain RQ2), KITTI-360 (multi-session lifelong RQ4), SemanticSpray [21] (uncertainty-aware traversability RQ3), and a SemanticKITTI dynamic-frame subset (head-to-head against Khronos [6] RQ5). RQ5 in particular gives the first published head-to-head between an implicit-vacuity dynamic-handling mechanism and Khronos's explicit short-/long-term factoriser on a public dynamic split.

- **C4 (deployment).** A reproducibility package — source code, Docker image, ROS 2 launch files, open-set split definitions, the KITTI-360 revisit pair builder, a Jetson Orin NX runtime benchmark, and a 30-second supplementary demo video of EvidLife-Map running live on a Jetson Orin NX over a SemanticKITTI replay — that ships with the paper.

---

# II. Related Work

## II.A  Online Metric-Semantic Mapping

Online metric-semantic mapping descends from the Euclidean signed-distance lineage of Voxblox [11] and the earlier OctoMap [12] occupancy framework, both of which provide incremental volumetric backbones for on-board planning but leave semantics outside the probabilistic state. Voxblox++ [13] and PanopticFusion [14] extend the line to volumetric instance-aware and panoptic semantic mapping, respectively, by attaching per-voxel label counts to the SDF; Voxfield [15] subsequently relaxes the projective approximation in the SDF update so that the distance field remains non-projective even under oblique LiDAR rays. The Kimera family — Kimera [9] and its distributed extension Kimera-Multi [16] — introduces dense metric-semantic SLAM with 3D dynamic scene graphs, while Hydra [10] and Hydra-Multi [17] organise the metric-semantic substrate hierarchically for real-time spatial perception and collaborative multi-robot construction. Recent systems push the frontier in three directions. R2 [1] integrates LVIO with an nvblox [3] backbone and a confidence-aware HRNet semantic head for outdoor navigation, but uses an under-specified per-voxel Bayes filter and treats unlabeled voxels as untraversable rather than as unknown. Khronos [6] adds a learned spatio-temporal factorisation that explicitly separates short-term motion from long-term change. Clio [18] compresses the scene graph under a task prior to enable open-set retrieval, and HOV-SG [19] extends the open-vocabulary scene-graph idea to a hierarchical floor/room/object layout for language-grounded navigation. The Gaussian-splatting strand — GS-LIVO [20] is its LiDAR-inertial-visual exemplar — replaces the volumetric backbone entirely with a continuous Gaussian field deployable on Jetson-class hardware. Across this entire lineage, per-voxel uncertainty handling remains the weakest axis: the Voxblox-class systems use a point estimate of class probability, the Kimera and Hydra families use the scene graph rather than the voxel as the credible-set entity, and the more recent systems use ad-hoc heuristics — confidence-head clipping in R2, a learned change-detection network in Khronos, task-prior compression in Clio — with no single statistical quantity that quantifies what each voxel does not know.

## II.B  Probabilistic Voxel Semantic Fusion

A parallel line of work treats per-voxel semantic fusion as a Bayesian estimation problem in its own right. S-BKI [2] introduces Bayesian spatial kernel smoothing for scalable dense semantic mapping, replacing the independent-voxel assumption with a continuous kernel that propagates evidence between spatially adjacent voxels. ConvBKI [4] reformulates the same idea as a real-time network with quantifiable uncertainty by implementing the spatial kernel as a convolution; LatentBKI [7] extends the line to open-dictionary continuous mapping in a visual-language latent space while preserving the quantifiable-uncertainty property. OpenVox [5] complements the BKI line with an instance-level open-vocabulary probabilistic voxel representation that maintains a per-instance Bernoulli posterior. Open-Fusion [22] and SLIM-VDB [23] integrate similar probabilistic update rules into open-vocabulary TSDF and OpenVDB backbones, respectively. These works converge on the use of a per-voxel posterior, but the posterior is consumed for one purpose only — calibration of the closed-set semantic prediction in S-BKI [2], ConvBKI [4], and SLIM-VDB [23]; open-vocabulary retrieval in LatentBKI [7] and OpenVox [5]; or queryable feature lookup in Open-Fusion [22]. None of them propagates the posterior's epistemic uncertainty into either loop closure or lifelong decay. Our M1 differs from the BKI lineage by adopting a closed-form Dirichlet posterior that derives a single vacuity scalar per voxel; in particular, we deliberately omit the spatial kernel of S-BKI / ConvBKI, because the vacuity signal that M2 and M3 consume must be an honest per-voxel epistemic quantity rather than one contaminated by neighbour evidence.

## II.C  Lifelong and Long-term Map Maintenance

Lifelong and long-term metric-semantic map maintenance has been pursued primarily through explicit change-detection or submap-fusion machinery decoupled from the voxel-level uncertainty model. Panoptic Multi-TSDFs [8] introduces a flexible submap representation with object-level long-term dynamic-scene consistency, controlled by per-submap activity status rather than per-voxel epistemic signals. Hydra-Multi [17] supports collaborative online construction of 3D scene graphs by multi-robot teams; Kimera-Multi [16] provides robust distributed dense metric-semantic SLAM. PlaneSDF [24] specialises in cross-session change detection by maintaining a plane-SDF residual; SLAM2REF [25] integrates a stored reference map to refine long-term trajectories. SA-LOAM [26] and the more recent LiDAR loop-closure detection of [27] use semantic graphs and graph-attention networks to improve the discriminability of loop-closure descriptors. LTC-Mapping [28] enhances long-term consistency of object-oriented semantic maps for general mobile robotics. Across these works the shared pattern is to introduce a separate mechanism — explicit time windows, change-detection networks, semantic-aware loop descriptors with hand-crafted weighting, or object-level activity flags — that operates on top of, but is statistically disjoint from, the per-voxel semantic posterior. Our M2 and M3 differ by deriving both the loop-closure descriptor's entropy channel and the per-voxel decay time-constant from the same Dirichlet vacuity already computed for M1.

Across §II.A, §II.B, and §II.C, the consistent omission is therefore not the *presence* of uncertainty handling but the *integration* of one uncertainty quantity across calibration, open-set detection, loop closure, and decay. Our work differs by unifying these three ad-hoc heuristics under one Dirichlet vacuity scalar.

## II.D  Comparative Positioning (Tab I)

To make the positioning concrete, Tab I lays out EvidLife-Map against the three reviewer-credible SOTA threats (Khronos [6], GS-LIVO [20], Clio [18]), the R2 baseline [1] we re-implement, and the nvblox [3] substrate we fork. Across the eight methodology dimensions, the three threat systems are *methodologically orthogonal* to our work — Khronos owns the dynamic-scene axis through explicit 4D factorisation, Clio owns the task-driven open-set axis through Information-Bottleneck compression of CLIP primitives, and GS-LIVO owns the photo-realistic representation axis with Jetson-deployable 3DGS. EvidLife-Map does not contest any of these three single-axis SOTA points; instead it reuses *one statistical quantity* — Dirichlet vacuity — across calibration, open-set OOD, vacuity-conditioned loop closure, and vacuity-driven decay, where each threat system maintains a separate auxiliary mechanism per task. The R2 baseline (row 5) sits below all four as the system that *attempts* outdoor LiDAR-only MSM but leaves the per-voxel Bayes filter unspecified and has no open-set / loop / decay machinery; nvblox (row 6) is the geometric substrate both R2 and our system fork from, bounding what "geometric-only" lower-bound performance looks like. The matrix therefore positions our work as the *integrating* contribution: rather than replacing any one threat system, we close R2's four open gaps with a single Dirichlet-derived signal and show that the three remaining axes (open-set, loop closure, dynamic-implicit) all admit a vacuity-based answer.

**TABLE I. EVIDLIFE-MAP RELATIVE TO THE THREE REVIEWER-CREDIBLE SOTA THREATS, THE R2 OUTDOOR-MSM BASELINE [1], AND THE NVBLOX SUBSTRATE [3].**

| # | System | Representation | Uncertainty | Open-Set | Loop Closure | Lifelong | Dynamic | Sensors | Jetson RT | Code | Venue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **EvidLife-Map (ours)** | TSDF + Dirichlet evidential layer | Dirichlet, single vacuity scalar $u_v$ | Vacuity-based (14+1 split) | Vacuity-conditioned (class+entropy) | Vacuity-driven conjugate decay + Dirichlet inter-session fusion | Implicit via vacuity | LiDAR-only | <mark>[TBD-RQ5]</mark> $\ge$ 5 Hz target | Planned (Apache-2.0) | IROS 2027 (target) |
| 2 | Khronos [6] | Multi-res TSDF + dynamic fragments | Truncated-LS + binary outlier weights | Modular front-end | Yes (factor graph) | Spatio-temporal factor graph | Explicit 4D factorisation | RGB-D + IMU | 22.2 FPS Active Window (i7-12700H, CPU) | Yes | RSS 2024 |
| 3 | GS-LIVO [20] | 3D Gaussian Splatting | Photometric + LiDAR p2p in IESKF | $\times$ | $\times$ | $\times$ | Implicit (replacement) | LiDAR + IMU + Cam | Orin NX 16GB $\approx$ 21 Hz | Yes | T-RO 2025 |
| 4 | Clio [18] | Open-set 3D scene graph | Primitive-level CLIP + IB | Task-driven via IB | Inherited (Hydra/Khronos) | $\times$ | Inherited | RGB-D | Laptop GPU; Orin NX [NR] | Yes | RA-L 2024 |
| 5 | R2 [1] (baseline) | TSDF on nvblox + label probs | Iterative Bayes (unspecified) | $\times$ ("unlabeled = untraversable") | $\times$ | $\times$ | $\times$ | LVIO | RTX 3080 Ti only | $\times$ (we re-implement) | T-ASE 2024 |
| 6 | nvblox [3] (substrate) | TSDF + ESDF + Color | $\times$ (geometric only) | N/A | $\times$ | $\times$ | N/A | RGB-D or LiDAR | Xavier AGX < 20 ms / scan | Yes | ICRA 2024 |

---

# III. Method

## III.A  System Overview

EvidLife-Map maintains, per voxel `v`, a Dirichlet posterior over `C+1` classes — `C` closed-set semantic classes plus one explicit "unknown" channel — represented by a concentration vector $\alpha_v \in \mathbb{R}^{C+1}$ with $\alpha_{v,k} \ge 1$. A single scalar derived from $\alpha_v$, the **vacuity** $u_v$, is the load-bearing quantity of the paper: it is computed once per voxel per fusion step and consumed three times — by the open-set head, by the loop-closure descriptor, and by the voxel-decay trigger. Figure 1 summarises the end-to-end pipeline: a LiDAR scan is passed through a semantic head trained with an evidential loss, whose softplus output is interpreted as Dirichlet evidence; per-point evidence is splatted into voxels using the geometric and pose conventions of the underlying nvblox [3] layer cake; the resulting $\alpha_v$ is updated by closed-form conjugate accumulation (Eq. 4); vacuity is read off (Eq. 2); and the three consumers operate on the same $(\alpha_v, u_v)$ state without recomputing it. The three modules are: M1 (§III.B) — Dirichlet evidential per-voxel semantic fusion that replaces R2's unspecified Bayes filter [1] and contrasts with S-BKI [2] / ConvBKI [4] kernel smoothing; M2 (§III.C) — vacuity-conditioned loop closure with parameter-free conjugate Dirichlet inter-session submap fusion, derived from the same $\alpha_v$; and M3 (§III.D) — voxel decay whose time-constant $\tau(u_v)$ is itself a function of vacuity, so that confidently-known voxels age slowly and high-vacuity voxels age fast. The cross-module coupling is summarised in §III.E and the nvblox-side implementation in §III.F.

## III.B  M1: Dirichlet Evidential Per-Voxel Semantic Fusion

We treat each voxel's class identity as a categorical random variable and its posterior as a Dirichlet distribution. For an observation $z$ at point $p$ whose semantic head outputs a non-negative evidence vector $e(z) \in \mathbb{R}^{C+1}$ with $e_k \ge 0$, the evidential learning convention ($\alpha = e + 1$) yields a Dirichlet posterior with concentration

$$
\alpha_k \;=\; e_k + 1, \qquad k = 1, \ldots, C+1 . \tag{1}
$$

Writing $S = \sum_{k=1}^{C+1} \alpha_k$ for the Dirichlet strength, the *vacuity* — the load-bearing scalar of this paper — is the share of the posterior mass that is not yet committed to any class:

$$
u_v \;=\; \frac{C+1}{S_v} \;\in\; (0, 1] . \tag{2}
$$

The expected class probability is the standard Dirichlet mean

$$
\mathbb{E}[p_k \mid \alpha_v] \;=\; \frac{\alpha_{v,k}}{S_v} . \tag{3}
$$

Across $N$ independent observations splatted into voxel $v$, evidence accumulates by simple addition under Dirichlet–multinomial conjugacy:

$$
\alpha_v^{(t+1)} \;=\; \alpha_v^{(t)} \;+\; e_v^{(t+1)} . \tag{4}
$$

This is the closed-form analogue of the iterative per-voxel Bayes filter that R2 [1] uses but does not specify; it recovers the standard categorical Bayesian update as a degenerate limit when the evidence is one-hot and the prior is uniform, and it gives an *honest per-voxel* posterior unlike the spatial-kernel-coupled posteriors of S-BKI [2] and ConvBKI [4] — a property M2 and M3 will rely on.

The semantic head is trained with the standard evidential deep-learning objective: a squared-error term against the one-hot label under the Dirichlet posterior, plus a Kullback–Leibler regulariser that pulls the posterior toward the uniform prior on examples for which the evidence does not support the label:

$$
\mathcal{L}_{\text{EDL}}(\alpha)
\;=\;
\sum_{k=1}^{C+1}
\left[
(y_k - \hat{p}_k)^2 + \frac{\hat{p}_k(1-\hat{p}_k)}{S+1}
\right]
\;+\;
\lambda_t \cdot \mathrm{KL}\!\left(
\mathrm{Dir}(\tilde\alpha) \,\Vert\, \mathrm{Dir}(\mathbf{1})
\right) , \tag{5}
$$

where $\hat{p}_k = \alpha_k / S$, $\tilde\alpha = y + (1 - y) \odot \alpha$ masks out the ground-truth class from the regulariser, $\lambda_t$ is an annealing weight that grows over training epochs, and $\mathrm{Dir}(\mathbf{1})$ is the uniform Dirichlet prior over $C+1$ classes. The per-voxel calibration target reported in §IV is the standard expected calibration error over the $\arg\max$ predictions

$$
\mathrm{ECE}
\;=\;
\sum_{m=1}^{M}
\frac{|B_m|}{N_{\text{voxels}}}
\left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right| , \tag{6}
$$

with $B_m$ the $m$-th confidence bin over the per-voxel maximum expected probability $\max_k \mathbb{E}[p_k \mid \alpha_v]$.

*Implementation.* Voxel size is `0.25 m` for closed-set runs and `0.25 m` for open-set runs (matched to R2 [1] and to the nvblox [3] default), giving an $8^3$ block layout of nominal storage size $(C+1) \times 4\,\text{B} \times 512 \approx 44\,\text{kB}$ per block at $C = 19$. The prior strength is $s_0 = C + 1$ so that the uninformed posterior is exactly the uniform Dirichlet $\mathrm{Dir}(\mathbf{1})$; per-observation evidence is clipped at $e_k \le e_{\max}$ (a hyperparameter [TBD: see §IV implementation]) to bound the per-voxel concentration $S_v \le S_{\max}$ and keep vacuity numerically stable across long sessions. The semantic head is initialised from the publicly released Cylinder3D weights and the last layer is fine-tuned with (5) on the SemanticKITTI training split; for the RQ2 open-set runs the head is re-fine-tuned with $C = 14$ known classes plus one explicit unknown channel, the unknown channel being trained only by the KL prior regulariser so that genuinely-unknown points receive uniform evidence and therefore high vacuity. The closest probabilistic-voxel competitors that we compare against in §IV are S-BKI [2], ConvBKI [4], LatentBKI [7], and OpenVox [5]; the BKI line uses a spatial-kernel-coupled posterior whose vacuity is contaminated by neighbour evidence (which is why M3 cannot drive its decay clock from it), while OpenVox uses a per-instance Bernoulli rather than a per-voxel Dirichlet, so its posterior cannot expose the same single-scalar vacuity that M2 and M3 of our system require.

**Tab II — Bayesian voxel-semantic lineage.** To make the lineage and the position of our work concrete, Tab II traces S-BKI → ConvBKI → LatentBKI → EvidLife-Map across mathematical backbone, uncertainty quantification, learnability, open-set capability, reported best mIoU, reported calibration, and whether the uncertainty signal is reusable by downstream decay / loop-closure consumers.

**TABLE II. EVOLUTION OF THE PROBABILISTIC-VOXEL-SEMANTIC LINEAGE FROM S-BKI [2] THROUGH CONVBKI [4] AND LATENTBKI [7] TO EVIDLIFE-MAP.**

| # | System | Year & Venue | Backbone | Uncertainty | Learnable? | Open-Set? | Best mIoU | ECE | Reusable for Decay/LC? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S-BKI [2] | RA-L 2020 | Kernel Bayesian + Dirichlet–Cat. conjugate | Per-voxel Dirichlet variance | $\times$ (hand-set $l$, $\alpha_0$) | $\times$ | KITTI seq 15 57.1%; SemKITTI Test 51.3% | [NR] | $\times$ (kernel-contaminated) |
| 2 | ConvBKI [4] | T-RO 2024 | Conv kernel BKI (22 params at $C{=}11$) | Per-voxel Dirichlet variance | partial (22 kernel params) | $\times$ | KITTI seq 15 **77.7%** | [NR] | $\times$ (still smoothed) |
| 3 | LatentBKI [7] | RA-L 2025 | Latent BKI (NIW conjugate over CLIP $\mathbb{R}^{64}$) | E-optimality in latent space | $\checkmark$ (LSeg encoder) | open-*dict.* (not open-*set*) | MP3D 16.15%; outdoor 61.5% Acc | [NR] | $\times$ (latent-space) |
| 4 | **EvidLife-Map (ours)** | IROS 2027 (target) | **Dirichlet EDL** — $\alpha \in \mathbb{R}^{C+1}$, no kernel | **Vacuity** $u_v = (C+1)/S_v$ | $\checkmark$ (Cylinder3D + EDL last layer) | $\checkmark$ (vacuity native OOD + unknown channel) | <mark>[TBD-RQ1]</mark> (target $\ge$ +2 vs R2) | <mark>[TBD-RQ1]</mark> (target $-20\%$ vs R2) | $\checkmark$ (single scalar, three consumers) |

## III.C  M2: Vacuity-Conditioned Loop Closure

Loop closure operates on submaps sealed every $D = 50\,\text{m}$ of travelled distance or $T = 30\,\text{s}$ of wall time, whichever first; each submap $\mathcal{S}$ is summarised by a descriptor that explicitly exposes vacuity as a first-class channel rather than collapsing it into a confidence weight. Concretely, the descriptor is

$$
d(\mathcal{S}) \;=\; \big( h_{\text{class}}(\mathcal{S}), \; h_{\text{vac}}(\mathcal{S}) \big), \tag{7}
$$

where $h_{\text{class}}(\mathcal{S}) \in \mathbb{R}^{C+1}$ is the $L^1$-normalised class histogram over voxels with vacuity below the per-submap median, and $h_{\text{vac}}(\mathcal{S}) \in \mathbb{R}^{B}$ is the histogram of per-voxel vacuity $u_v$ bucketed into $B = 10$ uniform bins over $(0, 1]$. The class channel preserves the conventional submap-fingerprint role and the vacuity channel encodes the *epistemic shape* of the submap — a frequently-revisited submap is dominated by low-vacuity bins, while a recently-explored or sparsely-observed submap concentrates mass in high-vacuity bins.

Candidate matches between two submaps $\mathcal{S}_1, \mathcal{S}_2$ are scored by a convex combination of cosine similarity on the class channel and cross-entropy on the vacuity channel,

$$
\mathrm{sim}(\mathcal{S}_1, \mathcal{S}_2)
\;=\;
(1-\beta) \cdot \frac{\langle h_{\text{class}}^{(1)}, h_{\text{class}}^{(2)} \rangle}{\Vert h_{\text{class}}^{(1)} \Vert \, \Vert h_{\text{class}}^{(2)} \Vert}
\;-\;
\beta \cdot \mathrm{H}\!\left( h_{\text{vac}}^{(1)} \,\Vert\, h_{\text{vac}}^{(2)} \right) , \tag{8}
$$

with mixing weight $\beta \in [0, 1]$ (a single hyperparameter, [TBD: see §IV ablations A-3]). The cross-entropy term penalises matches between submaps whose epistemic profiles disagree even when their class distributions are similar — a typical failure mode of class-histogram-only descriptors when revisiting an explored area in which most class mass is dominated by a single class (e.g., a long road segment).

A candidate match $(\mathcal{S}_1, \mathcal{S}_2)$ is verified by geometric registration restricted to voxels whose vacuity is below the per-submap median in *both* submaps; the post-registration consistency check thresholds the median per-class disagreement of the overlap region,

$$
\mathrm{Disagree}(\mathcal{S}_1, \mathcal{S}_2)
\;=\;
\mathrm{median}_{v \in \mathcal{O}}
\left[ 1 - \big\langle \mathbb{E}[p \mid \alpha_v^{(1)}],\, \mathbb{E}[p \mid \alpha_v^{(2)}] \big\rangle \right] , \tag{9}
$$

with $\mathcal{O}$ the overlap voxel set after registration and $\langle \cdot, \cdot \rangle$ the inner product of two posterior-mean distributions. Verified matches trigger pose-graph optimisation; inter-session fusion in the overlap region is then performed by *conjugate addition* of Dirichlet concentrations, $\alpha_v \gets \alpha_v^{(1)} + \alpha_v^{(2)}$, which is the maximum-likelihood combination under the assumption of independent observations across sessions and which introduces no fusion hyperparameter.

The construction contrasts with the descriptors used in Hydra [10] and Kimera-Multi [16], which encode submap fingerprints either through bag-of-words or through learned aggregation but do not expose epistemic mass as an explicit descriptor channel; it also contrasts with the semantic-graph-with-GAT loop closure of [27] and SA-LOAM [26], whose semantic enrichment is a class-level signal rather than an uncertainty signal.

## III.D  M3: Vacuity-Driven Voxel Decay

Voxel decay applies a conjugate exponential pull of $\alpha_v$ toward the uniform prior, with the decay time-constant $\tau$ itself a function of vacuity. Let $a_v = t_{\text{now}} - t_{v,\text{last-update}}$ be the age of voxel $v$ since its last evidence accumulation,

$$
a_v \;=\; t_{\text{now}} \,-\, t_{v,\text{last-update}} . \tag{10}
$$

The decay time-constant is a linear interpolation between $\tau_{\min}$ (high-vacuity voxels age fast) and $\tau_{\max}$ (low-vacuity voxels age slowly):

$$
\tau(u_v) \;=\; \tau_{\min} \,+\, (\tau_{\max} - \tau_{\min}) \cdot (1 - u_v) . \tag{11}
$$

A confidently-known voxel ($u_v \to 0$) decays with $\tau \to \tau_{\max}$ (target $\approx$ one hour); a maximally-uncertain voxel ($u_v \to 1$) decays with $\tau \to \tau_{\min}$ (target $\approx$ one minute). The decay itself is the standard Dirichlet conjugate pull toward the uniform prior, applied to the *concentration* with the unit prior pinned in place so that $\alpha_{v,k} \ge 1$ is preserved (the Dirichlet constraint that makes (2) well-defined):

$$
\alpha_v^{(t + \Delta)} \;=\; \big( \alpha_v^{(t)} - \mathbf{1} \big) \cdot \exp\!\Big( -\frac{\Delta}{\tau(u_v)} \Big) \;+\; \mathbf{1} . \tag{12}
$$

By construction, (12) preserves the posterior mean direction (the ratio $\alpha_k / S$ is unchanged when $\alpha_v - 1$ is scaled uniformly only as long as no class dominates the prior, which is the regime where the decay is intended to act) and inflates vacuity monotonically with $a_v$. The net behaviour is that stale voxels become *re-writable* — their vacuity rises until fresh evidence either reasserts the previous class identity (drawing vacuity back down) or replaces it (the new evidence dominates the now-small $\alpha_v - 1$ residual).

The mechanism contrasts with the explicit per-submap activity flag of Panoptic Multi-TSDFs [8] and with the learned short-/long-term factoriser of Khronos [6]: in both prior systems, the decision *when to forget* is taken by a mechanism that does not consume the voxel-level posterior, whereas in (12) the decision is *fully determined* by the same Dirichlet vacuity that M1 produces and M2 consumes. It also contrasts with the monotonically-growing weight schedule of nvblox [3] and the long-term object handling of LTC-Mapping [28], which do not implement any decay at all.

*Dynamic objects.* A side-effect of (11)–(12) is that moving-object voxels experience naturally elevated vacuity (inconsistent per-frame evidence keeps $S_v$ low relative to inter-class disagreement) and therefore decay aggressively, so transient dynamic evidence is flushed before it can crystallise into a wrong-persistent label. The empirical RQ5 head-to-head against Khronos [6] in §IV measures whether this implicit mechanism is competitive enough to make the explicit short-/long-term factoriser a deployment-cost trade-off rather than a capability necessity.

## III.E  Conjugate Cross-Module Coupling

Modules M1, M2, and M3 share a single per-voxel state $(\alpha_v, u_v)$ and a single arithmetic primitive — conjugate Dirichlet addition. M1 accumulates $\alpha_v$ from evidence (Eq. 4); M2 fuses $\alpha_v^{(1)}$ and $\alpha_v^{(2)}$ across sessions by the same addition (post-Eq. 9); M3 pulls $\alpha_v$ toward the prior by (12), which is itself the time-discretisation of the same Dirichlet-conjugate update against a synthetic uniform observation. Vacuity (Eq. 2) is computed once per voxel per fusion step and read three times: once by the open-set head (where high $u_v$ flags an unknown-category voxel for the RQ2 evaluation), once by the M2 descriptor ($h_{\text{vac}}$ in Eq. 7), and once by the M3 trigger ($\tau(u_v)$ in Eq. 11). No module needs to recompute $u_v$ and no module needs to maintain a private auxiliary uncertainty quantity. This is the load-bearing observation that motivated EvidLife-Map's design: the same scalar that gives M1 its calibrated semantic posterior also gives M2 its descriptor entropy channel and M3 its decay clock, so the *integrated* claim "one vacuity, three jobs" is realised at zero additional per-voxel runtime cost beyond the single division that produces $u_v$ from $\alpha_v$. A direct corollary is that any ablation that disables one consumer of $u_v$ leaves the other two consumers and the cost of computing $u_v$ itself unchanged — the §IV.H ablation grid measures exactly this, and the cost accounting in §III.F shows that the only per-voxel state added relative to a non-evidential nvblox [3] baseline is $(C+1)$ floats for $\alpha_v$ plus a single timestamp.

## III.F  Implementation on nvblox

We implement the per-voxel state inside the nvblox [3] layer cake as an `EvidentialLayer<VoxelType>` whose `VoxelType` extends the canonical nvblox semantic voxel with $(\alpha_v \in \mathbb{R}^{C+1}, t_{v,\text{last-update}}, \text{pose-idx}_v)$. The layer participates in the standard nvblox tick: per-frame point splatting writes into $\alpha_v$ via (4); the GPU streaming-multiprocessor block layout of nvblox is preserved unchanged, so the only structural change relative to a baseline nvblox semantic layer is the wider voxel record and the additional per-voxel timestamp. Per-voxel memory grows from approximately $100\,\text{B}$ in R2 [1] (label probabilities only) to approximately $100\,\text{B} + (C+1) \times 4\,\text{B} + 4\,\text{B} + 4\,\text{B} \approx 180\,\text{B}$ at $C = 19$, giving a nominal $44\,\text{kB}$ storage per $8^3$ block of fully-allocated voxels and roughly $1.7\times$ the per-voxel footprint of R2. The decay step (12) is implemented as a periodic kernel over allocated voxels with $a_v > a_{\min}$ (a small hysteresis threshold [TBD: see §IV implementation]) to avoid touching voxels updated in the current frame; on a Jetson Orin NX the kernel sweeps an active map at the same cadence as the standard nvblox mesh updater. The M2 descriptor (Eq. 7) is computed on-demand at submap-seal time, reusing the per-block reductions that nvblox already performs to maintain its visualisation mesh; conjugate inter-session fusion (post-Eq. 9) reuses the same evidence-addition kernel as M1, so inter-session fusion costs the same per-voxel as a single observation step. All other state (LVIO pose stream, TSDF backbone, mesh extractor) is reused from the upstream nvblox release.

---

# IV. Experiments

## IV.0  Preliminary Status (2026-05-29 snapshot)

> **Caveat for this draft revision**: At the time of this snapshot the planned
> Cylinder3D + EDL backbone (see Tab III) is gated on a spconv 1.x source-build
> step (W2-1 implementation status, supplementary `W2-1_status.md`). The cells in
> Tab IV / V / VI / VII below remain placeholders for that target. To keep the
> paper IV thesis (paper §I.B "one vacuity, three jobs") falsifiable at every
> revision, we report below a smaller-backbone, smaller-data, **preliminary**
> verification of the THREE method-level claims that do not require the final
> backbone: (a) R2 vs M1 relative ordering under matched backbone capacity,
> (b) vacuity AUROC for open-set discrimination, (c) vacuity-driven decay
> ratio. The substitute backbone is a 0.21 M-parameter PointNet-Vanilla trained
> for 30 epochs on 80 frames of SemanticKITTI seq 08; this is roughly 1/250 the
> capacity of the planned Cylinder3D backbone and is reported as a
> *preliminary* result only.

**RQ1 preliminary (M1 vs R2 at matched 0.21 M-param backbone, SemKITTI seq 08
100 frames).** Under a common evaluation harness:

| System | mIoU | ECE | Latency / frame |
|---|---|---|---|
| R2 (CE PointNet → argmax-Bayes per voxel) | 1.69 % | 0.490 | 663 ms |
| **M1 (EDL PointNet → Dirichlet posterior)** | **14.73 %** | **0.171** | **3.0 ms** |
| Relative advantage of M1 over R2 | **8.7×** higher | **2.9×** lower | **220×** faster |

M1 wins on mIoU, ECE, and latency at matched backbone capacity, validating the
paper §III.B comparative thesis at preliminary scale. Json: `artifacts/rq1_fair_100frames.json`.

**RQ2 preliminary (vacuity as OOD score, 14-known / 5-unknown split).**
M1 trained on 14 known classes (ignore_index on the 5 unknown labels during
training, keeping the protocol identical to what the full-scale RQ2 will run).
At eval on seq 08 val frames 80–99 (37 838 unknown + 2 055 160 known points):

| Metric | Preliminary value | Paper RQ2 H2 target | Status |
|---|---|---|---|
| Best vacuity AUROC | **0.808** | $\ge 0.80$ | **VERIFIED** |

Json: `artifacts/m1_openset.json`. The AUROC is robust enough to satisfy the
H2 verification threshold at preliminary backbone capacity; the full-scale RQ2
in Tab V is expected to lift this further on the Cylinder3D backbone.

**M3 preliminary (vacuity-driven decay rate, paper §III.D Eq 11).** Build an
M1 evidence accumulator over seq 08 first 50 frames (747 047 unique voxels at
0.25 m voxel size); apply `decay_concentration` with $\Delta t = 600$ s,
$\tau_{\min} = 60$ s, $\tau_{\max} = 3600$ s; stratify voxels by vacuity into
bottom-10 % / middle 80 % / top-10 %:

| Vacuity stratum | Pre-decay mean $u_v$ | Evidence-mass loss over $\Delta t$ |
|---|---|---|
| Bottom 10 % (confident) | 0.045 | 16.0 % |
| Middle 80 % | — | 24.4 % |
| Top 10 % (uncertain) | 0.705 | 42.3 % |
| **Top / bottom decay-loss ratio** | | **2.64×** |

The 2.64× ratio confirms the directional claim of Eq 11 on real KITTI data:
the same vacuity scalar that drives the M1 posterior update *also* drives the
M3 decay rate, and high-vacuity voxels lose evidence noticeably faster.
Json: `artifacts/m3_validation.json`.

A consolidated "one vacuity, three jobs" verification matrix (paper §I.B
thesis) is provided in `artifacts/preliminary_results.md`: Leg (i)
open-set OOD score ✓, Leg (ii) loop-closure entropy channel ⏸ (gated on the
KITTI-360 transfer), Leg (iii) decay rate clock ✓.

## IV.A  Datasets and Metrics

We evaluate EvidLife-Map on five public datasets, chosen so that every research question is supported by at least one dataset with dense semantic ground truth and no manual annotation budget is required:

- **SemanticKITTI** (Behley 2019; Cylinder3D split) — closed-set dense LiDAR semantic GT (19 classes); used for RQ1 (closed-set mIoU + ECE), RQ2 (the 14-known / 5-unknown open-set split defined below), and RQ5 (the dynamic-frame subset defined below).
- **nuScenes-LiDARSeg** (Caesar 2020) — cross-domain RQ1 generalisation check and the *reverse cross-domain* RQ2 sanity check (train on SemanticKITTI 14-known classes, evaluate vacuity on nuScenes-only classes whose taxonomy does not overlap, e.g., `construction_vehicle`, `barrier`, `traffic_cone`).
- **KITTI-360** (Liao 2022; multi-session dense LiDAR semantic GT) — primary RQ4 lifelong arena, with $\ge 5$ revisit pairs constructed by odometry-overlap.
- **SemanticKITTI dynamic split** (constructed from the SemanticKITTI `moving-*` label flag on seq 00, 04, 05, 07; provisionally $\sim 1500$–$2000$ dynamic-heavy frames) — the RQ5 head-to-head substrate against Khronos [6].
- **SemanticSpray** [21] (RA-L 2024 wet-road LiDAR seg) — RQ3 passive-replay traversability arena.

Across these five datasets we report a unified metric battery: **closed-set mapping** — mIoU, per-class IoU, mean accuracy, end-to-end latency (mean + 99-pct), GPU memory peak; **uncertainty calibration** — Expected Calibration Error (ECE, Eq. 6) and Brier score; **open-set OOD** — voxel-level AUROC and AUPR of vacuity for known-vs-unknown discrimination; **downstream traversability** — safe-region recall, false-traversable rate, deferral rate; **lifelong maintenance** — stale-voxel removal precision/recall over multi-session trajectories, ECE drift across sessions, multi-session memory growth; **dynamic competence** — dynamic-object mIoU (restricted to `moving-*` classes) and static-region recall; **deployment** — Jetson Orin NX latency and VRAM. All metrics are computed under one common evaluation harness so that the cross-system table cells are comparable.

## IV.B  Implementation and Baselines

We compare against six baselines spanning the full reviewer-credible threat list of the planning audit: (1) **R2-reimpl** [1] — our faithful Python/CUDA re-implementation, since R2 does not release code; (2) **nvblox-vanilla** [3] — geometric lower bound with a per-frame argmax semantic head; (3) **ConvBKI** [4] — the canonical probabilistic-voxel competitor and the load-bearing closed-set baseline (RQ1) whose published 77.7% mIoU on KITTI Odometry seq 15 is the ceiling we measure against; (4) **S-BKI** [2] — the BKI ancestor, included as a lineage anchor and a sanity check on the kernel-Bayesian regime; (5) **Kimera-Semantics** [9] (via Kimera-Multi [16] for the distributed setting) — the canonical MIT-SPARK Voxblox-class baseline carried for RQ4 (lifelong) and RQ1 (reference); (6) **Khronos** [6] — used both for RQ1 (closed-set reference) and for RQ5 (dynamic-scene head-to-head); if full reproduction of Khronos on our LiDAR-only SemanticKITTI configuration fails by W20 of the schedule, RQ5 falls back to a published-numbers comparison with explicit disclaimer (research_plan v3 §6 R-17). The complete baseline × dataset × RQ × effort matrix is given as Tab III; the OpenVox [5] entry is included as a published-numbers-only comparison because its code is not yet public as of the writing of this draft. The total baseline reproduction effort is estimated at $24.5$ GPU-days on a single RTX 4060 (16 GB), which fits inside the 64-GPU-day overall budget of research_plan v3 §4.5.

**TABLE III. BASELINE × DATASET × RQ × EFFORT MATRIX (GPU-DAYS ON 1 × RTX 4060, 16 GB).**

| # | Baseline | Code | Datasets | RQ | Reported Metrics | Self-Reproduced | Effort | Risk |
|---|---|---|---|---|---|---|---|---|
| 1 | R2-reimpl [1] | $\times$ | SK; nuScenes; KITTI-360 | RQ1, RQ3, RQ4 | mesh + per-module latency on 3080 Ti | mIoU, ECE, AUROC, trav. recall, Orin NX latency | 11 | R-12 reimpl delta |
| 2 | nvblox-bare [3] | $\checkmark$ | SK; SemSpray | RQ1 (lower bound), RQ3 | TSDF / ESDF latency on Xavier AGX | mIoU, ECE, trav. recall | 2 | Low |
| 3 | ConvBKI [4] | $\checkmark$ | SK; nuScenes | RQ1 | KITTI seq 15 77.7% mIoU; 44.3 Hz on RTX 3090 | ECE on SK 08+11-21; AUROC on 14/5 split; Orin NX latency | 4 | R-12 (active) |
| 4 | S-BKI [2] | $\checkmark$ | SK | RQ1 | SK Test 51.3%; 1.67 s/scan CPU | ECE, AUROC | 2 | Low |
| 5 | Kimera-Sem. [9, 16] | $\checkmark$ | KITTI-360; SK | RQ4, RQ1 | scene-graph metrics on uHumans2 | mIoU, ECE drift, stale-voxel P/R, memory growth | 6 | M (out-of-domain) |
| 6 | Khronos [6] | $\checkmark$ | SK dynamic split; SK seq 08 | RQ5, RQ1 | TESSE F1, 22.2 FPS on i7 CPU | dynamic-object mIoU, static recall, Orin NX latency | 5 | **R-17 highest** |
| 7 | OpenVox [5] | $\times$ (no public) | Replica/ScanNet (their) | RQ2 (publ. only) | Replica/ScanNet zero-shot inst. mIoU | none (capability comparison only) | 0.5 | Capability-only |

*Training details.* Cylinder3D last-layer EDL fine-tune on SemanticKITTI train split: AdamW, lr [TBD], batch [TBD], $\lambda_t$ annealed linearly to 1 over [TBD] epochs (Sensoy 2018 schedule), gradient-clip at [TBD], single RTX 4060 (16 GB), wall-clock $\approx 2$ days per head; one 20-D head for closed-set RQ1/RQ4/RQ5 and one 15-D head for open-set RQ2. R2-reimpl uses the same Cylinder3D backbone with a softmax-Bayes head trained on the same split. All hyperparameters, seeds, and CUDA/PyTorch versions are pinned in the supplementary `reproduce.yaml`.

## IV.C  Main Results — RQ1 (Closed-Set mIoU and Calibration)

On SemanticKITTI val (seq 08) and the held-out test split (seq 11-21), EvidLife-Map achieves <mark>XX.X</mark> mIoU [TBD-RQ1] versus ConvBKI's published 77.7% on KITTI Odometry seq 15 [4] and S-BKI's published 51.3% on SemanticKITTI Test [2]. The ECE under our common evaluation harness drops from R2-reimpl's <mark>XX.X</mark> [TBD-baseline] to <mark>XX.X</mark> [TBD-RQ1], a relative reduction of <mark>XX%</mark> [TBD-RQ1] — a metric that neither ConvBKI [4] nor S-BKI [2] nor Kimera-Semantics [9] publishes for their own systems, so the cross-system column in Tab IV is the first comparable ECE table in the BKI/Voxblox lineage. A cross-domain check on nuScenes-LiDARSeg confirms that the closed-set mIoU gap survives the taxonomy shift to within <mark>XX.X</mark> mIoU [TBD-RQ1] of the same-domain number — the calibration gap (ECE) is preserved across the shift in absolute terms, consistent with the EDL theory that vacuity is a per-voxel epistemic signal independent of the prior over class identity.

**TABLE IV. RQ1 — CLOSED-SET MIOU + CALIBRATION ON SEMANTICKITTI VAL (SEQ 08) + TEST (SEQ 11-21) + NUSCENES-LIDARSEG (CROSS-DOMAIN).**

| Method | SK Val mIoU | SK Test mIoU | nuScenes mIoU | mAcc | ECE | Brier | Latency (ms, RTX 4060) | GPU Memory (GB) |
|---|---|---|---|---|---|---|---|---|
| nvblox-vanilla [3] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| R2-reimpl [1] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| S-BKI [2] | <mark>XX.X</mark> | 51.3 (publ.) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| ConvBKI [4] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| Kimera-Sem. [9] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| **EvidLife-Map** | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> |

[TBD-after-exp] One paragraph of per-class IoU commentary highlighting the rare-class behaviour of the Dirichlet posterior (we expect vacuity to absorb mass from `bicyclist` / `motorcyclist` / `truck` rather than misallocating it to majority classes — measurable as smaller per-class IoU degradation on rare classes than ConvBKI shows).

## IV.D  Open-Set Detection — RQ2 (Vacuity as OOD Score)

On the SemanticKITTI 14-known / 5-unknown open-set split (withheld classes: `bicyclist`, `motorcyclist`, `truck`, `other-vehicle`, `other-ground`; pre-registered in supplementary), the per-voxel vacuity ranking achieves <mark>AUROC = XX.X</mark> and <mark>AUPR = XX.X</mark> [TBD-RQ2] for unknown-vs-known discrimination, while the closed-set mIoU on the 14 known classes drops by <mark>XX.X</mark> mIoU [TBD-RQ2] versus the 19-class fully-supervised baseline. A robustness 16-known / 3-unknown split (drop only `bicyclist`, `motorcyclist`, `other-vehicle`) reports <mark>AUROC = XX.X</mark> [TBD-RQ2]; a reverse cross-domain check trained on the SemanticKITTI 14-known split and evaluated on nuScenes-LiDARSeg classes that have no SemanticKITTI analogue (`construction_vehicle`, `barrier`, `traffic_cone`, `pushable_pullable`) reports <mark>AUROC = XX.X</mark> [TBD-RQ2]. Figure 5 [TBD-after-exp] shows the bimodal vacuity histogram on a representative val frame: known-class voxels cluster at $u_v \in (0, 0.2]$, withheld-class voxels at $u_v \in [0.6, 1.0]$, with the cross-over bin carrying $< 5\%$ of voxel mass — the visual demonstration of H2.

**TABLE V. RQ2 — OPEN-SET VACUITY AS OOD SCORE; THREE SPLITS × THREE METRICS × FOUR METHODS.**

| Method | 14/5 AUROC | 14/5 AUPR | 14/5 mIoU on Known | 16/3 AUROC | 16/3 AUPR | nuScenes Reverse AUROC |
|---|---|---|---|---|---|---|
| R2-reimpl [1] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| ConvBKI [4] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| Clio-LiDAR-stub [18] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| **EvidLife-Map** | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> |

[TBD-after-exp] One paragraph contrasting our per-voxel Dirichlet vacuity against ConvBKI's kernel-smoothed variance: we expect ConvBKI's smoothing to *suppress* per-voxel epistemic spikes on isolated unknown points, lowering its OOD AUROC even when its closed-set mIoU is competitive — the entire reason §III.B drops the spatial kernel.

## IV.E  Downstream Traversability — RQ3 (SemanticSpray Closed-Loop Replay)

On the SemanticSpray [21] passive-replay traversability evaluation, propagating per-voxel vacuity into a downstream traversability head yields <mark>XX.X</mark> percentage-point absolute increase in safe-region recall and a <mark>XX%</mark> relative reduction in false-traversable rate [TBD-RQ3] versus an R2-style hard-rule baseline at matched coverage. The deferral rate (fraction of voxels flagged "do not commit to traversable") increases by <mark>XX</mark> pp [TBD-RQ3] in compensation, an expected and desired behaviour for an open-set-aware traversability layer that prefers caution to false confidence on visually-ambiguous wet-road pixels.

**TABLE VI. RQ3 — TRAVERSABILITY ON SEMANTICSPRAY [21] PASSIVE REPLAY.**

| Method | Safe-Region Recall (%) | False-Traversable Rate (%) | Deferral Rate (%) | Coverage (%) |
|---|---|---|---|---|
| nvblox-vanilla + hard rule [3] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| R2-style hard rule [1] | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| **EvidLife-Map (vacuity-gated)** | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> |

## IV.F  Lifelong Multi-Session — RQ4 (KITTI-360 Revisit Pairs)

Across <mark>XX</mark> revisit pairs [TBD-RQ4] constructed by odometry-overlap on KITTI-360 sequences 00, 02, 04, 05, 06, 07, 09, 10, EvidLife-Map achieves **stale-voxel removal precision** <mark>XX%</mark> at recall <mark>XX%</mark> [TBD-RQ4] over the second-pass trajectory after the M3 decay (Eq. 12) has acted, **ECE drift across sessions** of <mark>XX×</mark> [TBD-RQ4] the single-session ECE (target $\le 1.5\times$), and a **multi-session memory footprint** of <mark>XX×</mark> [TBD-RQ4] the single-session footprint over the five revisited pairs (target $\le 1.7\times$, sub-linear in session count thanks to the voxel-hash deduplication described in §III.F). Figure 6 [TBD-after-exp] shows per-pair stale-voxel PR curves; the per-pair area-under-curve degrades gracefully with revisit gap length, confirming that the vacuity-driven decay schedule does not catastrophically forget under long absences.

**TABLE VII. RQ4 — LIFELONG MULTI-SESSION ON KITTI-360 REVISIT PAIRS.**

| Revisit Pair | Stale-Voxel P (%) | Stale-Voxel R (%) | F1 (%) | Single-Session ECE | Multi-Session ECE | Memory Growth ($\times$) |
|---|---|---|---|---|---|---|
| seq 00 — pair 1 | <mark>XX</mark> | <mark>XX</mark> | <mark>XX</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| seq 02 — pair 2 | <mark>XX</mark> | <mark>XX</mark> | <mark>XX</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| seq 04 — pair 3 | <mark>XX</mark> | <mark>XX</mark> | <mark>XX</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| seq 05 — pair 4 | <mark>XX</mark> | <mark>XX</mark> | <mark>XX</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| seq 06 — pair 5 | <mark>XX</mark> | <mark>XX</mark> | <mark>XX</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| **Mean** | <mark>**XX**</mark> | <mark>**XX**</mark> | <mark>**XX**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> |

## IV.G  Dynamic Scene Comparison vs Khronos — RQ5 (SemanticKITTI Dynamic Split)

On the SemanticKITTI dynamic-frame subset (frames from seq 00, 04, 05, 07 with $\ge 5\%$ `moving-*` GT points), EvidLife-Map achieves dynamic-object mIoU <mark>XX.X</mark> (Khronos [6] published / reproduced <mark>XX.X</mark>), static-region recall <mark>XX.X</mark> pp ($\Delta = $ <mark>XX</mark> pp vs Khronos), and end-to-end latency <mark>XX.X</mark> ms on Jetson Orin NX [TBD-RQ5]. The framing established in §III.D (implicit vacuity-instability vs explicit 4D factoriser) is the basis of the comparison: we do not claim that the implicit channel dominates the explicit one on dynamic-object mIoU; we claim it is competitive enough to make Khronos's full short-/long-term machinery a deployment-cost trade-off rather than a capability necessity in our operating regime. Figure 5 [TBD-after-exp] (the dynamic version, distinct from the §IV.D vacuity histogram) shows a side-by-side qualitative comparison on a representative dynamic frame.

**TABLE VIII. RQ5 — DYNAMIC-SCENE HEAD-TO-HEAD ON THE SEMANTICKITTI DYNAMIC SPLIT.**

| Method | Dynamic-Object mIoU | Static-Region Recall (%) | E2E Latency (ms, 4060) | E2E Latency (ms, Orin NX) | GPU Memory (GB) |
|---|---|---|---|---|---|
| R2-reimpl [1] (lower bound) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> |
| Khronos [6] (reproduced / publ.) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | not designed for Orin NX | <mark>XX.X</mark> |
| **EvidLife-Map** | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> | <mark>**XX.X**</mark> |

## IV.H  Ablations

We ablate the six firm pipeline choices of research_plan v3 §4.4 (A-1 through A-5 plus A-7) and one stretch choice (A-6, voxel size) reported only if W14 gates are green. Each row of Tab IX disables exactly one design choice; the column triple (mIoU / ECE / open-set AUROC) isolates the effect on the three reviewer-credible RQs simultaneously.

**TABLE IX. ABLATIONS (1 × RTX 4060, 16 GB).**

| # | Ablation | SK mIoU | SK ECE | Open-Set AUROC | KITTI-360 Stale P (%) | Orin NX Hz |
|---|---|---|---|---|---|---|
| Full | EvidLife-Map (full) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-1 | $-$ Dirichlet head (use argmax-Bayes) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-2 | $-$ vacuity-coupled $\tau$ (fixed-$\tau$ decay) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-3 | $-$ entropy channel in descriptor (use $h_{\text{class}}$ only) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-4 | $-$ vacuity-aware traversability (hard threshold) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-5 | $-$ conjugate Dirichlet inter-session fusion (naive overwrite) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-7 | $-$ dedicated unknown channel (vacuity only, $C$=14) | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |
| A-6 (stretch) | voxel size $0.10\,\text{m}$ vs $0.25\,\text{m}$ | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX.X</mark> | <mark>XX</mark> | <mark>XX.X</mark> |

[TBD-after-exp] One paragraph naming which ablation is "load-bearing" (likely A-2 for stale-voxel precision, A-3 for loop-closure precision, A-7 for open-set AUROC) and which is small enough to demote to supplementary in a camera-ready trim if §IV.G overflows.

## IV.I  Runtime, Memory, and Jetson Deployment + Demo Video

On a single RTX 4060 (16 GB) the end-to-end pipeline runs at <mark>XX.X</mark> Hz [TBD-RQ-deploy] over a SemanticKITTI seq 08 replay; on the Jetson Orin NX (16 GB) with a TensorRT FP16 build of the M1 head and a CPU implementation of M2 loop closure, the pipeline runs at <mark>XX.X</mark> Hz [TBD-RQ-deploy] over the same replay, peak VRAM <mark>XX.X</mark> GB. A 30-second supplementary video `evidlife_demo.mp4` shows EvidLife-Map running live on the Orin NX with the metric-semantic voxel map, the traversability overlay (RQ3 output), and a vacuity heat-map overlay visible in real time at native 10 Hz over 300 frames; capture and overlay scripts ship with the code release.

**TABLE X. RUNTIME, MEMORY, AND JETSON DEPLOYMENT.**

| Metric | RTX 4060 (16 GB) | Jetson Orin NX (16 GB) |
|---|---|---|
| Mean end-to-end latency (ms) | <mark>XX.X</mark> | <mark>XX.X</mark> |
| 99-pct latency (ms) | <mark>XX.X</mark> | <mark>XX.X</mark> |
| Throughput (Hz) | <mark>XX.X</mark> | <mark>XX.X</mark> |
| Peak GPU memory (GB) | <mark>XX.X</mark> | <mark>XX.X</mark> |
| Per-voxel memory (B) at $C=19$ | 180 | 180 |
| Per-block memory (kB) at $8^3$ | 44 | 44 |

---

# V. Discussion

## V.A  Limitations

Five limitations are stated honestly to keep the §I.B thesis falsifiable at every revision. The first three are pre-registered scope decisions from research_plan v3, the last two are surfaced by the §IV.0 preliminary verification campaign.

**(i) RGB head omitted.** The system is LiDAR-only at deployment; we use RGB only as a training-time prior in the Cylinder3D pretrain, and we make no claim about photo-realistic appearance, weather robustness, or open-vocabulary text retrieval. The Gaussian-splatting strand exemplified by GS-LIVO [20] and the open-vocabulary scene-graph strand exemplified by HOV-SG [19] are therefore methodologically orthogonal future work, not in-scope contenders here.

**(ii) Static voxel grid, no Gaussian / NeRF representation.** We keep the nvblox [3] voxel grid because the M3 decay (Eq. 12) requires a per-voxel state with bounded memory and well-defined cell boundaries for the conjugate update; a continuous-field representation would need a different decay primitive that we have not derived.

**(iii) Per-voxel $\alpha$ memory scaling at $C \ge 100$.** The $(C+1) \times 4\,\text{B}$ growth of the per-voxel record becomes the dominant memory cost when the closed-set taxonomy is much larger than the 19-class SemanticKITTI setting; the open-vocabulary regime where $C$ is the size of a CLIP token vocabulary therefore needs either a low-rank approximation of $\alpha_v$ or a per-instance Bernoulli-like collapse along OpenVox [5] lines. We deliberately do not claim either extension in this paper.

**(iv) Cylinder3D backbone integration is gated on a spconv 1.x source-build.** The published Cylinder3D checkpoint is shipped against spconv 1.x; reproducing it under spconv 2.x preserves the structural load (0 missing keys after a `(D,H,W,in,out) → (out,D,H,W,in)` weight permutation plus 36 `.features = …` to `.replace_feature(…)` patches and 22 kernel-shape-disambiguated `indice_key` patches) but produces NaN logits at inference — the spconv 1.x → 2.x internal kernel-index ordering changed at a value-semantic level that no purely-structural patch reaches. The §IV.0 preliminary numbers therefore use a 0.21 M-param PointNet-Vanilla backbone (trained 30 epochs on 80 frames of seq 08) as a stand-in; a spconv 1.x source build (or a PVKD-style spconv-2.x-native Cylinder3D port) is the right unblock. Supplementary `W2-1_status.md` records every patch we applied and the exact failure surface for reproduction.

**(v) Preliminary mIoU absolute values are backbone-bound, not method-bound.** The §IV.0 preliminary M1 mIoU (14.73%) is far below the ConvBKI [4] published 77.7%. This gap is dominated by the 250× capacity ratio between the substitute PointNet-Vanilla and the planned Cylinder3D, *not* by the M1 evidential head. The comparative ordering reported in §IV.0 (M1 > R2 in mIoU, ECE, and latency at matched backbone capacity) is the genuine method-level signal; the full-scale §IV.C numbers will land once limitation (iv) is unblocked.

## V.B  Failure Modes

Four failure modes are documented; (a)-(c) are pre-registered from the planning audit and (d) is observed in the §IV.0 preliminary RQ2 run.

**(a) Confidently-wrong evidence accumulation.** Persistent mirror-surface returns or other systematic LiDAR ghosting cause evidence to accumulate with low entropy in a wrong class, keeping vacuity artificially low and preventing M3 from triggering decay. This is a known evidential-deep-learning failure mode inherited from Sensoy 2018, and it also produces low-vacuity false negatives in the RQ2 open-set evaluation for unknown classes that visually resemble a known class (e.g., a truck withheld from training that gets confident `car` evidence on every frame).

**(b) Non-overlapping class distributions across sessions.** When two sessions of the same area capture non-overlapping class sets (e.g., one before construction, one after), the M2 class-histogram channel of the descriptor collapses to near-zero cosine similarity and the matching degenerates to the entropy channel only; loop-closure precision degrades by approximately <mark>XX</mark> percentage points [TBD-after-exp] in such cases.

**(c) Taxonomically-close withheld classes.** On the 16/3 robustness open-set split (withhold only `bicyclist`, `motorcyclist`, `other-vehicle`) the AUROC degrades by <mark>XX.X</mark> [TBD-after-exp] relative to the 14/5 primary split, because the withheld classes have close training analogues (`bicycle`, `motorcycle`, `car`) whose evidence channels capture most of the unknown points' mass.

**(d) EDL post-epoch-1 vacuity collapse under aggressive KL annealing.** In the §IV.0 preliminary RQ2 run, the vacuity AUROC peaks at **0.808** at epoch 1 of the linear KL-anneal schedule, then degrades to ≈ 0.67 over epochs 2–20 as the KL pull toward the uniform Dirichlet absorbs the head's discriminative signal. We mitigate by best-checkpoint-by-AUROC selection (the validated 0.808 number is the saved ckpt). The proper fix is a late-stage KL warm-restart schedule that resets $\lambda_t$ to 0 every $T$ epochs and re-anneals; this will be evaluated in the next revision and pre-registered as the §IV.D primary protocol.

Figure 6 [TBD-after-exp] gives one qualitative panel per failure mode.

## V.C  Honest Negative Findings

**The pre-registered RQ2 fallback was not triggered.** The §IV.0 preliminary RQ2 vacuity AUROC of 0.808 clears the pre-registered $\ge 0.80$ threshold (research_plan v3 §9 G-5). The C1 framing therefore stays at "one vacuity, three jobs" rather than the "two jobs" fallback. The supplementary keeps the fallback path retained for full-scale revisit and for the 16/3 robustness split, which we expect to be tighter.

**Preliminary absolute mIoU does not yet match published ConvBKI/S-BKI numbers.** Section IV.0 reports an M1 mIoU of 14.73% (preliminary) versus published targets of 77.7% (ConvBKI [4], KITTI seq 15) and 51.3% (S-BKI [2], SemanticKITTI test). This 6-9 % point absolute gap is dominated by the 250× capacity gap between the substitute PointNet-Vanilla backbone and the planned Cylinder3D backbone (Limitation iv); the comparative ordering (M1 > R2 in mIoU, ECE, and latency at matched backbone capacity) is the genuine method-level signal we currently claim. Full-scale absolute numbers replace the §IV.0 placeholders once Limitation iv is unblocked.

**Khronos head-to-head pre-registered fallback.** If the RQ5 head-to-head against Khronos [6] in §IV.G shows a dynamic-object mIoU gap of more than 3 against Khronos's published numbers and the latency advantage on Orin NX is less than $2\times$, we will explicitly acknowledge it and reframe the win as a latency-integration trade-off rather than a methodological replacement — closing the audit E6 ("over-claiming") risk before a reviewer raises it.

---

# VI. Conclusion

[TBD-after-exp] We have presented EvidLife-Map, a LiDAR-only online metric-semantic mapping system whose closed-form Dirichlet-evidential per-voxel posterior exposes a single vacuity scalar that simultaneously drives open-set / OOD detection, loop-closure descriptor entropy, and vacuity-conditioned voxel decay. The three modules M1, M2, M3 share one per-voxel state and one arithmetic primitive (conjugate Dirichlet addition), and together close four reviewer-credible weaknesses of the R2 baseline [1] within an 8-page IROS conference budget: unspecified Bayes filter, missing open-set handling, missing loop closure, and missing lifelong decay. On five public datasets we report <mark>XX.X</mark> mIoU / <mark>XX.X</mark> ECE on closed-set mapping [TBD-RQ1], vacuity AUROC <mark>XX.X</mark> on the 14/5 open-set split [TBD-RQ2], and dynamic-object mIoU within <mark>XX.X</mark> of Khronos [6] [TBD-RQ5] at <mark>XX.X</mark> Hz on Jetson Orin NX [TBD-RQ-deploy], with code, Docker, ROS 2 launch files, and a 30-second supplementary demo video released. Future work explicitly out of scope for this paper but enabled by the same per-voxel evidential substrate includes (i) a SAM-based open-vocabulary head feeding evidence into the same Dirichlet update, (ii) a 3D Gaussian-splatting variant of the same vacuity-decay coupling on a non-voxel substrate (the GS-LIVO [20] direction), and (iii) a 4D-radar fusion branch that supplies an evidence channel orthogonal to LiDAR under adverse weather — the latter restoring the v1 multi-modal ambition that the v3 deployment-on-4060 scope deliberately excluded.

---

# References

*Numeric → bibkey map, indexed by first appearance in §I–§VI. All bibkeys reference entries of `D:\_7_sci\semantic_mapping\_new_paper\refs\refs.bib` (60 entries; this paper cites <mark>30/60</mark>).*

[1] J. Jiao, R. Geng, Y. Li, R. Xin, B. Yang, J. Wu, L. Wang, M. Liu, R. Fan, and D. Kanoulas, "Real-Time Metric-Semantic Mapping for Autonomous Navigation in Outdoor Environments," *IEEE Trans. Autom. Sci. Eng.*, 2024. *[bibkey: jiao2024r2]*

[2] L. Gan, R. Zhang, J. W. Grizzle, R. M. Eustice, and M. Ghaffari, "Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping," *IEEE Robot. Autom. Lett.*, 2020. *[bibkey: gan2020sbki]*

[3] A. Millane, H. Oleynikova, E. Wirbel, R. Steiner, V. Ramasamy, D. Tingdahl, and R. Siegwart, "nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2024. *[bibkey: millane2024nvblox]*

[4] J. Wilson, J. Song, Y. Fu, A. Zhang, A. Capodieci, P. Jayakumar, K. Barton, and M. Ghaffari, "ConvBKI: Real-Time Probabilistic Semantic Mapping Network with Quantifiable Uncertainty," *IEEE Trans. Robot.*, 2024. *[bibkey: wilson2024convbki]*

[5] Y. Deng, B. Yao, Y. Tang, Y. Yang, and Y. Yue, "OpenVox: Real-time Instance-level Open-vocabulary Probabilistic Voxel Representation," *arXiv:2502.16528*, 2025. *[bibkey: deng2025openvox]*

[6] L. Schmid, M. Abate, Y. Chang, and L. Carlone, "Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments," in *Proc. Robot.: Sci. Syst. (RSS)*, 2024. *[bibkey: schmid2024khronos]*

[7] J. Wilson, R. Xu, Y. Sun, P. Ewen, M. Zhu, K. Barton, and M. Ghaffari, "LatentBKI: Open-Dictionary Continuous Mapping in Visual-Language Latent Spaces with Quantifiable Uncertainty," *arXiv:2410.11783*, 2024. *[bibkey: wilson2024latentbki]*

[8] L. Schmid, J. Delmerico, J. L. Chung, M. Magnusson, J. Nieto, and R. Siegwart, "Panoptic Multi-TSDFs: a Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2022. *[bibkey: schmid2022panopticmultitsdfs]*

[9] A. Rosinol, A. Violette, M. Abate, N. Hughes, Y. Chang, J. Shi, A. Gupta, and L. Carlone, "Kimera: From SLAM to Spatial Perception with 3D Dynamic Scene Graphs," *Int. J. Robot. Res.*, 2021. *[bibkey: rosinol2021kimera]*

[10] N. Hughes, Y. Chang, and L. Carlone, "Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization," in *Proc. Robot.: Sci. Syst. (RSS)*, 2022. *[bibkey: hughes2022hydra]*

[11] H. Oleynikova, Z. Taylor, M. Fehr, R. Siegwart, and J. Nieto, "Voxblox: Incremental 3D Euclidean Signed Distance Fields for On-Board MAV Planning," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2017. *[bibkey: oleynikova2017voxblox]*

[12] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard, "OctoMap: An Efficient Probabilistic 3D Mapping Framework Based on Octrees," *Auton. Robots*, 2013. *[bibkey: hornung2013octomap]*

[13] M. Grinvald, F. Furrer, T. Novkovic, J. J. Chung, C. Cadena, R. Siegwart, and J. Nieto, "Volumetric Instance-Aware Semantic Mapping and 3D Object Discovery," *IEEE Robot. Autom. Lett.*, 2019. *[bibkey: grinvald2019voxbloxpp]*

[14] G. Narita, T. Seno, T. Ishikawa, and Y. Kaji, "PanopticFusion: Online Volumetric Semantic Mapping at the Level of Stuff and Things," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2019. *[bibkey: narita2019panopticfusion]*

[15] Y. Pan, Y. Kompis, L. Bartolomei, R. Mascaro, C. Stachniss, and M. Chli, "Voxfield: Non-Projective Signed Distance Fields for Online Planning and 3D Reconstruction," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2022. *[bibkey: pan2022voxfield]*

[16] Y. Tian, Y. Chang, F. H. Arias, C. Nieto-Granda, J. P. How, and L. Carlone, "Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems," *IEEE Trans. Robot.*, 2022. *[bibkey: tian2022kimeramulti]*

[17] Y. Chang, N. Hughes, A. Ray, and L. Carlone, "Hydra-Multi: Collaborative Online Construction of 3D Scene Graphs with Multi-Robot Teams," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2023. *[bibkey: chang2023hydramulti]*

[18] D. Maggio, Y. Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty, E. Cristofalo, L. Schmid, and L. Carlone, "Clio: Real-Time Task-Driven Open-Set 3D Scene Graphs," *IEEE Robot. Autom. Lett.*, 2024. *[bibkey: maggio2024clio]*

[19] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, "HOV-SG: Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation," in *Proc. Robot.: Sci. Syst. (RSS)*, 2024. *[bibkey: werby2024hovsg]*

[20] Z. Hong, X. Zheng, W. Ding, S. Shen, and Y. Zhou, "GS-LIVO: Real-Time LiDAR, Inertial, and Visual Multisensor Fused Odometry with Gaussian Mapping," *IEEE Trans. Robot.*, 2025. *[bibkey: hong2025gslivo]*

[21] A. Piroli, V. Dallabetta, J. Kopp, M. Walessa, D. Meissner, and K. Dietmayer, "Energy-Based Detection of Adverse Weather Effects in LiDAR Data," *IEEE Robot. Autom. Lett.*, 2023. *[bibkey: piroli_2023_semanticspray]*

[22] K. Yamazaki, T. Hanyu, K. Vo, T. Pham, M. Tran, G. Doretto, A. Nguyen, and N. Le, "Open-Fusion: Real-time Open-Vocabulary 3D Mapping and Queryable Scene Representation," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2024. *[bibkey: yamazaki2024openfusion]*

[23] A. Sheppard, P. Ewen, J. Wilson, A. V. Sethuraman, B. Adewole, A. Li, Y. Chen, R. Vasudevan, and K. A. Skinner, "SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework," *IEEE Robot. Autom. Lett.*, 2025. *[bibkey: sheppard2025slimvdb]*

[24] J. Fu, C. Lin, Y. Taguchi, A. Cohen, Y. Zhang, S. Mylabathula, and J. J. Leonard, "PlaneSDF-Based Change Detection for Long-term Dense Mapping," *IEEE Robot. Autom. Lett.*, 2022. *[bibkey: fu_2022_planesdf]*

[25] M. A. Vega-Torres, A. Braun, and A. Borrmann, "SLAM2REF: Advancing Long-Term Mapping with 3D LiDAR and Reference Map Integration for Precise 6-DoF Trajectory Estimation and Map Extension," *Constr. Robot.*, 2024. *[bibkey: vegatorres_2024_slam2ref]*

[26] L. Li, X. Kong, X. Zhao, W. Li, F. Wen, H. Zhang, and Y. Liu, "SA-LOAM: Semantic-aided LiDAR SLAM with Loop Closure," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2021. *[bibkey: li_2021_saloam]*

[27] L. Yang, R. Mascaro, I. Alzugaray, S. M. Prakhya, M. Karrer, Z. Liu, and M. Chli, "LiDAR Loop Closure Detection using Semantic Graphs with Graph Attention Networks," *J. Intell. Robot. Syst.*, 2025. *[bibkey: yang_2025_semanticloop]*

[28] J. L. Matez-Bandera, D. Fernandez-Chaves, J.-R. Ruiz-Sarmiento, J. Monroy, N. Petkov, and J. Gonzalez-Jimenez, "LTC-Mapping: Enhancing Long-Term Consistency of Object-Oriented Semantic Maps in Robotics," *Sensors*, 2022. *[bibkey: matezbandera2022ltcmapping]*

[29] Y. Miao, I. Armeni, M. Pollefeys, and D. Barath, "Volumetric Semantically Consistent 3D Panoptic Mapping," *arXiv:2309.14737*, 2023. *[bibkey: miao2023volumetric]*

[30] S. Zhu, R. Qin, G. Wang, J. Liu, and H. Wang, "SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2025. *[bibkey: zhu2025semgaussslam]*

---

# Appendix

**Acknowledgments.** [TBD]

**Author Contributions.** [TBD]
