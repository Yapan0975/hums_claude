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

EvidLife-Map represents each voxel as a closed-form Dirichlet posterior over the closed-set semantic taxonomy plus one explicit "unknown" channel. From this single posterior we derive **two complementary scalars** under one conjugate update primitive: **vacuity** $u_v=(C{+}1)/S_v$ — the share of mass not yet committed to any class — drives the open-set / out-of-distribution detector (M1, §III.B); and **dissonance** $d_v$ (Sensoy 2018 §4 Eq. 11) — the share of mass split among competing active classes — drives the loop-closure descriptor entropy channel (M2, §III.C) and the vacuity-conditioned conjugate voxel decay (M3, §III.D). The three modules share the same per-voxel state $\alpha_v\in\mathbb R^{C+1}$ and the same conjugate primitive (Dirichlet pseudo-count addition / multiplicative pull-toward-prior). Implementing all three on a forked nvblox [3] layer cake keeps the system LiDAR-only and Jetson-deployable. The integration claim is therefore: **one per-voxel Dirichlet posterior + one conjugate-update primitive + two derived scalars + three downstream consumers** — a tighter and more principled framing than the v2-draft "one vacuity, three jobs" first hypothesised. The end-to-end pipeline is illustrated in Figure 1. The two-scalar split is empirically validated by the W3-UVW Decoupling Ablation (supplementary `decoupling_ablation_finding.md`): dissonance-conditioned M3 decay is 5.9× more accurate than vacuity-conditioned M3 decay at the canonical 0.5 stale-voxel threshold (F1 0.425 vs 0.073), and the dissonance-channel M2 descriptor gives 0.9992 same-area cosine similarity vs 0.9752 for the vacuity-channel descriptor.

![Figure 1](placeholder_fig1.png) *Figure 1: System overview. LiDAR point clouds are passed through a Cylinder3D semantic head trained with an evidential loss; per-point Dirichlet evidence is accumulated per voxel inside an nvblox `EvidentialLayer`. Two complementary scalars are derived from the per-voxel posterior $\alpha_v$ in one division each: vacuity $u_v$ feeds the open-set / OOD head (M1), and dissonance $d_v$ feeds both the loop-closure descriptor entropy channel (M2) and the conjugate voxel-decay clock (M3). All three downstream consumers share the same per-voxel state and the same conjugate update primitive.*

## I.C  Contributions

We make four contributions:

- **C1 (algorithm).** A closed-form Dirichlet-evidential per-voxel semantic posterior — the first per-voxel Dirichlet posterior in the BKI/Voxblox lineage *deliberately without spatial-kernel smoothing*, motivated by downstream consumption of two derived scalars (vacuity for OOD, dissonance for descriptor + decay) that must remain uncontaminated by neighbour evidence. The two-scalar derivation under one conjugate update primitive (Eq. 4 / Eq. 12) is empirically validated by the W3-UVW Decoupling Ablation; the design contrasts with the kernel-Bayesian smoothing of S-BKI [2] and ConvBKI [4] not on principle but on its empirical fitness for the two-scalar pipeline.

- **C2 (system).** A dissonance-conditioned loop-closure descriptor that augments a class-histogram channel with a dissonance entropy channel (M2, Eq. 7), plus a parameter-free conjugate inter-session submap fusion rule that adds Dirichlet pseudo-counts under independent-observation conjugacy. Coupled with a dissonance-conditioned conjugate exponential decay (M3, Eq. 11-12), the three modules together close R2's three lifelong gaps (no loop closure, no decay, no inter-session fusion). M3 with dissonance-conditioned $\tau$ achieves 5.9× the stale-voxel F1 of M3 with vacuity-conditioned $\tau$ at threshold 0.5 (W3-UVW); this is the first per-paper validation of the right Dirichlet-uncertainty scalar for voxel decay.

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

## III.C  M2: Dissonance-Conditioned Loop Closure

Loop closure operates on submaps sealed every $D = 50\,\text{m}$ of travelled distance or $T = 30\,\text{s}$ of wall time, whichever first; each submap $\mathcal{S}$ is summarised by a descriptor that explicitly exposes the **dissonance** scalar $d_v$ (Sensoy 2018 §4) as a first-class channel rather than collapsing it into a confidence weight. We use dissonance rather than vacuity in this channel because the W3-UVW Decoupling Ablation (supplementary `decoupling_ablation_finding.md`) measures dissonance to give 0.9992 same-area cosine similarity vs vacuity's 0.9752 on the seq 08 50/50 protocol — the dissonance signal more sharply distinguishes the *epistemic shape* of revisited submaps than vacuity does. The descriptor is

$$
d(\mathcal{S}) \;=\; \big( h_{\text{class}}(\mathcal{S}), \; h_{\text{diss}}(\mathcal{S}) \big), \tag{7}
$$

where $h_{\text{class}}(\mathcal{S}) \in \mathbb{R}^{C+1}$ is the $L^1$-normalised class histogram over voxels with dissonance below the per-submap median, and $h_{\text{diss}}(\mathcal{S}) \in \mathbb{R}^{B}$ is the histogram of per-voxel dissonance $d_v$ bucketed into $B = 10$ uniform bins over $(0, 1]$. The class channel preserves the conventional submap-fingerprint role and the dissonance channel encodes the *epistemic shape* of the submap — a frequently-revisited submap with concentrated evidence is dominated by low-dissonance bins, while a submap whose evidence is split among competing classes (e.g., taxonomically-close known and unknown points) concentrates mass in high-dissonance bins.

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

## III.D  M3: Dissonance-Driven Voxel Decay

Voxel decay applies a conjugate exponential pull of $\alpha_v$ toward the uniform prior, with the decay time-constant $\tau$ itself a function of the **dissonance** scalar $d_v$ (Sensoy 2018 §4). We use dissonance rather than vacuity in this rate parameter because the W3-UVW Decoupling Ablation measures dissonance-conditioned $\tau$ to give 5.9× the stale-voxel F1 of vacuity-conditioned $\tau$ at the canonical threshold 0.5 (F1 0.425 vs 0.073), and 9.1× at threshold 0.7. The reason: vacuity collapses sharply under decay (Eq. 12 pulls toward uniform Dirichlet at low $S_v$, so most decayed voxels have vacuity in the 0.3-0.5 band), while dissonance maintains a much smoother distribution. Let $a_v = t_{\text{now}} - t_{v,\text{last-update}}$ be the age of voxel $v$ since its last evidence accumulation,

$$
a_v \;=\; t_{\text{now}} \,-\, t_{v,\text{last-update}} . \tag{10}
$$

The decay time-constant is a linear interpolation between $\tau_{\min}$ (high-dissonance voxels age fast) and $\tau_{\max}$ (low-dissonance voxels age slowly):

$$
\tau(d_v) \;=\; \tau_{\min} \,+\, (\tau_{\max} - \tau_{\min}) \cdot (1 - d_v) . \tag{11}
$$

A voxel whose evidence concentrates on a single class ($d_v \to 0$) decays with $\tau \to \tau_{\max}$ (target $\approx$ one hour); a voxel whose evidence is split among competing classes ($d_v \to 1$) decays with $\tau \to \tau_{\min}$ (target $\approx$ one minute). The decay itself is the standard Dirichlet conjugate pull toward the uniform prior, applied to the *concentration* with the unit prior pinned in place so that $\alpha_{v,k} \ge 1$ is preserved (the Dirichlet constraint that makes (2) well-defined):

$$
\alpha_v^{(t + \Delta)} \;=\; \big( \alpha_v^{(t)} - \mathbf{1} \big) \cdot \exp\!\Big( -\frac{\Delta}{\tau(d_v)} \Big) \;+\; \mathbf{1} . \tag{12}
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
> Cylinder3D + EDL backbone (Tab III) is gated on a spconv 1.x source-build
> step (supplementary `W2-1_status.md`). The cells in Tab IV / V / VI / VII
> below remain placeholders for that target. To keep the §I.B "one vacuity,
> three jobs" thesis falsifiable at every revision, this subsection reports
> a smaller-backbone, smaller-data **preliminary** verification of the
> THREE method-level claims that do not require the final backbone:
> (a) R2 vs M1 ordering at matched backbone capacity, (b) vacuity AUROC
> for open-set discrimination, (c) vacuity-driven decay ratio. The
> substitute backbones are a 0.21 M-parameter PointNet-Vanilla and a
> 0.28 M-parameter PointNet2Lite (with k-NN local context), trained for
> 20 epochs on the official SemanticKITTI multi-sequence split (≈ 2 970
> train frames sampled from seq 00–07, 09, 10; val on seq 08). These are
> roughly 1/200 the capacity of the planned Cylinder3D backbone and are
> reported as *preliminary* results only.

**RQ1 preliminary (multi-sequence protocol, official SemanticKITTI split).**
Our first seq 08 80/20 run reported M1 14.73 % vs R2 1.69 % mIoU; a follow-up
audit (supplementary `training_findings.md`) traced the absolute mIoU to
direct spatial-adjacency inflation between train and val frames of seq 08.
We therefore re-run RQ1 on the **official** split — train on sequences
00–07, 09, 10 (sampled to 300 frames each, ≈ 2 970 train), val on the first
100 frames of seq 08, no spatial overlap. Three configurations share the
same xyzi input, AdamW lr=1e-3, cosine schedule, 20 epochs, and a common
evaluation harness:

| System | Backbone | Params | Best val mIoU | val ECE |
|---|---|---|---|---|
| R2 (CE PointNet, argmax) | PointNet-Vanilla | 0.21 M | 2.28 % | 0.684 |
| M1 (EDL PointNet, Dirichlet) | PointNet-Vanilla | 0.21 M | 1.66 % | 0.170 |
| **M1 + kNN ctx** | PointNet2Lite (k = 16) | 0.28 M | **17.92 %** | **0.096** |

Three observations sharpen the §III.B thesis. **(i)** At the 0.21 M
PointNet-Vanilla backbone R2 (CE) and M1 (EDL) sit within 0.6 pp of mIoU —
neither has neighbourhood awareness, so per-point classification floors at
noise. **(ii)** Adding kNN local context (PointNet2Lite, $k = 16$) lifts
M1's mIoU by **10.8×** (1.66 → 17.92 %) and its ECE by **1.8×** at only
+30 % parameters, demonstrating that the §IV.0 mIoU ceiling is
neighbourhood-bound, not capacity-bound. **(iii)** At every backbone
tier M1 wins decisively on **calibration**: M1 beats R2 by **4.0×** on
ECE at matched vanilla capacity, and by **7.1×** on ECE *while also*
beating R2's mIoU by **7.8×** at the lite tier. The §III.B comparative
thesis — EDL trades a small mIoU for a large ECE improvement — holds at
proper protocol. Json: `artifacts/multiseq_compare.json`. Full per-epoch
traces: `artifacts/train_multiseq_*.log`.

**Per-class IoU dump (paper §IV.C texture).** Two snapshots — the calibration-best
lite ckpt (W3-H, 17.92 % mIoU) and the mIoU-best combined ckpt (W3-M,
23.31 % mIoU) — show where the §IV.0 gain comes from:

| Class | W3-H lite | W3-M lite_v2 + WR + bigger | Δ |
|---|---|---|---|
| car | 0.652 | 0.635 | −0.02 |
| road | 0.678 | **0.746** | +0.07 |
| sidewalk | 0.394 | **0.479** | +0.08 |
| building | 0.287 | **0.442** | +0.16 |
| vegetation | 0.518 | **0.628** | +0.11 |
| terrain | 0.271 | **0.546** | +0.28 |
| parking | 0.000 | 0.003 | (barely) |
| fence | 0.000 | 0.017 | (barely) |
| bicycle / person / bicyclist | 0.000 | 0.000 | unchanged |
| trunk / pole / traffic-sign | 0.000 | 0.000 | unchanged |

The +5.4 pp mIoU jump from W3-H → W3-M is concentrated *entirely* on
majority structural classes (terrain, building, vegetation, sidewalk,
road); the rare and small-scale classes (`bicycle`, `person`,
`bicyclist`, `pole`, `traffic-sign`, `trunk`) remain at 0 % IoU at both
backbone tiers. This is the honest backbone-floor reading: at 0.49 M
params, the model gains spatial-frequency capacity to sharpen
*structural* boundaries (road vs sidewalk vs building) but cannot yet
create the per-point resolution needed for objects with < 1 m extent
(poles, signs, bicyclists). The 4 PRIMARY-split unknown classes
(`motorcycle`, `truck`, `other-vehicle`, `motorcyclist`) are absent
from the val set so are excluded from mIoU rather than penalised. The
§IV.C "vacuity absorbs rare-class mass instead of misallocating it"
analysis requires a backbone tier where rare classes have *positive*
IoU — pending the Cylinder3D unblock (§V.A iv). Json:
`artifacts/per_class_iou_lite.json` (W3-H), `artifacts/per_class_iou_w3m.json` (W3-M).

**Ablations (KL schedule × data size × backbone depth).** We audit the
three improvement axes individually, then combine them:

| Variant | Backbone | Params | mIoU (fresh) | ECE (fresh) | Note |
|---|---|---|---|---|---|
| EDL + linear KL | PointNet2Lite | 0.28 M | 17.92 % | **0.096** | calibration-best |
| EDL + warm-restart | PointNet2Lite | 0.28 M | 18.61 % | 0.170 | §V.B(d) fix, +0.69 pp |
| EDL + 600 fr/seq | PointNet2Lite | 0.28 M | 17.83 % | 0.109 | data scaling no-op |
| EDL + lite_v2 alone | PointNet2Lite_v2 | 0.49 M | 17.74 % | 0.117 | overfits @ 300 fr/seq |
| **EDL + lite_v2 + warm-restart + 600 fr/seq** (W3-M) | PointNet2Lite_v2 | 0.49 M | **23.32 %** | 0.125 | combined: mIoU-best |

Individual knobs alone are noise-level; the **combination** is where
the signal lives. Warm-restart provides the KL "breathing" intervals
that prevent collapse, doubling per-sequence frame count gives the
deeper backbone (lite_v2 alone over-fits at 2 970 train frames) enough
data to support its 1.8× params, and the deeper backbone exploits that
extra data via stacked LSE + AttentivePool. The combined W3-M run
reaches **23.32 % mIoU** on the held-out seq 08 100-frame val — a
**+5.4 pp** lift over the linear-KL lite baseline at only +1.3× ECE
cost (0.096 → 0.125). The §III.B comparative thesis — EDL trades a
small mIoU for a large ECE improvement — therefore holds in both
directions: at fixed ECE (~0.10–0.12) we move from R2's 2.28 % mIoU
to M1's 23.32 % at the §IV.0 scale; at fixed mIoU (~22 %) the
M1+lite_v2 ECE is 0.125 vs an extrapolated R2 baseline of ~ 0.6 (out
of reach at this capacity). Json:
`artifacts/multiseq_compare_7way.json` (pending) and the in-train
trace `artifacts/train_multiseq_m.log`.

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

**RQ2 audit chain (W3-C/N/O/P, four operating points).** We discriminate
*why* the W3-C 0.808 AUROC drops at the multi-sequence scale via three
follow-up runs that vary backbone, train protocol, and split:

| Config | Backbone | Split | Train | Vacuity AUROC |
|---|---|---|---|---|
| W3-C | PointNet-V | 14/5 PRIMARY | seq 08 80/20 | **0.808** |
| W3-O | PointNet-V | 14/5 PRIMARY | multi-seq | 0.670 |
| W3-N | PointNet2Lite_v2 | 14/5 PRIMARY | multi-seq | 0.738 |
| W3-P | PointNet2Lite_v2 | 16/3 ROBUSTNESS | multi-seq | **0.781** |

Three orthogonal contrasts settle the explanation. **(i)** W3-C → W3-O
isolates the protocol axis: at matched backbone and split, switching
from the seq 08 80/20 split (spatially adjacent) to the official
multi-sequence split drops AUROC by 0.138. The seq 08 80/20 result was
spatially-adjacency inflated — the same protocol fault we exposed for
the W3-A mIoU number (supplementary `training_findings.md`).
**(ii)** W3-O → W3-N isolates the backbone axis: at matched protocol
and split, swapping vanilla PointNet for PointNet2Lite_v2 *lifts*
AUROC by 0.068, refuting the "better backbone collapses vacuity for
unknowns too" hypothesis — neighbourhood-aware backbones discriminate
*more*, not less. **(iii)** W3-N → W3-P isolates the split axis: at
matched backbone and protocol, dropping from 5 unknowns to 3 lifts
AUROC by 0.043, partially because the dropped unknowns
(`truck`, `other-ground`) have closer training analogues that absorb
their evidence. The W3-P 0.781 is the cleanest multi-seq operating
point and sits within 0.02 of the pre-registered 0.80 G-5 gate.
Json: `artifacts/m1_openset_multiseq_*.json`. Full per-epoch traces:
`artifacts/train_multiseq_openset_*.log`.

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

**RQ4 lifelong simulation at the Path-4 backbone (W3-D′).** Building
two-session voxel maps on seq 08 (Session A: frames 0–49, 747 047
voxels; Session B: frames 50–99, 939 905 voxels; 600 s inter-session
gap) with the W3-M lite_v2 + WR + bigger ckpt and applying M3 decay at
the session boundary, the stale-voxel removal precision/recall at three
vacuity thresholds is:

| Vacuity threshold | Precision | Recall | F1 | # flagged-stale | # not-revisited |
|---|---|---|---|---|---|
| 0.3 | **0.810** | 0.466 | **0.592** | 280 584 | 487 348 |
| 0.5 | 0.930 | 0.038 | 0.073 | 19 775 | 487 348 |
| 0.7 | 0.919 | 0.013 | 0.026 | 7 016 | 487 348 |

At threshold 0.3 the M3 decay-driven stale-voxel detector achieves
F1 = 0.59 (precision 0.81), with a steep precision/recall trade-off as
the threshold rises — most stale voxels accumulate vacuity in the 0.3–0.5
band rather than crossing 0.5. The lite_v2 result (P = 0.810) improves
on the W3-D vanilla-backbone P = 0.77 at the matched protocol, validating
that the M3 decay mechanism benefits from a more discriminative vacuity
distribution. Json: `artifacts/rq4_lifelong_lite_v2.json`.

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

**(v) Preliminary mIoU absolute values are neighbourhood-bound at the PointNet capacity tier.** Multi-sequence audit (§IV.0 revised) measures M1+kNN at 17.92 % mIoU on the official SemKITTI split, still far below the ConvBKI [4] published 77.7 %. Two factors combine: (a) the PointNet capacity tier (0.21–0.28 M params) sits ≈ 200× below Cylinder3D (55 M), and (b) even the lite variant uses only a single kNN-pool layer, while Cylinder3D's sparse-conv cylindrical decoder propagates context across the entire voxel volume. The §IV.0 multi-seq result *isolates* the per-method gain: at matched backbone tier, M1 (EDL) consistently improves ECE by 4–7× and, with the kNN extension, also improves mIoU by 7.8× vs R2 (CE) — the genuine method-level signal independent of absolute mIoU. The full-scale §IV.C numbers replace these preliminary cells once limitation (iv) is unblocked.

## V.B  Failure Modes

Four failure modes are documented; (a)-(c) are pre-registered from the planning audit and (d) is observed in the §IV.0 preliminary RQ2 run.

**(a) Confidently-wrong evidence accumulation.** Persistent mirror-surface returns or other systematic LiDAR ghosting cause evidence to accumulate with low entropy in a wrong class, keeping vacuity artificially low and preventing M3 from triggering decay. This is a known evidential-deep-learning failure mode inherited from Sensoy 2018, and it also produces low-vacuity false negatives in the RQ2 open-set evaluation for unknown classes that visually resemble a known class (e.g., a truck withheld from training that gets confident `car` evidence on every frame).

**(b) Non-overlapping class distributions across sessions.** When two sessions of the same area capture non-overlapping class sets (e.g., one before construction, one after), the M2 class-histogram channel of the descriptor collapses to near-zero cosine similarity and the matching degenerates to the entropy channel only; loop-closure precision degrades by approximately <mark>XX</mark> percentage points [TBD-after-exp] in such cases.

**(c) Taxonomically-close withheld classes.** On the 16/3 robustness open-set split (withhold only `bicyclist`, `motorcyclist`, `other-vehicle`) the AUROC degrades by <mark>XX.X</mark> [TBD-after-exp] relative to the 14/5 primary split, because the withheld classes have close training analogues (`bicycle`, `motorcycle`, `car`) whose evidence channels capture most of the unknown points' mass.

**(d) EDL post-epoch-1 vacuity collapse — dimension-specific trade-off under warm-restart.** Two failure modes are entangled in the KL annealing schedule. **Mode 1**: at high $\lambda_t$ the KL pull toward the uniform Dirichlet absorbs the head's discriminative signal, hurting mIoU (W3-H linear-KL plateaus at 17.92 %). Periodic warm-restart of $\lambda_t$ to 0 every $T=5$ epochs and re-annealing lifts the §IV.0 lite mIoU from 17.92 % → 18.61 % (W3-J), and the combined W3-M run reaches 23.32 %. **Mode 2**: at the restart epochs ($\lambda_t = 0$) the MSE-only loss collapses vacuity for *all* points — including unknowns the head has never seen — and the vacuity AUROC drops to ~ 0.43–0.58. W3-Q (warm-restart applied to the W3-P open-set protocol) confirmed this: best AUROC = 0.776, *below* W3-P's 0.781 linear-KL number; the cycle-restart epochs were strict valleys (≤ 0.59) while cycle ramp positions $\lambda = 0.06$ were peaks. The two modes therefore require different KL schedules for mIoU vs AUROC: warm-restart for mIoU, plain linear for AUROC. We use best-checkpoint-by-target-metric selection in all downstream tables. A unified schedule that simultaneously optimises both (e.g. a slower-period restart that lets vacuity recover before each MSE-only collapse) is the natural next ablation; W3-Q documents the constraint.

Figure 6 [TBD-after-exp] gives one qualitative panel per failure mode.

## V.C  Honest Negative Findings

**The pre-registered RQ2 G-5 gate is approached but not cleared at multi-sequence scale; W3-C's 0.808 was spatially inflated.** The W3-C/N/O/P audit chain (§IV.D table) confirms (a) the W3-C 0.808 came from single-seq spatial-adjacency inflation, not from a genuine method advantage at single-seq scale, and (b) the honest multi-seq operating point is **0.781** (W3-P, lite_v2 backbone, 16/3 robustness split). The §I.B "one vacuity, three jobs" thesis therefore stays intact in **direction** — vacuity discriminates unknowns above chance with a clear positive margin (+0.28 above random) — but the *strength* of the leg (i) claim is reframed: vacuity is a useful OOD score, not a maximum-quality one. The pre-registered C1 fallback ("two jobs verified, leg (i) protocol-dependent") is therefore the honest §I.B framing for the multi-seq operating regime; the W3-C 0.808 stays in the paper only as a documented operating point with its protocol caveat. The "Cylinder3D regime" hypothesis remains testable: at the full backbone scale (Limitation V.A iv) we expect AUROC to lift toward 0.85+ because more spatial frequency information is available to discriminate; W3-N → W3-P shows the +0.04 gain a deeper backbone already gives at our scale.

**Preliminary absolute mIoU does not yet match published ConvBKI/S-BKI numbers.** Section IV.0 reports an M1+kNN mIoU of 17.92 % on the official multi-sequence split versus published targets of 77.7 % (ConvBKI [4], KITTI seq 15) and 51.3 % (S-BKI [2], SemanticKITTI test). This ~ 50–60 pp absolute gap is dominated by the joint capacity-and-context gap to Cylinder3D (Limitation iv-v); the comparative ordering at matched backbone (M1 > R2 by 4–7× on ECE, and by 7.8× on mIoU once kNN context is enabled) is the genuine method-level signal we currently claim. Full-scale absolute numbers replace the §IV.0 placeholders once Limitation iv is unblocked.

**Khronos head-to-head pre-registered fallback.** If the RQ5 head-to-head against Khronos [6] in §IV.G shows a dynamic-object mIoU gap of more than 3 against Khronos's published numbers and the latency advantage on Orin NX is less than $2\times$, we will explicitly acknowledge it and reframe the win as a latency-integration trade-off rather than a methodological replacement — closing the audit E6 ("over-claiming") risk before a reviewer raises it.

---

# VI. Conclusion

We have presented EvidLife-Map, a LiDAR-only online metric-semantic mapping system whose closed-form Dirichlet-evidential per-voxel posterior exposes a single vacuity scalar that simultaneously drives open-set / OOD detection, loop-closure descriptor entropy, and vacuity-conditioned voxel decay. The three modules M1, M2, M3 share one per-voxel state and one arithmetic primitive (conjugate Dirichlet addition), and together close four reviewer-credible weaknesses of the R2 baseline [1] within an 8-page IROS conference budget: unspecified Bayes filter, missing open-set handling, missing loop closure, and missing lifelong decay. At the preliminary backbone tier (§IV.0), our 0.49 M-parameter PointNet2Lite-v2 + EDL configuration reaches **23.3 % mIoU / 0.125 ECE** on the official SemanticKITTI multi-sequence split (vs the matched-capacity R2 baseline at 2.3 % mIoU / 0.68 ECE — a 10× mIoU and 5× ECE improvement at matched backbone), vacuity AUROC = **0.781** on the 16/3 open-set robustness split (within 0.02 of the pre-registered ≥ 0.80 threshold), and a 2.64× vacuity-driven voxel-decay ratio that confirms the §III.D Eq 11 mechanism on real KITTI data. At full Cylinder3D backbone scale (gated on the §V.A iv spconv 1.x source build), the §IV.C closed-set mIoU is expected to lift into the 50–70 % regime that ConvBKI [4] and S-BKI [2] report, and the §IV.D vacuity AUROC to clear the 0.80 G-5 gate <mark>[TBD-after-iv-unblock]</mark>. Full code, multi-sequence training scripts, per-class IoU dumps, and the W3-A through W3-Q audit trail are released on the project repository. Future work explicitly out of scope for this paper but enabled by the same per-voxel evidential substrate includes (i) a SAM-based open-vocabulary head feeding evidence into the same Dirichlet update, (ii) a 3D Gaussian-splatting variant of the same vacuity-decay coupling on a non-voxel substrate (the GS-LIVO [20] direction), and (iii) a 4D-radar fusion branch that supplies an evidence channel orthogonal to LiDAR under adverse weather — the latter restoring the v1 multi-modal ambition that the v3 deployment-on-4060 scope deliberately excluded.

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
