---
title: "EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay"
authors: "[TBD: lead + co-authors]"
target_venue: "IROS 2027 (8 pages, double-column IEEE conference format)"
date: 2026-05-28
version: v1
status: "skeleton — §I/§II/§III only; §IV–§VI to be back-filled after experiments"
notes: |
  - Citation tokens use [bibkey] form; Stage-5 format-convert will translate to \cite{bibkey}.
  - Numerical placeholders use "XX.X" pending §IV measurements.
  - All citations are drawn from D:\_7_sci\semantic_mapping\_new_paper\refs\refs.bib (60 entries).
  - Companion: research_plan.md (v3) and paper_outline_v2.md (v3 contents).
---

# §I  Introduction

## I.A  Motivation

Outdoor robot autonomy across repeated traversal of degraded or partially unmapped environments demands a metric-semantic voxel map that simultaneously (i) quantifies what it knows about each voxel, (ii) flags voxels whose observed evidence does not match any trained category, and (iii) ages stale evidence as the world changes between visits. Current online metric-semantic mapping (MSM) systems address these requirements only piecewise. The recent R2 system [jiao2024r2] couples LiDAR–visual–inertial odometry with an nvblox [millane2024nvblox] truncated signed-distance backbone and a per-voxel iterative Bayes filter, but leaves the filter under-specified, makes no provision for unknown categories, and provides no mechanism for ageing or revisiting stale voxels — three properties that audit trails of the system have identified as the most reviewer-credible weaknesses. Probabilistic-voxel competitors close one gap each: S-BKI [gan2020sbki] and ConvBKI [wilson2024convbki] give kernel-Bayesian calibration; Khronos [schmid2024khronos] gives a learned short-/long-term factorisation for dynamics; Panoptic Multi-TSDFs [schmid2022panopticmultitsdfs] gives submap-level long-term consistency. No single online system delivers all three jobs from one statistical quantity, leaving uncertainty quantification, lifelong maintenance, and open-set awareness as three independently parameterised heuristics.

## I.B  Approach Overview

EvidLife-Map represents each voxel as a closed-form Dirichlet posterior over the closed-set semantic taxonomy plus one explicit "unknown" channel; the posterior's vacuity mass — a single scalar derived from the Dirichlet concentration — is reused as (i) the open-set / out-of-distribution score, (ii) the entropy channel of the loop-closure descriptor, and (iii) the trigger for a vacuity-conditioned conjugate decay that ages stale voxels. The system is built as three modules sharing the same per-voxel state: **M1**, Dirichlet evidential per-voxel semantic fusion (§III.B); **M2**, vacuity-conditioned loop closure with parameter-free conjugate inter-session submap fusion (§III.C); and **M3**, vacuity-driven voxel decay (§III.D). Implementing all three on a forked nvblox [millane2024nvblox] layer cake allows EvidLife-Map to remain LiDAR-only and Jetson-deployable while replacing R2's three independent heuristics with a single Dirichlet-derived signal. The end-to-end pipeline is illustrated in Figure 1.

![Figure 1](placeholder_fig1.png) *Figure 1: System overview. LiDAR point clouds are passed through a Cylinder3D semantic head trained with an evidential loss; per-point Dirichlet evidence is accumulated per voxel inside an nvblox `EvidentialLayer`. The resulting per-voxel vacuity mass feeds three downstream consumers (open-set head, loop-closure descriptor, voxel-decay trigger) — the load-bearing "one vacuity, three jobs" wiring of this paper.*

## I.C  Contributions

We make four contributions:

- **C1 (algorithm).** A closed-form Dirichlet-evidential per-voxel semantic posterior whose vacuity mass is, by construction, simultaneously usable as (i) an open-set / OOD score, (ii) a loop-closure descriptor entropy channel, and (iii) a lifelong decay trigger. The posterior generalises the unspecified Bayes filter of R2 [jiao2024r2] and contrasts with the kernel-Bayesian smoothing of S-BKI [gan2020sbki] and ConvBKI [wilson2024convbki] by giving an honest per-voxel epistemic signal uncontaminated by neighbour evidence.

- **C2 (system).** A vacuity-conditioned loop-closure descriptor that augments a class-histogram channel with an entropy channel, plus a parameter-free conjugate inter-session submap fusion rule that adds Dirichlet pseudo-counts under the standard conjugacy of independent observations. Coupled with a vacuity-driven conjugate exponential decay, the three modules together close R2's three lifelong gaps (no loop closure, no decay, no inter-session fusion) without introducing any heuristic threshold that is not derived from a Dirichlet quantity already computed for M1.

- **C3 (empirical evidence).** The first systematic evaluation of a single LiDAR-only evidential MSM system on five public datasets — SemanticKITTI (closed-set RQ1 and a 14-known/5-unknown open-set split RQ2), nuScenes-LiDARSeg (cross-domain RQ1 and reverse cross-domain RQ2), KITTI-360 (multi-session lifelong RQ4), SemanticSpray [piroli_2023_semanticspray] (uncertainty-aware traversability RQ3), and a SemanticKITTI dynamic-frame subset (head-to-head against Khronos [schmid2024khronos] RQ5). RQ5 in particular gives the first published head-to-head between an implicit-vacuity dynamic-handling mechanism and Khronos's explicit short-/long-term factoriser on a public dynamic split.

- **C4 (deployment).** A reproducibility package — source code, Docker image, ROS 2 launch files, open-set split definitions, the KITTI-360 revisit pair builder, a Jetson Orin NX runtime benchmark, and a 30-second supplementary demo video of EvidLife-Map running live on a Jetson Orin NX over a SemanticKITTI replay — that ships with the paper.

---

# §II  Related Work

## II.A  Online Metric-Semantic Mapping

Online metric-semantic mapping descends from the Euclidean signed-distance lineage of Voxblox [oleynikova2017voxblox] and the earlier OctoMap [hornung2013octomap] occupancy framework, both of which provide incremental volumetric backbones for on-board planning but leave semantics outside the probabilistic state. Voxblox++ [grinvald2019voxbloxpp] and PanopticFusion [narita2019panopticfusion] extend the line to volumetric instance-aware and panoptic semantic mapping, respectively, by attaching per-voxel label counts to the SDF; Voxfield [pan2022voxfield] subsequently relaxes the projective approximation in the SDF update so that the distance field remains non-projective even under oblique LiDAR rays. The Kimera family — Kimera [rosinol2021kimera] and its distributed extension Kimera-Multi [tian2022kimeramulti] — introduces dense metric-semantic SLAM with 3D dynamic scene graphs, while Hydra [hughes2022hydra] and Hydra-Multi [chang2023hydramulti] organise the metric-semantic substrate hierarchically for real-time spatial perception and collaborative multi-robot construction. Recent systems push the frontier in three directions. R2 [jiao2024r2] integrates LVIO with an nvblox [millane2024nvblox] backbone and a confidence-aware HRNet semantic head for outdoor navigation, but uses an under-specified per-voxel Bayes filter and treats unlabeled voxels as untraversable rather than as unknown. Khronos [schmid2024khronos] adds a learned spatio-temporal factorisation that explicitly separates short-term motion from long-term change. Clio [maggio2024clio] compresses the scene graph under a task prior to enable open-set retrieval, and HOV-SG [werby2024hovsg] extends the open-vocabulary scene-graph idea to a hierarchical floor/room/object layout for language-grounded navigation. The Gaussian-splatting strand — GS-LIVO [hong2025gslivo] is its LiDAR-inertial-visual exemplar, with SemGauss-SLAM [zhu2025semgaussslam] and OpenGS-SLAM [yang2025opengsslam] as the indoor counterparts — replaces the volumetric backbone entirely with a continuous Gaussian field deployable on Jetson-class hardware. A related panoptic-3D line [miao2023volumetric] uses Voxblox-style backbones with consistency optimisation. Across this entire lineage, per-voxel uncertainty handling remains the weakest axis: the Voxblox-class systems use a point estimate of class probability, the Kimera and Hydra families use the scene graph rather than the voxel as the credible-set entity, and the more recent systems use ad-hoc heuristics — confidence-head clipping in R2, a learned change-detection network in Khronos, task-prior compression in Clio — with no single statistical quantity that quantifies what each voxel does not know.

## II.B  Probabilistic Voxel Semantic Fusion

A parallel line of work treats per-voxel semantic fusion as a Bayesian estimation problem in its own right. S-BKI [gan2020sbki] introduces Bayesian spatial kernel smoothing for scalable dense semantic mapping, replacing the independent-voxel assumption with a continuous kernel that propagates evidence between spatially adjacent voxels. ConvBKI [wilson2024convbki] reformulates the same idea as a real-time network with quantifiable uncertainty by implementing the spatial kernel as a convolution; LatentBKI [wilson2024latentbki] extends the line to open-dictionary continuous mapping in a visual-language latent space while preserving the quantifiable-uncertainty property. OpenVox [deng2025openvox] complements the BKI line with an instance-level open-vocabulary probabilistic voxel representation that maintains a per-instance Bernoulli posterior. Open-Fusion [yamazaki2024openfusion] and SLIM-VDB [sheppard2025slimvdb] integrate similar probabilistic update rules into open-vocabulary TSDF and OpenVDB backbones, respectively. These works converge on the use of a per-voxel posterior, but the posterior is consumed for one purpose only — calibration of the closed-set semantic prediction in S-BKI [gan2020sbki], ConvBKI [wilson2024convbki], and SLIM-VDB [sheppard2025slimvdb]; open-vocabulary retrieval in LatentBKI [wilson2024latentbki] and OpenVox [deng2025openvox]; or queryable feature lookup in Open-Fusion [yamazaki2024openfusion]. None of them propagates the posterior's epistemic uncertainty into either loop closure or lifelong decay. Our M1 differs from the BKI lineage by adopting a closed-form Dirichlet posterior that derives a single vacuity scalar per voxel; in particular, we deliberately omit the spatial kernel of S-BKI / ConvBKI, because the vacuity signal that M2 and M3 consume must be an honest per-voxel epistemic quantity rather than one contaminated by neighbour evidence.

## II.C  Lifelong and Long-term Map Maintenance

Lifelong and long-term metric-semantic map maintenance has been pursued primarily through explicit change-detection or submap-fusion machinery decoupled from the voxel-level uncertainty model. Panoptic Multi-TSDFs [schmid2022panopticmultitsdfs] introduces a flexible submap representation with object-level long-term dynamic-scene consistency, controlled by per-submap activity status rather than per-voxel epistemic signals. Hydra-Multi [chang2023hydramulti] supports collaborative online construction of 3D scene graphs by multi-robot teams; Kimera-Multi [tian2022kimeramulti] provides robust distributed dense metric-semantic SLAM. PlaneSDF [fu_2022_planesdf] specialises in cross-session change detection by maintaining a plane-SDF residual; SLAM2REF [vegatorres_2024_slam2ref] integrates a stored reference map to refine long-term trajectories. SA-LOAM [li_2021_saloam] and the more recent LiDAR loop-closure detection of [yang_2025_semanticloop] use semantic graphs and graph-attention networks to improve the discriminability of loop-closure descriptors. LTC-Mapping [matezbandera2022ltcmapping] enhances long-term consistency of object-oriented semantic maps for general mobile robotics. Across these works the shared pattern is to introduce a separate mechanism — explicit time windows, change-detection networks, semantic-aware loop descriptors with hand-crafted weighting, or object-level activity flags — that operates on top of, but is statistically disjoint from, the per-voxel semantic posterior. Our M2 and M3 differ by deriving both the loop-closure descriptor's entropy channel and the per-voxel decay time-constant from the same Dirichlet vacuity already computed for M1.

Across §II.A, §II.B, and §II.C, the consistent omission is therefore not the *presence* of uncertainty handling but the *integration* of one uncertainty quantity across calibration, open-set detection, loop closure, and decay. Our work differs by unifying these three ad-hoc heuristics under one Dirichlet vacuity scalar.

---

# §III  Method

## III.A  System Overview

EvidLife-Map maintains, per voxel `v`, a Dirichlet posterior over `C+1` classes — `C` closed-set semantic classes plus one explicit "unknown" channel — represented by a concentration vector `α_v ∈ R^{C+1}` with `α_{v,k} ≥ 1`. A single scalar derived from `α_v`, the **vacuity** `u_v`, is the load-bearing quantity of the paper: it is computed once per voxel per fusion step and consumed three times — by the open-set head, by the loop-closure descriptor, and by the voxel-decay trigger. Figure 1 summarises the end-to-end pipeline: a LiDAR scan is passed through a semantic head trained with an evidential loss, whose softplus output is interpreted as Dirichlet evidence; per-point evidence is splatted into voxels using the geometric and pose conventions of the underlying nvblox [millane2024nvblox] layer cake; the resulting `α_v` is updated by closed-form conjugate accumulation (Eq. 4); vacuity is read off (Eq. 2); and the three consumers operate on the same `(α_v, u_v)` state without recomputing it. The three modules are: M1 (§III.B) — Dirichlet evidential per-voxel semantic fusion that replaces R2's unspecified Bayes filter [jiao2024r2] and contrasts with S-BKI [gan2020sbki] / ConvBKI [wilson2024convbki] kernel smoothing; M2 (§III.C) — vacuity-conditioned loop closure with parameter-free conjugate Dirichlet inter-session submap fusion, derived from the same `α_v`; and M3 (§III.D) — voxel decay whose time-constant `τ(u_v)` is itself a function of vacuity, so that confidently-known voxels age slowly and high-vacuity voxels age fast. The cross-module coupling is summarised in §III.E and the nvblox-side implementation in §III.F.

## III.B  M1: Dirichlet Evidential Per-Voxel Semantic Fusion

We treat each voxel's class identity as a categorical random variable and its posterior as a Dirichlet distribution. For an observation `z` at point `p` whose semantic head outputs a non-negative evidence vector `e(z) ∈ R^{C+1}` with `e_k ≥ 0`, the evidential learning convention (`α = e + 1`) yields a Dirichlet posterior with concentration

$$
\alpha_k \;=\; e_k + 1, \qquad k = 1, \ldots, C+1 . \tag{1}
$$

Writing `S = Σ_{k=1}^{C+1} α_k` for the Dirichlet strength, the *vacuity* — the load-bearing scalar of this paper — is the share of the posterior mass that is not yet committed to any class:

$$
u_v \;=\; \frac{C+1}{S_v} \;\in\; (0, 1] . \tag{2}
$$

The expected class probability is the standard Dirichlet mean

$$
\mathbb{E}[p_k \mid \alpha_v] \;=\; \frac{\alpha_{v,k}}{S_v} . \tag{3}
$$

Across `N` independent observations splatted into voxel `v`, evidence accumulates by simple addition under Dirichlet–multinomial conjugacy:

$$
\alpha_v^{(t+1)} \;=\; \alpha_v^{(t)} \;+\; e_v^{(t+1)} . \tag{4}
$$

This is the closed-form analogue of the iterative per-voxel Bayes filter that R2 [jiao2024r2] uses but does not specify; it recovers the standard categorical Bayesian update as a degenerate limit when the evidence is one-hot and the prior is uniform, and it gives an *honest per-voxel* posterior unlike the spatial-kernel-coupled posteriors of S-BKI [gan2020sbki] and ConvBKI [wilson2024convbki] — a property M2 and M3 will rely on.

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

where `\hat{p}_k = α_k / S`, `\tilde\alpha = y + (1 - y) \odot α` masks out the ground-truth class from the regulariser, `λ_t` is an annealing weight that grows over training epochs, and `Dir(\mathbf{1})` is the uniform Dirichlet prior over `C+1` classes. The per-voxel calibration target reported in §IV is the standard expected calibration error over the `argmax` predictions

$$
\mathrm{ECE}
\;=\;
\sum_{m=1}^{M}
\frac{|B_m|}{N_{\text{voxels}}}
\left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right| , \tag{6}
$$

with `B_m` the `m`-th confidence bin over the per-voxel maximum expected probability `max_k E[p_k | α_v]`.

*Implementation.* Voxel size is `0.25 m` for closed-set runs and `0.25 m` for open-set runs (matched to R2 [jiao2024r2] and to the nvblox [millane2024nvblox] default), giving an `8^3` block layout of nominal storage size `(C+1) × 4 B × 512 ≈ 44 kB` per block at `C = 19`. The prior strength is `s_0 = C + 1` so that the uninformed posterior is exactly the uniform Dirichlet `Dir(\mathbf{1})`; per-observation evidence is clipped at `e_k ≤ e_{max}` (a hyperparameter [TBD: see §IV implementation]) to bound the per-voxel concentration `S_v ≤ S_{max}` and keep vacuity numerically stable across long sessions. The semantic head is initialised from the publicly released Cylinder3D weights and the last layer is fine-tuned with (5) on the SemanticKITTI training split; for the RQ2 open-set runs the head is re-fine-tuned with `C = 14` known classes plus one explicit unknown channel, the unknown channel being trained only by the KL prior regulariser so that genuinely-unknown points receive uniform evidence and therefore high vacuity. The closest probabilistic-voxel competitors that we compare against in §IV are S-BKI [gan2020sbki], ConvBKI [wilson2024convbki], LatentBKI [wilson2024latentbki], and OpenVox [deng2025openvox]; the BKI line uses a spatial-kernel-coupled posterior whose vacuity is contaminated by neighbour evidence (which is why M3 cannot drive its decay clock from it), while OpenVox uses a per-instance Bernoulli rather than a per-voxel Dirichlet, so its posterior cannot expose the same single-scalar vacuity that M2 and M3 of our system require.

## III.C  M2: Vacuity-Conditioned Loop Closure

Loop closure operates on submaps sealed every `D = 50 m` of travelled distance or `T = 30 s` of wall time, whichever first; each submap `S` is summarised by a descriptor that explicitly exposes vacuity as a first-class channel rather than collapsing it into a confidence weight. Concretely, the descriptor is

$$
d(\mathcal{S}) \;=\; \big( h_{\text{class}}(\mathcal{S}), \; h_{\text{vac}}(\mathcal{S}) \big), \tag{7}
$$

where `h_class(S) ∈ R^{C+1}` is the `L^1`-normalised class histogram over voxels with vacuity below the per-submap median, and `h_vac(S) ∈ R^{B}` is the histogram of per-voxel vacuity `u_v` bucketed into `B = 10` uniform bins over `(0, 1]`. The class channel preserves the conventional submap-fingerprint role and the vacuity channel encodes the *epistemic shape* of the submap — a frequently-revisited submap is dominated by low-vacuity bins, while a recently-explored or sparsely-observed submap concentrates mass in high-vacuity bins.

Candidate matches between two submaps `S_1, S_2` are scored by a convex combination of cosine similarity on the class channel and cross-entropy on the vacuity channel,

$$
\mathrm{sim}(\mathcal{S}_1, \mathcal{S}_2)
\;=\;
(1-\beta) \cdot \frac{\langle h_{\text{class}}^{(1)}, h_{\text{class}}^{(2)} \rangle}{\Vert h_{\text{class}}^{(1)} \Vert \, \Vert h_{\text{class}}^{(2)} \Vert}
\;-\;
\beta \cdot \mathrm{H}\!\left( h_{\text{vac}}^{(1)} \,\Vert\, h_{\text{vac}}^{(2)} \right) , \tag{8}
$$

with mixing weight `β ∈ [0, 1]` (a single hyperparameter, [TBD: see §IV ablations A-3]). The cross-entropy term penalises matches between submaps whose epistemic profiles disagree even when their class distributions are similar — a typical failure mode of class-histogram-only descriptors when revisiting an explored area in which most class mass is dominated by a single class (e.g., a long road segment).

A candidate match `(\mathcal{S}_1, \mathcal{S}_2)` is verified by geometric registration restricted to voxels whose vacuity is below the per-submap median in *both* submaps; the post-registration consistency check thresholds the median per-class disagreement of the overlap region,

$$
\mathrm{Disagree}(\mathcal{S}_1, \mathcal{S}_2)
\;=\;
\mathrm{median}_{v \in \mathcal{O}}
\left[ 1 - \big\langle \mathbb{E}[p \mid \alpha_v^{(1)}],\, \mathbb{E}[p \mid \alpha_v^{(2)}] \big\rangle \right] , \tag{9}
$$

with `\mathcal{O}` the overlap voxel set after registration and `\langle \cdot, \cdot \rangle` the inner product of two posterior-mean distributions. Verified matches trigger pose-graph optimisation; inter-session fusion in the overlap region is then performed by *conjugate addition* of Dirichlet concentrations, `α_v ← α_v^{(1)} + α_v^{(2)}`, which is the maximum-likelihood combination under the assumption of independent observations across sessions and which introduces no fusion hyperparameter.

The construction contrasts with the descriptors used in Hydra [hughes2022hydra] and Kimera-Multi [tian2022kimeramulti], which encode submap fingerprints either through bag-of-words or through learned aggregation but do not expose epistemic mass as an explicit descriptor channel; it also contrasts with the semantic-graph-with-GAT loop closure of [yang_2025_semanticloop] and SA-LOAM [li_2021_saloam], whose semantic enrichment is a class-level signal rather than an uncertainty signal.

## III.D  M3: Vacuity-Driven Voxel Decay

Voxel decay applies a conjugate exponential pull of `α_v` toward the uniform prior, with the decay time-constant `τ` itself a function of vacuity. Let `a_v = t_{\text{now}} - t_{v,\text{last-update}}` be the age of voxel `v` since its last evidence accumulation,

$$
a_v \;=\; t_{\text{now}} \,-\, t_{v,\text{last-update}} . \tag{10}
$$

The decay time-constant is a linear interpolation between `τ_{\min}` (high-vacuity voxels age fast) and `τ_{\max}` (low-vacuity voxels age slowly):

$$
\tau(u_v) \;=\; \tau_{\min} \,+\, (\tau_{\max} - \tau_{\min}) \cdot (1 - u_v) . \tag{11}
$$

A confidently-known voxel (`u_v \to 0`) decays with `τ \to τ_{\max}` (target [TBD: ≈ one hour]); a maximally-uncertain voxel (`u_v \to 1`) decays with `τ \to τ_{\min}` (target [TBD: ≈ one minute]). The decay itself is the standard Dirichlet conjugate pull toward the uniform prior, applied to the *concentration* with the unit prior pinned in place so that `α_{v,k} \geq 1` is preserved (the Dirichlet constraint that makes (2) well-defined):

$$
\alpha_v^{(t + \Delta)} \;=\; \big( \alpha_v^{(t)} - \mathbf{1} \big) \cdot \exp\!\Big( -\frac{\Delta}{\tau(u_v)} \Big) \;+\; \mathbf{1} . \tag{12}
$$

By construction, (12) preserves the posterior mean direction (the ratio `α_k / S` is unchanged when `α_v - 1` is scaled uniformly only as long as no class dominates the prior, which is the regime where the decay is intended to act) and inflates vacuity monotonically with `a_v`. The net behaviour is that stale voxels become *re-writable* — their vacuity rises until fresh evidence either reasserts the previous class identity (drawing vacuity back down) or replaces it (the new evidence dominates the now-small `α_v - 1` residual).

The mechanism contrasts with the explicit per-submap activity flag of Panoptic Multi-TSDFs [schmid2022panopticmultitsdfs] and with the learned short-/long-term factoriser of Khronos [schmid2024khronos]: in both prior systems, the decision *when to forget* is taken by a mechanism that does not consume the voxel-level posterior, whereas in (12) the decision is *fully determined* by the same Dirichlet vacuity that M1 produces and M2 consumes. It also contrasts with the monotonically-growing weight schedule of nvblox [millane2024nvblox] and the long-term object handling of LTC-Mapping [matezbandera2022ltcmapping], which do not implement any decay at all.

*Dynamic objects.* A side-effect of (11)–(12) is that moving-object voxels experience naturally elevated vacuity (inconsistent per-frame evidence keeps `S_v` low relative to inter-class disagreement) and therefore decay aggressively, so transient dynamic evidence is flushed before it can crystallise into a wrong-persistent label. The empirical RQ5 head-to-head against Khronos [schmid2024khronos] in §IV measures whether this implicit mechanism is competitive enough to make the explicit short-/long-term factoriser a deployment-cost trade-off rather than a capability necessity.

## III.E  Conjugate Cross-Module Coupling

Modules M1, M2, and M3 share a single per-voxel state `(α_v, u_v)` and a single arithmetic primitive — conjugate Dirichlet addition. M1 accumulates `α_v` from evidence (Eq. 4); M2 fuses `α_v^{(1)}` and `α_v^{(2)}` across sessions by the same addition (post-Eq. 9); M3 pulls `α_v` toward the prior by (12), which is itself the time-discretisation of the same Dirichlet-conjugate update against a synthetic uniform observation. Vacuity (Eq. 2) is computed once per voxel per fusion step and read three times: once by the open-set head (where high `u_v` flags an unknown-category voxel for the RQ2 evaluation), once by the M2 descriptor (`h_{\text{vac}}` in Eq. 7), and once by the M3 trigger (`τ(u_v)` in Eq. 11). No module needs to recompute `u_v` and no module needs to maintain a private auxiliary uncertainty quantity. This is the load-bearing observation that motivated EvidLife-Map's design: the same scalar that gives M1 its calibrated semantic posterior also gives M2 its descriptor entropy channel and M3 its decay clock, so the *integrated* claim "one vacuity, three jobs" is realised at zero additional per-voxel runtime cost beyond the single division that produces `u_v` from `α_v`. A direct corollary is that any ablation that disables one consumer of `u_v` leaves the other two consumers and the cost of computing `u_v` itself unchanged — the §IV.G ablation grid measures exactly this, and the cost accounting in §III.F shows that the only per-voxel state added relative to a non-evidential nvblox [millane2024nvblox] baseline is `(C+1)` floats for `α_v` plus a single timestamp.

## III.F  Implementation on nvblox

We implement the per-voxel state inside the nvblox [millane2024nvblox] layer cake as an `EvidentialLayer<VoxelType>` whose `VoxelType` extends the canonical nvblox semantic voxel with `(α_v ∈ R^{C+1}, t_{v,\text{last-update}}, \text{pose-idx}_v)`. The layer participates in the standard nvblox tick: per-frame point splatting writes into `α_v` via (4); the GPU streaming-multiprocessor block layout of nvblox is preserved unchanged, so the only structural change relative to a baseline nvblox semantic layer is the wider voxel record and the additional per-voxel timestamp. Per-voxel memory grows from approximately `100 B` in R2 [jiao2024r2] (label probabilities only) to approximately `100 B + (C+1) \times 4 B + 4 B + 4 B \approx 180 B` at `C = 19`, giving a nominal `44 kB` storage per `8^3` block of fully-allocated voxels and roughly `1.7×` the per-voxel footprint of R2. The decay step (12) is implemented as a periodic kernel over allocated voxels with `a_v > a_{\min}` (a small hysteresis threshold [TBD: see §IV implementation]) to avoid touching voxels updated in the current frame; on a Jetson Orin NX the kernel sweeps an active map at the same cadence as the standard nvblox mesh updater. The M2 descriptor (Eq. 7) is computed on-demand at submap-seal time, reusing the per-block reductions that nvblox already performs to maintain its visualisation mesh; conjugate inter-session fusion (post-Eq. 9) reuses the same evidence-addition kernel as M1, so inter-session fusion costs the same per-voxel as a single observation step. All other state (LVIO pose stream, TSDF backbone, mesh extractor) is reused from the upstream nvblox release.

---

# §IV  Experiments

[§IV–§VI to be back-filled after experiments — see research_plan.md (v3) §4 and paper_outline_v2.md (v3) §IV for the planned tables, figures, and metric definitions. Headline numbers to instantiate: closed-set mIoU XX.X vs ConvBKI's 77.7 [wilson2024convbki] (RQ1); ECE XX.X% (RQ1); open-set AUROC XX.X / AUPR XX.X on the 14/5 split (RQ2); safe-region recall XX.X% on SemanticSpray [piroli_2023_semanticspray] (RQ3); stale-voxel removal precision XX.X% on KITTI-360 (RQ4); dynamic-object mIoU within XX.X of Khronos [schmid2024khronos] (RQ5); end-to-end latency XX.X Hz on Jetson Orin NX (C4).]

[Table I: placeholder, fill at §IV experiments — closed-form posterior comparison across EDL-Dirichlet (ours) vs S-BKI [gan2020sbki] vs ConvBKI [wilson2024convbki] vs R2 argmax-Bayes [jiao2024r2].]

[Table II: placeholder, fill at §IV experiments — datasets × RQ matrix.]

[Table III: placeholder, fill at §IV experiments — RQ1 main result.]

[Table IV: placeholder, fill at §IV experiments — RQ2 open-set lead.]

[Table V: placeholder, fill at §IV experiments — RQ5 Khronos dynamic head-to-head.]

[Table VI: placeholder, fill at §IV experiments — RQ3 traversability.]

[Table VII: placeholder, fill at §IV experiments — RQ4 lifelong multi-session.]

[Table VIII: placeholder, fill at §IV experiments — ablations + Jetson runtime.]

---

# §V  Discussion

[To be back-filled after §IV experiments — see research_plan.md §6 risk register and paper_outline_v2.md §V for the planned limitations, failure modes, and honest-comparison structure.]

---

# §VI  Conclusion

[To be back-filled after §IV experiments — see paper_outline_v2.md §VI for the planned single-paragraph recap of C1–C4.]
