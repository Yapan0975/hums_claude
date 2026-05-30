# Phase 0 — Field Analysis Report

(Persona configuration for the 5-reviewer panel. Full content captured in this file from Phase 0 of /ars-reviewer.)

## Paper Basic Information
- **Title**: EvidLife-Map: Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay
- **Target venue (declared)**: IROS 2027 (~2027-03 deadline), 8-page IEEE conference
- **Length**: 683 lines, ~12k words (currently 132% of outline budget)
- **References cited**: 30/60 entries in refs.bib

## Field Analysis (6 dimensions)
- **Primary discipline**: Robotics — 3D Perception / Online Metric-Semantic Mapping (LiDAR voxel-grid SLAM extension; nvblox → Voxblox → S-BKI/ConvBKI → R2/Jiao 2024 lineage)
- **Secondary disciplines**: (a) Evidential / Bayesian Deep Learning (Sensoy 2018, Amini 2020); (b) 3D Point-Cloud Semantic Segmentation; (c) Probabilistic Robotics & Long-Term Autonomy; (d) Open-Set / OOD Recognition
- **Research paradigm**: Systems-integration empirical paper with small formal-derivation core (Eqs 1-12 textbook Dirichlet conjugacy on new substrate)
- **Methodology type**: Quantitative empirical + system engineering with closed-form math + 5-dataset evaluation + ablation grid + Jetson Orin NX deployment
- **Target venue tier**: IROS 2027 = Q1 robotics; appropriate-to-slightly-conservative for the contribution breadth
- **Paper maturity**: Late-skeleton / preliminary-verified, pre-camera-ready (§I-III drafted, §IV-VI placeholders gated on Cylinder3D unblock)

## Recommended Target Venues (Top 3)
1. **IROS 2027** (author's declared target) — Strong / canonical-correct
2. **ICRA 2027** — Strong / fully equivalent (earlier deadline, sibling venue)
3. **IEEE RA-L** + optional IROS 2027 presentation — Strong as journal extension

## 5 Reviewer Persona Cards
- **EIC**: Cesar Cadena (ETH Zurich Autonomous Systems Lab) — IROS PC chair persona
- **R1 Methodology**: CMU Robotics Institute Associate Prof — W3 audit chain rigor, KL schedule, per-class honesty
- **R2 Domain**: MIT-SPARK Lab Senior Research Scientist — lineage, Khronos/Clio/GS-LIVO positioning, kernel-rejection ablation, R2-reimpl faithfulness
- **R3 Perspective**: MIT CSAIL Associate Prof + AV Safety Lead — EDL theory (vacuity vs dissonance), calibration-vs-discrimination deployment trade-off
- **R4 Devil's Advocate**: T-RO/IJRR Associate Editor hostile persona — attacks "one vacuity, three jobs" framing
