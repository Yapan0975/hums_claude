# Phase 1 — R4 Devil's Advocate Report

**Persona**: Hostile-stance T-RO/IJRR Associate Editor, has published rebuttals to 2 prior conjugate-uncertainty SLAM papers where single-scalar claim collapsed. NOT a Voxblox insider. Skeptical of "one scalar, N jobs" framings.

**Recommendation**: Major Revision with stretch toward Reject

## Strongest Counter-Argument (~280 words)

The §I.B core thesis — "one vacuity, three jobs" — is presented as the load-bearing scientific contribution that justifies a single IROS paper rather than three separate Letters. **The paper's own preliminary evidence destroys this thesis.** §V.B(d) admits: "the two modes therefore require different KL schedules for mIoU vs AUROC: warm-restart for mIoU, plain linear for AUROC. We use best-checkpoint-by-target-metric selection in all downstream tables." Translation: the system CANNOT deliver job (i) open-set discrimination AND the calibration target of job (iii) from a single model. Two checkpoints, two schedules, one ad-hoc selection per metric.

Jobs (i) and (ii) are not novel by themselves: (i) is the 2018 Sensoy vacuity-as-OOD pattern applied to LiDAR points (published 10+ times since); (ii) is a one-line entropy-histogram add-on that ANY uncertainty scalar would equally fill, with no demonstration vacuity is specifically load-bearing. Only job (iii), the conjugate vacuity-conditioned decay, is genuinely novel.

**Principled paper: one RA-L on M3 alone, with M1 demoted to standard Dirichlet-EDL and M2 demoted to "we add an entropy channel" ablation against prior art.** IROS-length packaging is rhetorical, not scientific integration.

## 3 CRITICAL issues (block Accept per IRON RULE #4)
1. System ships 2 checkpoints / 2 KL schedules / best-ckpt-by-target-metric selection — paper-internal evidence integration thesis fails
2. Vacuity-in-particular NEVER shown load-bearing for jobs (ii)+(iii); no competing-scalar ablation
3. §V.C admits AUROC 0.781 < pre-registered G-5 = 0.80; "Cylinder3D extrapolates to 0.85+" has ZERO empirical support — post-hoc rescue clause

## 5 MAJOR issues
- W4: §III.B kernel rejection forced-by-M3 not principled (hybrid not even mentioned in §V.A)
- W5: Eq. 12 mean-direction preservation only when no class dominates — NOT the regime it claims to serve
- W6: Rare-class IoU = 0 confounds RQ2 AUROC between "no concept" and "high vacuity"
- W7: Job (i) is standard Sensoy 2018 — C1 overclaimed
- W8: Tab II "≥ +2 vs R2" set against UNRELEASED R2 number not against ConvBKI ceiling

## So-What Test
- **M1**: medium loss / standard Sensoy substrate adaptation. Verdict: retain but DOWNGRADE C1 to "MSM substrate adaptation."
- **M2**: small loss / load-bearing-rhetoric-only / §V.B(b) failure mode collapse admission self-undermines. Verdict: CUT and demote to half-page ablation.
- **M3**: LARGE LOSS / only component surviving / paper structure should be one IROS or RA-L on M3 alone.

## Adjudication Demand: Decoupling Ablation
3 trainings × ~40 min = ~2 hours on 5090:
1. **Vacuity-everywhere** (paper's current)
2. **Dissonance-for-decay**: u_v for OOD + h_vac, but τ(dissonance) in Eq. 11
3. **Softmax-entropy-for-M2**: u_v for OOD + τ, but softmax-entropy histogram in Eq. 7

**Pass condition for §I.B thesis**: config (1) strictly Pareto-dominates configs (2) and (3) on all 4 metrics. Otherwise §I.B reframed as "system-engineering convenience, not scientific contribution."

## Strengths grudgingly acknowledged
- §V.B(d) honest disclosure of warm-restart trade-off
- W3-A → W3-E spatial-adjacency audit genuinely good science
- M3 conjugate decay (Eq. 11-12) genuinely novel
- nvblox `EvidentialLayer<VoxelType>` non-trivial engineering
- Tab III risk anchoring excellent project management
