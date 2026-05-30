# Phase 1 — R3 Cross-Disciplinary Perspective Review

**Persona**: MIT CSAIL Associate Professor, NeurIPS Bayesian DL Workshop co-author + Tier-1 AV Safety Lead. NOT a Voxblox/SLAM specialist. Has seen EDL collapse in production on rare decision-relevant OOD events.

**Recommendation**: Major Revision  
**Confidence**: 3/5 (defers SLAM-lineage to R2, W3 ablation rigor to R1)  
**Weighted average**: 63/100

## Critical
- **W1 (Critical)** — Wrong Dirichlet scalar: paper uses *vacuity* but §V.B(a) confidently-wrong-evidence failure mode is exactly where *dissonance* targets. Vacuity is LOW when S_v is high — exactly when wrong-confident evidence has accumulated. Dissonance RISES when class evidence splits. Add §III.B Sensoy 2018 family enumeration; state explicitly that vacuity is the only Dirichlet scalar admitting Eq. 12 closed-form conjugate decay; add §IV.H "A-1.5 dissonance-conditioned τ" ablation on §V.B(a)/(c) subsets.
- **W2 (Critical)** — §V.B(d) per-table best-checkpoint-by-target-metric = model-selection oracle NO REAL DEPLOYMENT can access. §V.B(d) is paper-internal evidence that the "one model" thesis breaks. Add §IV.H "Single-checkpoint joint serving" row reporting all 4 headline metrics from ONE checkpoint; explore unified KL schedule.

## Major
- **W3** — §IV.E traversability + §IV.F lifelong numbers are perception-internal, NOT decision-relevant for downstream planner. Name deferral consumer (Nav2 or R2's planner); qualitative video panel; OR soften §VI "closes four weaknesses" to "exposes four signals."
- **W4** — §V.B(c) taxonomically-close failure has no EDL-theoretic bound. Add closed-form upper bound on AUROC degradation vs cosine angle to closest training class, OR per-withheld-class AUROC vs taxonomic distance plot.

## Minor
- **W5** — §VI "≥ 0.82 at Cylinder3D" extrapolation in tension with §V.C honest framing. Rewrite §VI to mirror §V.C language; move extrapolation to §V.C as labeled hypothesis.

## Dimension scores
| Dim | Score | Descriptor |
|---|---|---|
| Originality (20%) | 72 | Strong |
| Methodological Rigor (25%) | 60 | Adequate |
| Evidence Sufficiency (25%) | 55 | Adequate |
| Argument Coherence (15%) | 65 | Adequate |
| Writing Quality (15%) | 75 | Strong |
| Significance & Impact | 58 | Adequate |
| **Weighted avg** | **63** | Major Revision |

**Acknowledgment**: vacuity IS uniquely admitting Eq. 12 closed-form — that's a real justification, paper should STATE it explicitly.
