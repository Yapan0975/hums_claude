# Phase 1 — R2 Domain Review

**Persona**: MIT-SPARK Lab Senior Research Scientist (Carlone group). Co-author/heavy user of Kimera/Hydra/Clio/Khronos. Reviewed Khronos RSS 2024 & Clio RA-L 2024.

**Recommendation**: Major Revision  
**Confidence**: 5/5  
**Weighted average**: 63/100

## Critical
- **W3 (Critical)** — R2-reimpl is load-bearing baseline for RQ1/RQ3/RQ4/RQ5 but no `reproduce.yaml`; §IV.0 "R2 (CE PointNet, argmax)" is NOT R2's confidence-aware iterative Bayes filter. Need §IV.B "R2 re-implementation faithfulness" subsection (matched/reconstructed/swapped table); two R2 rows in Tab IV (CE-argmax proxy + iterative Bayes); drop §I.C C1 "generalises R2 Bayes filter" unless iterative-Bayes-as-special-case is proven.

## Major
- **W1** — §III.B kernel-rejection asserted as principled but §IV.H lacks "+light spatial kernel on M1, vacuity un-smoothed" ablation. §V.A(v) admits mIoU is "neighbourhood-bound" — exactly what kernel addresses. Add §IV.H A-8 row.
- **W2** — §IV.G RQ5 vs Khronos pre-concedes methodological parity; §III.D dynamic-vacuity mechanism is post-hoc (vacuity ≠ consistency-of-evidence). Benchmark Khronos in native RGB-D + workstation config alongside Orin NX; derive moving-object vacuity-elevation formally.
- **W5** — SLIM-VDB ECE mis-claim: §IV.C line 461 "first comparable ECE in BKI/Voxblox lineage" is FACTUALLY WRONG (SLIM-VDB publishes ECE). Also missing Hydra-Multi systems-unification precedent; missing ConceptGraphs/OpenScene CLIP-volumetric branch.

## Minor
- **W4** — C1/C2 over-credit lineage-shared Dirichlet primitives. C1→"first per-voxel Dirichlet posterior DELIBERATELY without spatial-kernel smoothing"; C2→"first conjugate Dirichlet pseudo-count addition for inter-session submap fusion, replacing per-submap activity flag of Panoptic Multi-TSDFs."

## Dimension scores
| Dim | Score | Descriptor |
|---|---|---|
| Originality (20%) | 68 | Adequate |
| Methodological Rigor (25%) | 62 | Adequate |
| Evidence Sufficiency (25%) | 48 | Weak |
| Argument Coherence (15%) | 74 | Strong |
| Writing Quality (15%) | 76 | Strong |
| Literature Integration | 65 | Adequate |
| **Weighted avg** | **63** | Major Revision |
