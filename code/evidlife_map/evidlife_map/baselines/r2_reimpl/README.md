# R2 re-implementation

**Source paper:** Jiao et al., "Real-time Metric-Semantic Mapping for Autonomous
Navigation in Outdoor Environments", T-ASE 2024 (arXiv 2412.00291).
**Cleaned text:** `D:\_7_sci\semantic_mapping\_Online_Metric_Semantic_Mapping_for_Autonomous.txt`.

## Why we re-implement

R2 publishes no code (paper_tables.md Table III row 1). The R2 audit (S1/S2/S4)
notes that the per-voxel Bayes filter is unspecified — we therefore implement
the **most charitable** interpretation: an argmax-Bayes update of the per-voxel
posterior, using HRNet's softmax + confidence head as the per-point likelihood.

## Pipeline mapping

```
R2 stage           Our file                         Notes
-----------------  -------------------------------- -------------------------------
LVIO odometry      r2_pipeline.py (TODO W1)         delegate to FAST-LIO2 / R3LIVE
                                                    via the LVIO config in YAML
LiDAR ray-cast     r2_pipeline.py (TODO W1)         reuse nvblox's projective splat
HRNet semantic     r2_pipeline.py (TODO W1)         confidence-aware aleatoric head
                                                    (paper §III-B of R2; arXiv p.3)
per-voxel Bayes    bayes_filter.py                  argmax-Bayes; this file is
                                                    actually implemented
mesh extraction    r2_pipeline.py (TODO W2)         reuse nvblox MeshLayer
trav classifier    r2_pipeline.py (TODO W4)         R2's "unlabeled = untraversable"
                                                    hard rule
```

## Charitable-reimpl checklist

* [ ] HRNet pretrained weights resolved.
* [ ] LVIO module wired up (FAST-LIO2 on KITTI-360, R3LIVE on SemKITTI).
* [ ] Per-voxel argmax-Bayes update reproduced with paper-published latency
  on RTX 3080 Ti translated to RTX 4060 timing (≈ 2× slow-down).
* [ ] mIoU within ±2 of the "unspecified-filter charitable interpretation"
  documented in this file.

The W1 deliverable (research_plan v3 §7 P1) is the first three bullets; the
last two are completed in W2 - W3 (G-1 checkpoint).
