# Khronos wrapper (R-17 high-risk)

**Upstream:** `https://github.com/MIT-SPARK/Khronos` (RSS 2024).
**Paper:** Schmid et al., "Khronos: A Unified Approach for Spatio-Temporal
Metric-Semantic SLAM in Dynamic Environments", RSS 2024.

## Risk

R-17 is the **single highest-risk** reproduction in the v3 plan because
Khronos targets RGB-D + IMU on TESSE (Apartment / Office), not LiDAR-only on
SemKITTI; *every* sensor + dataset + platform axis differs. Mitigations:

1. W17 — email authors about the SemKITTI-port effort.
2. W18 – W20 — full Khronos bring-up on our docker, gated by **G-6 (W20)**.
3. Fallback: ship Table V as a "published numbers vs ours" hybrid with explicit
   disclaimer in §V. RQ5 H5 weakens to "competitive with published Khronos
   numbers".

## Reproduction notes

* Pin Khronos commit `<TBD>` matching the arXiv 2402.13817 release.
* CPU-only execution is fine for the dynamic-split numbers — we measure
  Jetson latency on EvidLife-Map only (research_plan v3 §4.6).
* SemKITTI dynamic split selection: see
  `../../experiments/rq5_khronos_dynamic/plan.md`.
