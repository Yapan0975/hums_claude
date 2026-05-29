# RQ2 — Open-set vacuity (lead RQ in v3)

## Hypothesis (research_plan v3 §2 H2 primary)

On SemanticKITTI 14-known / 5-unknown split, EvidLife-Map's vacuity achieves
**voxel-level AUROC ≥ 0.80** and **AUPR ≥ 0.60**, while closed-set mIoU on
the 14 known classes drops by **≤ 1.5** vs the 19-class supervised baseline.

## Gates

* G-5 at W17 (research_plan v3 §9): AUROC ≥ 0.80 AND AUPR ≥ 0.60.
* If red: switch to Robo3D fallback per H2-fallback within the same week.

## Run grid

| Split           | Systems                       | Datasets         |
|-----------------|-------------------------------|------------------|
| Primary (14/5)  | EvidLife-Map, R2-reimpl,      | SemKITTI val 08  |
|                 | ConvBKI, S-BKI                |                  |
| Robustness (16/3)| EvidLife-Map only            | SemKITTI val 08  |
| Reverse OOD     | EvidLife-Map                  | nuScenes-LiDARSeg|

Estimated compute: 4 GPU-days (research_plan v3 §4.5).
