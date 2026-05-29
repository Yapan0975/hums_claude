# Kimera-Semantics wrapper

**Upstream:** `https://github.com/MIT-SPARK/Kimera-Semantics` (IJRR 2021).
**Paper:** Rosinol et al., "Kimera: an Open-Source Library for Real-Time
Metric-Semantic Localization and Mapping", IJRR 2021.

Closes the reviewer demand "did you cite the seminal MIT-SPARK ancestor"
(research_plan v3 §4.2 baseline #4). Configured for KITTI-360 RQ4 lifelong.

## Bring-up

```
docker pull mit-spark/kimera-semantics:latest
```

The adapter wraps the ROS node via the standard Kimera-VIO interface (LiDAR
front-end converts our SemKITTI / KITTI-360 streams to Kimera-Semantics inputs).
