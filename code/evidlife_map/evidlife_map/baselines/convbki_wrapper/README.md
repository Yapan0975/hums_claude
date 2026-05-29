# ConvBKI wrapper

**Upstream:** `https://github.com/UMich-CURLY/BKI_ROS` (T-RO 2024).
**Paper:** Wilson et al., "ConvBKI: Real-Time Probabilistic Semantic Mapping…",
T-RO 2024 (deep-read at `精读_T1.4_ConvBKI.html`).

## Bring-up notes (research_plan v3 §6 R-12)

ConvBKI is one of the two **R-12 high-risk** reproductions (research_plan v3
allocates W2 + W3 to bring it up on RTX 4060).

Steps:

1. `git clone https://github.com/UMich-CURLY/BKI_ROS external/convbki`
2. Pin CUDA 12.1 + PyTorch 2.2 inside our Docker image (matches the upstream
   requirements at the time of fork).
3. Cross-check published `77.7 %` mIoU on KITTI Odom seq 15 with our 4060 run
   (must be within ±2 mIoU per G-1).
4. If R-12 fires (reproduction fails after 2 weeks): document the deviation
   and use published numbers in a quoted-only Table I cell.

## What the wrapper exposes

`ConvBKIAdapter` accepts our standard ``(N, 3)`` LiDAR cloud + per-point
class probabilities and returns a per-voxel posterior in the **same shape**
EvidLife-Map uses, so the §IV harness can swap implementations cleanly.
