# RQ4 — Lifelong / multi-session

## Hypothesis (research_plan v3 §2 H4)

Across KITTI-360 sequences 00, 02, 04, 05, 06, 07, 09, 10 (≥ 5 revisit pairs
built from odometry overlap):

1. Stale-voxel removal precision ≥ 80 %.
2. Inter-session map ECE drift ≤ 1.5 × single-session ECE.
3. Multi-session memory footprint ≤ 1.7 × single-session footprint.

## Gates

* G-3 at W3: revisit-pair list delivered with ≥ 5 pairs, ≥ 30 m sustained
  overlap each.
* If red: synthetic-temporal-gap injection on single sessions (R-13 mitigation).

## Sequences and revisit pairs

To be populated by `build_revisit_matrix.py` (W3 deliverable). Expected output:

```json
[
  {"sequence_a": "...00", "sequence_b": "...02",
   "frame_a": 120, "frame_b": 980, "overlap_m": 47.2},
  ...
]
```

Estimated compute: 4 GPU-days (research_plan v3 §4.5).
