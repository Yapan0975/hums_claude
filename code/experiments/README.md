# Experiments

One directory per RQ in research_plan v3 §2 + §4. Each carries:

* `plan.md` — the per-RQ run plan (datasets, baselines, seeds, gating).
* Helper scripts to assemble the splits and parse the JSON outputs.

The actual `*.json` results land in `runs/`; that directory is git-ignored.

## RQ index

| RQ | Lead question                        | Directory                         |
|----|--------------------------------------|-----------------------------------|
| 1  | Closed-set mIoU + ECE                | `rq1_baselines/`                  |
| 2  | Open-set vacuity (14/5 split)        | `rq2_openset/`                    |
| 3  | Uncertainty-aware traversability     | `rq3_traversability/`             |
| 4  | Lifelong multi-session on KITTI-360  | `rq4_lifelong/`                   |
| 5  | Dynamic vs Khronos                   | `rq5_khronos_dynamic/`            |
