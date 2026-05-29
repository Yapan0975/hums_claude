# Reproducibility (closes R2 audit E7)

The IROS supplementary package will ship:

1. This Git repository at the submission-time commit hash.
2. The `evidlife_map:cu121-humble` Docker image (≈ 4 GB; hosted on the lab
   registry, mirror on Docker Hub).
3. The trained M1 head checkpoints (`evidlife_kitti_closed_last.ckpt`,
   `evidlife_kitti_openset_last.ckpt`, `evidlife_kitti360_last.ckpt`).
4. The open-set 14/5 split definition (`configs/kitti_openset_split.json`).
5. The KITTI-360 revisit-pair list (`configs/kitti360_revisit_pairs.json`).
6. The SemKITTI dynamic-split selection script
   (`experiments/rq5_khronos_dynamic/select_dynamic_frames.py`).
7. The Jetson Orin NX 30 s demo video (`evidlife_demo.mp4`).
8. Per-experiment seeds and hyperparameter tables
   (`experiments/*/plan.md` + `runs/*.json`).
9. LVIO configuration files documented per dataset.

## How to reproduce a §IV table from scratch

* **Table III (RQ1 main):** `bash scripts/eval_rq1.sh configs/evidlife_kitti.yaml <ckpt>`
* **Table IV (RQ2 lead):**  `bash scripts/eval_rq2_openset.sh <ckpt> primary`
* **Table V (RQ5 Khronos):** `bash scripts/eval_rq5_khronos.sh experiments/rq5_khronos_dynamic/dynamic_frames.txt <ckpt>`
* **Table VI (RQ3 trav):**   `bash scripts/eval_rq3_traversability.sh <ckpt>` (script added in W21)
* **Table VII (RQ4 lifelong):** `bash scripts/eval_rq4_lifelong.sh configs/kitti360_revisit_pairs.json <ckpt>`
* **Table VIII (ablations + Jetson):** `bash scripts/benchmark_jetson.sh <cfg> <ckpt> <rosbag>`

Each script writes a JSON to `/runs/` whose schema is documented inline.
