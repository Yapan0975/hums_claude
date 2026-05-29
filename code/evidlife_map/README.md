# EvidLife-Map

**Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay.**
Target venue: IROS 2027. Companion repo to `paper_v1.md` and `research_plan.md`
(both at `../_new_paper/drafts/`).

This is the **P1 implementation skeleton**: the package imports cleanly, all
formulas listed in `paper_v1.md` §III have a corresponding Python entry point,
all tests can be collected by `pytest`, but no end-to-end training is run yet.
The W1–W3 bring-up gate (G-1, G-3) consumes this skeleton as its starting point.

## Repository layout

```
evidlife_map/
  m1_evidential/        # M1 — Dirichlet evidential per-voxel fusion (§III.B, Eq 1–6)
  m2_loop_closure/      # M2 — vacuity-conditioned loop closure  (§III.C, Eq 7–9)
  m3_decay/             # M3 — vacuity-driven voxel decay       (§III.D, Eq 10–12)
  core/                 # shared voxel-map abstraction + nvblox bridge
  data/                 # dataset wrappers (SemKITTI / nuScenes / KITTI-360 / SemSpray)
  eval/                 # metrics (mIoU / ECE / AUROC / lifelong / traversability)
  baselines/            # R2 reimpl, ConvBKI / S-BKI / Kimera / Khronos / OpenVox wrappers
  jetson/               # Jetson Orin NX deployment + 30 s demo capture (C4)
configs/                # YAML configs, one per dataset
scripts/                # training + evaluation shell entry points
tests/                  # pytest unit tests
docs/                   # architecture diagram, Go/No-Go checklist, reproducibility
```

## Reproduce in 5 steps

```bash
# 1. Build the Docker image (CUDA 12 + ROS 2 humble + nvblox bindings)
docker compose build evidlife_x86

# 2. Drop into the container with this repo mounted at /workspace
docker compose run --rm evidlife_x86 bash

# 3. Editable install (inside the container)
pip install -e .

# 4. Sanity-check the skeleton: every public symbol imports, every test collects
pytest --collect-only

# 5. (After datasets are downloaded — outside this skeleton) bring up R2-reimpl
bash scripts/train_m1.sh configs/evidlife_kitti.yaml
```

## Hardware

* **Training / main eval:** 1× RTX 4060 (16 GB). Batch size = 1 with grad
  accumulation; Cylinder3D backbone frozen; only the last layer fine-tuned with
  the EDL loss (Eq 5).
* **Deployment + demo video:** 1× Jetson Orin NX (16 GB), TensorRT FP16 build of
  the inference path. See `evidlife_map/jetson/`.

## Citation pointers

The Python module docstrings cite the relevant equation by `§III.B Eq 4` form;
all references are to `../_new_paper/drafts/paper_v1.md` v1.

## License

Apache-2.0 (planned; see `LICENSE` once added). All third-party baseline
wrappers retain their original licences — see each `baselines/*/README.md`.
