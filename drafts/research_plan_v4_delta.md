---
title: "EvidLife-Map · Research Plan v4 Delta (v3 → v4)"
version: v4 delta
date: 2026-05-28
supersedes: research_plan.md (v3, 2026-05-28)
trigger: "Hardware reality check — RTX 5060 (8 GB) on hand, not RTX 4060 (16 GB) as v3 assumed; user opted for dual-track (5060 local + cloud train)"
---

# v3 → v4 Delta — Hardware-Reality Patch

> This is a **delta document**, not a full rewrite. Apply on top of research_plan.md (v3). Items not listed here are unchanged.

## §0.7 Why v4

User confirmed hardware reality during W0 preflight (2026-05-28):

```
GPU:     NVIDIA GeForce RTX 5060 (Blackwell, sm_120)
VRAM:    8 GB
Driver:  595.79 / CUDA 13.2 (runtime-compat with cu128 wheels)
```

This diverges from v3's "RTX 4060 16 GB" assumption in two ways:
1. **VRAM halved** (16 → 8 GB) — large-model training, big-batch ablations, multi-class C=100+ extensions all break.
2. **Blackwell sm_120** — needs torch ≥ 2.7 + CUDA 12.8 wheels (no cu121 / cu118 fallback). Torch 2.11.0+cu128 confirmed available.

User chose **dual-track execution**:
- **Local 5060** → development, smoke tests, unit tests, real-data sanity (1-2 frame inference), Jetson cross-compile, paper writing
- **Cloud GPU** → multi-day training runs, full SemanticKITTI eval, ConvBKI/Khronos reproduction, ablation matrix

## §0.8 v3 → v4 Change Log

| Dim | v3 | v4 | Reason |
|-----|-----|-----|--------|
| GPU | RTX 4060 16 GB single | 5060 8 GB local + cloud 4090/A100 on demand | Hardware reality |
| Compute model | Local-only | Dual-track | User decision |
| Batch size | 4-8 (Cylinder3D) | 1-2 local fp16 + cloud as needed | VRAM-halved |
| Ablation matrix | 5 firm + 1-2 stretch | 4 firm local + 1-2 cloud-only | VRAM + cloud budget |
| Khronos reproduction | Local | Cloud-required | Khronos uses TESSE + 3DGS ≥ 12 GB |
| ConvBKI reproduction | Local | Cloud (4090) | Published numbers on RTX 3090; needs ≥ 12 GB |
| Cloud cost budget | Not present | $30-80 (~10 GPU-hours @ A100, ~80 GPU-hours @ T4) | New constraint |
| Timeline | 36 weeks | 36 weeks (unchanged) | Cloud absorbs VRAM cost |
| Page budget | 8 IROS | 8 IROS (unchanged) | — |

## §4.4 Ablation Matrix — REVISED

| Ablation | Where to run | v3 status | v4 status |
|----------|--------------|-----------|-----------|
| A-1 ± Dirichlet evidential head | Local 5060 (small fp16) | firm | **firm-local** |
| A-2 ± vacuity-conditioned decay τ(m_u) | Local 5060 | firm | **firm-local** |
| A-3 ± entropy channel in submap descriptor | Local 5060 | firm | **firm-local** |
| A-4 ± vacuity-aware traversability | Local 5060 | firm | **firm-local** |
| A-5 ± conjugate Dirichlet inter-session fusion | Cloud A100 (multi-session memory) | firm | **firm-cloud** |
| A-7 ± open-set channel (20-D vs 19-D) | Cloud A100 (full SemKITTI val) | stretch | **firm-cloud** |
| A-6 ± voxel size (0.10 vs 0.25 m) | Cloud A100 (memory-bound) | stretch | **drop** (cloud GPU-hr too expensive for rebuttal-only) |

Total: 6 ablations (5 firm + 1 promoted) — same coverage as v3, redistributed across hardware.

## §4.6 Cloud Train Strategy (new section)

### Cloud provider candidates
1. **autoDL (autodl.com)** — Chinese, A100 40GB ~¥4-6/hr, T4 16GB ~¥1.5/hr, ssh + jupyter, persistent storage
2. **Vast.ai** — global spot, RTX 4090 ~$0.30-0.50/hr, A100 80GB ~$1.20-2.00/hr
3. **RunPod** — global, A100 80GB ~$1.69/hr, fast spin-up

**Recommendation**: autoDL for routine ConvBKI / S-BKI reproduction (cheap + ZJUT network is fast). Vast.ai or RunPod for Khronos + final eval (need newer drivers + larger VRAM).

### Cost budget
- ConvBKI reproduction: ~6 GPU-hr on A100 → ~$10
- Khronos reproduction: ~10 GPU-hr on A100 → ~$15-20
- Full SemKITTI val (6 systems): ~12 GPU-hr → ~$18
- Ablation A-5, A-7: ~8 GPU-hr → ~$12
- 30% buffer for re-runs and OOM debugging → ~$20
- **Total budget cap**: **$80** (~¥600)

### Workflow
1. Develop + unit-test on local 5060 (fast, no $$).
2. Smoke test on local with 100-point subsample.
3. Push to cloud GPU, run via `bash scripts/cloud_train.sh <experiment_name>`.
4. Pull artifacts (model + logs + figures) back to D:\_7_sci\.
5. Iterate.

### `scripts/cloud_train.sh` checklist (W2 deliverable)
- [ ] Cloud env setup script (apt + conda + torch + repo clone)
- [ ] Data sync (rsync from local SemanticKITTI subset → cloud)
- [ ] Experiment runner with auto-shutdown on completion
- [ ] Artifact pull-back with checksums

## §6 Risk Register — DELTA

### New risks added in v4
- **R-18** (H): Cloud cost overrun. **Trigger**: cumulative spend > $60. **Mitigation**: hard stop at $80; switch from A100 to T4 for non-critical ablations; reuse published numbers where possible.
- **R-19** (M): Local 5060 OOM on data loader. **Trigger**: OOM during nuScenes load with batch ≥ 2. **Mitigation**: chunked frame loading (already in place in `data/` modules); accumulate gradients.
- **R-20** (M): Cloud train artifact pull-back fails. **Trigger**: cloud session expires before sync. **Mitigation**: auto-rsync after each epoch; checkpoint to cloud-persistent volume; explicit "fetch when done" workflow.

### Risks reclassified
- **R-11** (4060 VRAM) → **R-11'** (5060 8GB VRAM) — even more critical; severity unchanged H, probability ↑
- **R-12** (ConvBKI reproduction) — probability ↓ from M to L because we'll use the published-on-RTX-3090 numbers from Wilson 2024 Table I and rerun only on a single seq to verify, not full reproduce.

### Total v4 risks
- **H = 6** (was 5: + R-18)
- **M = 9** (was 8: + R-19, R-20; − R-12 demoted to L)
- **L = 2** (was 1: + R-12)
- **Total = 17** (was 14)

## §7 Timeline — UNCHANGED

The 36-week schedule survives because cloud absorbs what would otherwise have been a 2-3 week delay from VRAM constraints. Cloud cost ($80) is the resource that buys the time.

W1 milestone (2026-06-04) unchanged:
> R2-reimpl baseline running end-to-end on SemanticKITTI seq 08, producing at least a (possibly poor) mIoU number.

This runs **on local 5060** with batch 1 fp16. SemKITTI inference (no training) on a single sequence fits in 8 GB easily.

## §9 Go/No-Go — UNCHANGED

All 6 G-criteria are unchanged. The only operational difference: G-4 (W12, M2 loop closure) and G-6 (W20, Khronos baseline) will run on cloud; the trigger thresholds remain identical.

## §10 Open Questions to User (3 new)

1. **Cloud provider preference?** autoDL is cheaper for China-based users; Vast.ai/RunPod are global. Recommend autoDL primary + RunPod backup. Confirm or override.
2. **Storage strategy?** 80 GB SemanticKITTI lives on D:\datasets locally. Cloud needs its own copy. OK to rsync 80 GB once at start (1-time ~1 hr cost over 100 Mbps), or prefer to download from official sources each time (free but slow)?
3. **Hard cost cap?** v4 budgets $80. If you want firmer cap (e.g., $50), I'll cut A-7 ablation from cloud and use published numbers. If softer (e.g., $150), I can re-promote A-6 voxel-size ablation.
