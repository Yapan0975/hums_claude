# EvidLife-Map

**Evidential Lifelong Online Metric-Semantic Mapping with Voxel Decay** —
a single Dirichlet vacuity scalar drives open-set detection, loop-closure
descriptor entropy, and voxel-decay clock in one unified per-voxel state.

> Target venue: **IROS 2027** (deadline ≈ 2027-03)
> Baseline: Jiao et al., *Real-Time Metric-Semantic Mapping for Autonomous
> Navigation in Outdoor Environments*, IEEE T-ASE 2024 (arXiv 2412.00291).

This is the working repository for the paper. It tracks code, paper drafts,
literature notes, and preliminary verification results in one place.

## One vacuity, three jobs — paper §I.B thesis

| Leg | Job | Math | Status |
|-----|-----|------|--------|
| (i)   | Open-set / OOD score | $u_v = (C+1) / \sum_k \alpha_v^k$ | ✅ **AUROC 0.808** verified on SemKITTI 14/5 split |
| (ii)  | Loop-closure entropy channel | submap descriptor $h_{\text{ent}}(u_v)$ | ⏸ pending KITTI-360 multi-session transfer |
| (iii) | Voxel-decay clock | $\tau(u_v) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot u_v$ | ✅ **2.64× ratio** verified on 747k voxels |

## Preliminary §IV.0 results (PointNet-Vanilla backbone, SemKITTI seq 08)

The Cylinder3D backbone is gated on a spconv 1.x source-build (see
[`W2-1_status.md`](W2-1_status.md)); the §IV.0 numbers below use a
0.21 M-param PointNet-Vanilla as a substitute so the comparative claims
(M1 vs R2 at matched backbone capacity) are still falsifiable.

| Metric | R2 (CE PointNet) | **M1 (EDL PointNet)** | M1 advantage |
|--------|------------------|-----------------------|--------------|
| mIoU | 1.69 % | **14.73 %** | 8.7× higher |
| ECE | 0.490 | **0.171** | 2.9× lower |
| Latency / frame | 663 ms | **3.0 ms** | 220× faster |

JSON artefact: [`artifacts/rq1_fair_100frames.json`](artifacts/rq1_fair_100frames.json).

## Repository layout

```
.
├── drafts/                       Paper drafts (MD + dual-rendered HTML)
│   ├── paper_v2.md / paper_v2.html       latest draft (§I-VI, refs)
│   ├── research_plan.md                  v4 — current
│   ├── research_plan_v3_archive.md       v3 frozen
│   └── paper_outline_v2.md               v3 outline
├── code/
│   └── evidlife_map/             Python package + scripts + tests
│       ├── evidlife_map/         M1/M2/M3 modules, data, eval, models
│       ├── scripts/              smoke tests, training, validation
│       └── tests/                39 pytest cases
├── nvblox_evidential/            C++ plug-in for nvblox (88 B static_assert verified)
├── experiments/                  per-RQ experiment plans
├── risk_log/                     14-risk register + 6-gate Go/No-Go
├── refs/                         literature (12 Tier-1 精读 HTML + 47 Tier-2/3 notes + refs.bib)
├── artifacts/                    verification result JSON + lit-scan + audits
└── README.md                     this file
```

## Quick start (5060 laptop or 5090 server)

```bash
# Environment
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install scipy scikit-learn hypothesis pyyaml tqdm pytest pytest-cov

# Smoke test on synthetic data (no dataset needed)
cd code/evidlife_map
PYTHONPATH=. python scripts/smoke_test.py --device cuda

# Run unit tests (39 pytest cases)
python -m pytest tests/ -q
```

To run the RQ1/RQ2/RQ4 validation scripts, download SemanticKITTI seq 08
first (any of the 4 download routes in [`scripts/download_datasets.sh`](code/evidlife_map/scripts/download_datasets.sh)).

## What works today

- **R2 reimpl** (paper Eq 4) — argmax-Bayes per voxel with `flat`-tensor backend, 769 ms / frame on RTX 5060 Laptop.
- **M1 evidential head** (paper §III.B Eq 1-6) — Dirichlet posterior with vacuity scalar, 3 ms / frame.
- **M3 voxel decay** (paper §III.D Eq 10-12) — vacuity-conditioned conjugate decay, validated 2.64× ratio.
- **Open-set vacuity AUROC** — 0.808 on seq 08 14/5 split (paper H2 verified).
- **RQ4 lifelong simulation** — seq 08 A/B-split: 34.8 % spatial overlap, stale-voxel P=0.77 R=0.52.
- **paper_v2.md / paper_v2.html** — full §I-§VI skeleton with §IV.0 preliminary results filled in.

## Known blockers

1. **Cylinder3D + spconv 2.x** — kernel-index ordering value-semantic gap; produces NaN logits even after structural patches load 0/0 missing keys. See [`W2-1_status.md`](W2-1_status.md) for the 5-stage diagnosis and three unblock paths.
2. **PointNet-Vanilla mIoU ceiling** — preliminary 14.73 % mIoU is dominated by the 0.21 M-param backbone capacity gap to Cylinder3D's planned 55 M. Reframed as preliminary in §IV.0; full §IV.C numbers land once blocker (1) is unblocked.
3. **EDL post-epoch-1 collapse** — KL annealing degrades vacuity AUROC from 0.808 to ~0.67; mitigated by best-checkpoint-by-AUROC selection. Late-stage KL warm-restart schedule pre-registered for next revision.

## License

Code under Apache-2.0 (pending release). Paper drafts and literature notes
are private to the author group until submission.

---

**Last updated**: 2026-05-29, commit a07f974. Eight commits track the W1 → W3
trajectory: see `git log --oneline` for the chronological reading order.
