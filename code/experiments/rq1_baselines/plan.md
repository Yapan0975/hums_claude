# RQ1 — Closed-set mIoU + ECE main result

## Hypothesis (research_plan v3 §2 H1)

EDL-Dirichlet update yields **≥ +2 mIoU** and **≥ −20 % ECE** vs R2-reimpl
argmax-Bayes baseline at equal voxel size (0.25 m) and equal compute, on
SemanticKITTI sequences 08, 11–21.

## Run grid

| Sequence | EvidLife-Map | R2-reimpl | nvblox-bare | ConvBKI | S-BKI | Kimera-Sem |
|----------|--------------|-----------|-------------|---------|-------|------------|
| 08       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 11       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 12       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 13       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 14       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 15       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 16       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 17       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 18       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 19       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 20       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |
| 21       | ✓            | ✓         | ✓           | ✓       | ✓     | ✓          |

12 sequences × 6 systems = 72 runs. Estimated compute: 12 GPU-days
(research_plan v3 §4.5).

## Seeds & determinism

`{20260528, 20260529, 20260530}`; report mean ± std across the three
seeds.

## Execution

```
bash ../../evidlife_map/scripts/eval_rq1.sh \
     ../../evidlife_map/configs/evidlife_kitti.yaml \
     /runs/evidlife_kitti_closed/last.ckpt
```

## Output schema

Each `runs/rq1_<system>.json`:

```json
{
  "system": "evidlife_map",
  "config": "configs/evidlife_kitti.yaml",
  "sequences": ["08", "11", ...],
  "per_sequence": {
    "08": { "mIoU": 0.0, "ECE": 0.0, "latency_ms_mean": 0.0,
            "latency_ms_p99": 0.0, "per_class_iou": [0.0, ...] }
  },
  "macro": { "mIoU": 0.0, "ECE": 0.0 },
  "seed": 20260528
}
```

`parse_results.py` aggregates the JSONs into the Table III LaTeX cell values.
