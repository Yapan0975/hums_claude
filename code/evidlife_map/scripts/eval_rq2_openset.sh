#!/usr/bin/env bash
# RQ2 lead: vacuity-as-OOD AUROC + AUPR on SemKITTI 14/5 open-set split,
# plus the nuScenes reverse cross-domain check.
# G-5 deliverable (W17): AUROC >= 0.80 AND AUPR >= 0.60.
set -euo pipefail

CKPT="${1:?usage: eval_rq2_openset.sh <m1_openset_checkpoint.ckpt>}"
SPLIT="${2:-primary}"   # primary | robustness

python3.10 -m evidlife_map.eval.rq2_openset \
    --checkpoint "${CKPT}" \
    --split "${SPLIT}" \
    --output "/runs/rq2_${SPLIT}.json"

python3.10 -m evidlife_map.eval.rq2_nuscenes_reverse \
    --checkpoint "${CKPT}" \
    --output "/runs/rq2_nuscenes_reverse.json"
