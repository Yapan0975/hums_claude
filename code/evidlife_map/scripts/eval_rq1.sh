#!/usr/bin/env bash
# RQ1 main result: closed-set mIoU + ECE on SemKITTI seq 08 + 11-21.
# Six systems × one harness; produces paper Table III + Table I.
set -euo pipefail

CONFIG="${1:-configs/evidlife_kitti.yaml}"
CKPT="${2:?usage: eval_rq1.sh <config.yaml> <m1_checkpoint.ckpt>}"
SEQS="${3:-08,11,12,13,14,15,16,17,18,19,20,21}"

for system in evidlife_map r2_reimpl convbki sbki kimera_semantics nvblox_bare; do
    echo "[eval_rq1] system=${system}"
    python3.10 -m evidlife_map.eval.rq1 \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --system "${system}" \
        --sequences "${SEQS}" \
        --output "/runs/rq1_${system}.json"
done
