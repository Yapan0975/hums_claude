#!/usr/bin/env bash
# RQ5 dynamic-scene head-to-head vs Khronos on the SemKITTI dynamic subset.
# R-17 high risk; G-6 (W20) checkpoint decides full-run vs published-numbers fallback.
set -euo pipefail

DYNAMIC_LIST="${1:-experiments/rq5_khronos_dynamic/dynamic_frames.txt}"
CKPT="${2:?usage: eval_rq5_khronos.sh <dynamic_frames.txt> <m1_checkpoint.ckpt>}"

python3.10 -m evidlife_map.eval.rq5_khronos \
    --dynamic-frames "${DYNAMIC_LIST}" \
    --checkpoint "${CKPT}" \
    --output "/runs/rq5_khronos.json"

echo "[eval_rq5_khronos] If Khronos reproduction failed, see"
echo "    risk_log/risk_register_v3.md R-17 mitigation: published-numbers fallback."
