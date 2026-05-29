#!/usr/bin/env bash
# Fine-tune the M1 EDL head on top of a frozen Cylinder3D backbone.
# Usage:  bash scripts/train_m1.sh configs/evidlife_kitti.yaml
#
# W2 deliverable per research_plan v3 §7 P2: 20-D head reaches G-2 (+1 mIoU
# over R2-reimpl on SemKITTI seq 08).
set -euo pipefail

CONFIG="${1:?usage: train_m1.sh <config.yaml>}"
RUN_NAME="$(basename "${CONFIG%.*}")_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/runs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

echo "[train_m1] config=${CONFIG}"
echo "[train_m1] log_dir=${LOG_DIR}"

python3.10 -m evidlife_map.m1_evidential.train \
    --config "${CONFIG}" \
    --output-dir "${LOG_DIR}" \
    "${@:2}"
