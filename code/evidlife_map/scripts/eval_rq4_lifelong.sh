#!/usr/bin/env bash
# RQ4 lifelong: stale-voxel removal + inter-session ECE drift on KITTI-360.
# Consumes the revisit-pair JSON delivered at G-3 (W3).
set -euo pipefail

REVISIT_LIST="${1:-configs/kitti360_revisit_pairs.json}"
CKPT="${2:?usage: eval_rq4_lifelong.sh <revisit_pairs.json> <m1_checkpoint.ckpt>}"

python3.10 -m evidlife_map.eval.rq4_lifelong \
    --revisit-pairs "${REVISIT_LIST}" \
    --checkpoint "${CKPT}" \
    --output "/runs/rq4_lifelong.json"
