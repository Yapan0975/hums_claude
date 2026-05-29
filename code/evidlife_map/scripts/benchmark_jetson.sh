#!/usr/bin/env bash
# C4 deployment benchmark + 30 s demo video on Jetson Orin NX.
# Runs against the rosbag at $ROSBAG; pushes ``evidlife_demo.mp4`` to the
# supplementary directory.
set -euo pipefail

CONFIG="${1:?usage: benchmark_jetson.sh <config.yaml> <m1_checkpoint.ckpt> <rosbag>}"
CKPT="${2:?usage: benchmark_jetson.sh <config.yaml> <m1_checkpoint.ckpt> <rosbag>}"
ROSBAG="${3:?usage: benchmark_jetson.sh <config.yaml> <m1_checkpoint.ckpt> <rosbag>}"

python3.10 -m evidlife_map.jetson.deploy_orin_nx \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --rosbag "${ROSBAG}" \
    --fp16 \
    --duration-s 30.0

python3.10 -m evidlife_map.jetson.record_demo \
    --rosbag "${ROSBAG}" \
    --output /runs/evidlife_demo.mp4 \
    --overlay both \
    --fps 10 \
    --duration-s 30.0

echo "[benchmark_jetson] /runs/jetson_latency.csv + /runs/evidlife_demo.mp4 produced"
