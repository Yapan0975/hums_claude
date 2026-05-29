"""Jetson Orin NX deployment entrypoint (research_plan v3 §4.6).

Loads the trained EvidLife-Map checkpoint, builds the TensorRT FP16 engine for
the M1 head, wires the M3 decay sweep to the GPU, and prints the per-frame
latency table that becomes Table VIII in §IV.

Invocation (matches docker-compose ``evidlife_jetson`` service):

    python3.10 -m evidlife_map.jetson.deploy_orin_nx \\
        --config configs/evidlife_kitti.yaml \\
        --checkpoint /runs/evidlife_kitti_closed/last.ckpt

The body remains a placeholder until the W24 deployment-week run, but the
arg parsing + sanity checks are wired up now so the W24 task is "fill three
TODO functions" rather than "design from scratch".
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy EvidLife-Map on Jetson Orin NX (paper §IV.G)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config (one of configs/evidlife_*.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained M1 head checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--rosbag",
        type=Path,
        default=None,
        help="Optional rosbag to replay; default = live LiDAR topic.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Build the TensorRT engine in FP16 (default: True).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Run duration in seconds (default 30 s — matches demo video).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    # TODO(W24): build TensorRT engine, wire ROS 2 subscribers, start the
    # M3 decay sweep timer, and dump per-frame latency to /runs/jetson_latency.csv.
    print(
        f"[deploy_orin_nx] config={args.config} ckpt={args.checkpoint} "
        f"rosbag={args.rosbag} fp16={args.fp16} duration_s={args.duration_s}"
    )
    print(
        "[deploy_orin_nx] P1 skeleton — actual deployment loop fills in at W24 "
        "(research_plan v3 §7 P5)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
