"""30 s demo capture for the IROS supplementary video (C4).

Produces ``evidlife_demo.mp4`` with the traversability overlay (RQ3 output)
and a vacuity heat-map overlay. Captured at W24 (research_plan v3 §4.6).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the 30-second IROS supplementary demo video."
    )
    parser.add_argument("--rosbag", type=Path, required=True, help="Input rosbag to replay.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidlife_demo.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--overlay",
        choices=["traversability", "vacuity", "both"],
        default="both",
        help="Visual overlay style.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Recording frame rate; default 10 Hz to match SemKITTI native.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Recording duration in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.rosbag.exists():
        raise SystemExit(f"rosbag not found: {args.rosbag}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # TODO(W24-W25): subscribe to rosbag topics, render via VoxelMapViewer,
    # encode to MP4 with ffmpeg via subprocess.
    print(
        f"[record_demo] rosbag={args.rosbag} -> output={args.output} "
        f"overlay={args.overlay} fps={args.fps} duration_s={args.duration_s}"
    )
    print("[record_demo] P1 skeleton — actual capture loop fills in at W24/W25.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
