#!/usr/bin/env python
"""Build the KITTI-360 revisit matrix (research_plan v3 §4.1 D-c + G-3).

Identifies frame pairs (i, j) where the ego trajectory revisits an earlier
location, defined as: there exists a contiguous sub-trajectory of length
``min_overlap_m`` whose Euclidean ground-projection distance to a contiguous
sub-trajectory at frame j is ≤ ``threshold_m``.

This is the W3 G-3 deliverable. Without it, RQ4 lifelong cannot start because
the loader at ``data/kitti_360.py`` expects this JSON to know which frame
pairs to fuse.

Usage::

    python -m scripts.build_revisit_matrix \
        --kitti360-root D:/datasets/KITTI-360 \
        --sequences 2013_05_28_drive_0000_sync 2013_05_28_drive_0004_sync \
        --min-overlap-m 30.0 \
        --threshold-m 2.0 \
        --out artifacts/revisit_matrix.json

Output JSON schema::

    {
        "version": 1,
        "sequences": ["2013_05_28_drive_0000_sync", ...],
        "min_overlap_m": 30.0,
        "threshold_m": 2.0,
        "pairs": [
            {
                "seq_a": "2013_05_28_drive_0000_sync",
                "frame_a_start": 1024,
                "frame_a_end":   1090,
                "seq_b": "2013_05_28_drive_0000_sync",
                "frame_b_start": 5210,
                "frame_b_end":   5285,
                "overlap_m": 35.4,
                "max_dist_m":  1.84
            },
            ...
        ]
    }
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class RevisitPair:
    seq_a: str
    frame_a_start: int
    frame_a_end: int
    seq_b: str
    frame_b_start: int
    frame_b_end: int
    overlap_m: float
    max_dist_m: float


def load_kitti360_poses(root: Path, sequence: str) -> np.ndarray:
    """Load the KITTI-360 pose file for a given sequence.

    Returns a ``(N, 3)`` array of XYZ world-frame positions.

    KITTI-360 poses live at ``root/data_poses/<sequence>/poses.txt`` with rows
    of the form ``<frame_idx> r00 r01 r02 t0 r10 ... r22 t2`` (frame_idx +
    12 floats). The translation is the camera-frame world position.
    """
    pose_path = root / "data_poses" / sequence / "poses.txt"
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing poses.txt: {pose_path}")
    rows = np.loadtxt(pose_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 13:
        raise ValueError(f"Expected 13 cols (idx + 12 floats) in {pose_path}; got {rows.shape}")
    # Translation = rows[:, [4, 8, 12]] (the 4th col of each 3x4 row).
    xyz = rows[:, [4, 8, 12]]
    return xyz.astype(np.float64)


def cumulative_arclength(xyz: np.ndarray) -> np.ndarray:
    """Per-frame cumulative arclength along the ego trajectory."""
    if xyz.shape[0] < 2:
        return np.zeros(xyz.shape[0], dtype=np.float64)
    deltas = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(deltas)])


def find_revisit_pairs(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    *,
    threshold_m: float,
    min_overlap_m: float,
    min_temporal_gap: int,
    seq_a: str,
    seq_b: str,
) -> list[RevisitPair]:
    """Find revisit pairs between two trajectories.

    Algorithm (O(N²) but trivially correct):
      1. For every pair (i, j) compute distance.
      2. Mark close pairs (dist ≤ threshold_m).
      3. Group contiguous (i, j) runs where the offset i-j stays bounded.
      4. Each run with arclength ≥ min_overlap_m becomes a RevisitPair.
    """
    if xyz_a.size == 0 or xyz_b.size == 0:
        return []

    cum_a = cumulative_arclength(xyz_a)
    cum_b = cumulative_arclength(xyz_b)

    # Pairwise distances (vectorised; can be chunked for very long sequences).
    # For 10k x 10k frames this is 800 MB — chunk to 2k rows at a time.
    pairs: list[RevisitPair] = []
    chunk = 2048
    n_a = xyz_a.shape[0]
    n_b = xyz_b.shape[0]
    same_seq = (seq_a == seq_b)

    # Track currently-open runs per i row to allow stitching across chunks.
    open_runs: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for i0 in range(0, n_a, chunk):
        i1 = min(i0 + chunk, n_a)
        block = xyz_a[i0:i1, None, :] - xyz_b[None, :, :]
        dists = np.linalg.norm(block, axis=2)  # (chunk, n_b)
        close = dists <= threshold_m
        if same_seq:
            # Skip near-diagonal trivially-close pairs.
            ii, jj = np.indices(close.shape)
            ii_abs = ii + i0
            close = close & (np.abs(ii_abs - jj) >= min_temporal_gap)

        # Scan each j column for contiguous i-runs.
        for j in range(n_b):
            col = close[:, j]
            if not col.any():
                continue
            # Find contiguous True runs along i.
            i_indices = np.nonzero(col)[0] + i0
            runs = _contiguous_runs(i_indices.tolist())
            for (i_start, i_end) in runs:
                overlap = cum_a[i_end] - cum_a[i_start]
                if overlap < min_overlap_m:
                    continue
                # Estimate matching j-window by tracking nearest j over [i_start, i_end].
                j_best = _nearest_j_window(xyz_a[i_start:i_end + 1], xyz_b, threshold_m)
                if j_best is None:
                    continue
                j_start, j_end = j_best
                j_overlap = cum_b[j_end] - cum_b[j_start]
                if j_overlap < min_overlap_m * 0.5:  # tolerant on the b side
                    continue
                # Max dist over the matched window pair.
                a_slice = xyz_a[i_start:i_end + 1]
                b_slice = xyz_b[j_start:j_end + 1]
                # Resample to common length for cheap max-dist.
                m = min(a_slice.shape[0], b_slice.shape[0])
                a_re = a_slice[np.linspace(0, a_slice.shape[0] - 1, m, dtype=int)]
                b_re = b_slice[np.linspace(0, b_slice.shape[0] - 1, m, dtype=int)]
                max_dist = float(np.linalg.norm(a_re - b_re, axis=1).max())
                pairs.append(RevisitPair(
                    seq_a=seq_a, frame_a_start=int(i_start), frame_a_end=int(i_end),
                    seq_b=seq_b, frame_b_start=int(j_start), frame_b_end=int(j_end),
                    overlap_m=float(overlap), max_dist_m=max_dist,
                ))
            # Break to next j to avoid duplicate (we re-detect at next chunk).
        del block, dists, close

    return _dedup_pairs(pairs)


def _contiguous_runs(sorted_ints: list[int]) -> list[tuple[int, int]]:
    """Group sorted ints into contiguous runs returning [(start, end), ...]."""
    if not sorted_ints:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = sorted_ints[0]
    for x in sorted_ints[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs


def _nearest_j_window(
    a_slice: np.ndarray, xyz_b: np.ndarray, threshold_m: float,
) -> tuple[int, int] | None:
    """For each frame in a_slice find the nearest j in xyz_b; return tightest j window."""
    # KD-tree would be O(N log N); for the W3 deliverable plain numpy is fine.
    js: list[int] = []
    for p in a_slice:
        d = np.linalg.norm(xyz_b - p, axis=1)
        j = int(np.argmin(d))
        if d[j] <= threshold_m * 2.0:  # tolerant
            js.append(j)
    if not js:
        return None
    return min(js), max(js)


def _dedup_pairs(pairs: Iterable[RevisitPair]) -> list[RevisitPair]:
    """Drop pairs whose (i, j) bounding boxes are subsumed by a larger pair."""
    sorted_p = sorted(pairs, key=lambda r: -(r.frame_a_end - r.frame_a_start))
    kept: list[RevisitPair] = []
    for p in sorted_p:
        subsumed = False
        for k in kept:
            if (
                p.seq_a == k.seq_a and p.seq_b == k.seq_b
                and p.frame_a_start >= k.frame_a_start and p.frame_a_end <= k.frame_a_end
                and p.frame_b_start >= k.frame_b_start and p.frame_b_end <= k.frame_b_end
            ):
                subsumed = True
                break
        if not subsumed:
            kept.append(p)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kitti360-root", type=Path, required=True,
                    help="KITTI-360 root directory (containing data_poses/).")
    ap.add_argument("--sequences", nargs="+", required=True,
                    help="Sequence names, e.g. 2013_05_28_drive_0000_sync.")
    ap.add_argument("--min-overlap-m", type=float, default=30.0,
                    help="Minimum sustained overlap in metres (default: 30.0).")
    ap.add_argument("--threshold-m", type=float, default=2.0,
                    help="Distance threshold for 'same place' in metres (default: 2.0).")
    ap.add_argument("--min-temporal-gap", type=int, default=200,
                    help="Minimum frame index gap (same sequence; default: 200).")
    ap.add_argument("--out", type=Path, default=Path("artifacts/revisit_matrix.json"),
                    help="Output JSON path.")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load all sequences once.
    print(f"[build_revisit_matrix] loading {len(args.sequences)} sequence(s)...")
    pose_by_seq: dict[str, np.ndarray] = {
        seq: load_kitti360_poses(args.kitti360_root, seq) for seq in args.sequences
    }
    for seq, xyz in pose_by_seq.items():
        print(f"  {seq}: {xyz.shape[0]} frames, "
              f"trajectory length {cumulative_arclength(xyz)[-1]:.1f} m")

    # All sequence pairs (incl. self-loops).
    all_pairs: list[RevisitPair] = []
    for i, seq_a in enumerate(args.sequences):
        for seq_b in args.sequences[i:]:
            print(f"[build_revisit_matrix] matching {seq_a} <-> {seq_b}...")
            pairs = find_revisit_pairs(
                pose_by_seq[seq_a], pose_by_seq[seq_b],
                threshold_m=args.threshold_m,
                min_overlap_m=args.min_overlap_m,
                min_temporal_gap=args.min_temporal_gap,
                seq_a=seq_a, seq_b=seq_b,
            )
            print(f"  found {len(pairs)} revisit pair(s)")
            all_pairs.extend(pairs)

    out = {
        "version": 1,
        "sequences": list(args.sequences),
        "min_overlap_m": float(args.min_overlap_m),
        "threshold_m": float(args.threshold_m),
        "pairs": [p.__dict__ for p in all_pairs],
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[build_revisit_matrix] wrote {len(all_pairs)} pair(s) to {args.out}")
    # G-3 success gate (research_plan v3 §9): ≥ 5 revisit pairs with ≥ 30 m overlap.
    g3_pass = len(all_pairs) >= 5
    print(f"[G-3] {len(all_pairs)} pair(s) ≥ 30 m overlap — "
          f"{'PASS' if g3_pass else 'FAIL (consider synthetic-revisit injection per R-13)'}")
    return 0 if g3_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
