#!/usr/bin/env python
"""W3-X KITTI-360 revisit-pair M2 + M3 verification — Path-to-Accept condition 5.

Closes leg (ii) of the §I.B "one Dirichlet posterior, two scalars, three
jobs" thesis on real multi-session data, addressing round-1 R-EIC W2.

KITTI-360 `data_3d_semantics/train/<drive>/static/*.ply` provides 630
per-window accumulated point clouds in WORLD FRAME (no separate poses
needed). Each PLY contains:
  - xyz (m)
  - RGB
  - semantic ID (KITTI-360 19-class semantics)
  - instance ID
  - confidence (from KITTI-360 annotation pipeline)

For multi-session revisit, drive 00 has multiple traversals of overlapping
Karlsruhe areas; revisit pairs are windows from the same drive with
spatial overlap > threshold (no temporal contiguity).

Workflow:
  1. Scan all `.ply` files; load metadata (centroid + bbox).
  2. Find revisit pairs: (file_i, file_j) where bbox overlap > threshold.
  3. For each revisit pair, compute:
       a. M2 descriptor similarity (vacuity/dissonance/softmax-entropy channels)
          using *synthetic* Dirichlet posterior derived from the PLY semantic
          labels + confidence (since we don't run our trained EDL head here)
       b. M3 stale-voxel detection: which voxels in window-A also appear in
          window-B; flag stale by post-decay scalar threshold.
  4. Aggregate metrics across all revisit pairs; report mean ± std.

Output: `artifacts/w3x_kitti360_revisit.json` with per-pair + aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_ply_header(path: Path) -> dict:
    """Extract # vertices, property names from ASCII or binary PLY header."""
    info = {"n_vertices": 0, "properties": [], "fmt": "unknown"}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("format"):
                info["fmt"] = line.split()[1]
            elif line.startswith("element vertex"):
                info["n_vertices"] = int(line.split()[-1])
            elif line.startswith("property"):
                # e.g. "property float x"
                parts = line.split()
                info["properties"].append((parts[1], parts[-1]))
            elif line == "end_header":
                info["header_size"] = f.tell()
                break
    return info


_DTYPE_MAP = {
    "float": np.float32, "float32": np.float32, "float64": np.float64,
    "double": np.float64,
    "int": np.int32, "int32": np.int32, "int64": np.int64,
    "uchar": np.uint8, "uint8": np.uint8,
    "ushort": np.uint16, "uint16": np.uint16,
    "short": np.int16, "int16": np.int16,
    "uint": np.uint32, "uint32": np.uint32,
}


def load_ply(path: Path, verbose: bool = False) -> dict:
    """Load a binary PLY file. Returns dict of named numpy arrays."""
    info = parse_ply_header(path)
    n = info["n_vertices"]
    dtype = np.dtype([
        (name, _DTYPE_MAP[ty])
        for ty, name in info["properties"]
    ])
    with open(path, "rb") as f:
        f.seek(info["header_size"])
        if info["fmt"].startswith("binary_little_endian"):
            arr = np.fromfile(f, dtype=dtype, count=n)
        elif info["fmt"].startswith("binary_big_endian"):
            arr = np.fromfile(f, dtype=dtype.newbyteorder(">"), count=n)
        else:
            raise NotImplementedError("ASCII PLY not supported")
    return {name: arr[name] for name, _ in dtype.descr}


def scan_metadata(ply_dir: Path, sample_step: int = 1) -> list[dict]:
    """For each .ply in ply_dir, load only header + sample of points to get
    metadata (file, centroid, bbox)."""
    out = []
    files = sorted(ply_dir.glob("*.ply"))
    print(f"  scanning {len(files)} PLY files...", flush=True)
    for i, p in enumerate(files):
        if sample_step > 1 and i % sample_step != 0:
            continue
        try:
            d = load_ply(p)
            xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1)
            cen = xyz.mean(axis=0)
            mn = xyz.min(axis=0)
            mx = xyz.max(axis=0)
            n = xyz.shape[0]
            # Extract window indices from filename: 0000000002_0000000385.ply
            stem = p.stem  # e.g. "0000000002_0000000385"
            parts = stem.split("_")
            wstart = int(parts[0]); wend = int(parts[1])
            out.append({
                "file": p.name, "path": str(p),
                "n_points": int(n),
                "centroid": cen.tolist(),
                "bbox_min": mn.tolist(), "bbox_max": mx.tolist(),
                "win_start": wstart, "win_end": wend,
            })
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return out


def bbox_overlap_volume(a: dict, b: dict) -> float:
    mn_a = np.array(a["bbox_min"]); mx_a = np.array(a["bbox_max"])
    mn_b = np.array(b["bbox_min"]); mx_b = np.array(b["bbox_max"])
    overlap_mn = np.maximum(mn_a, mn_b)
    overlap_mx = np.minimum(mx_a, mx_b)
    if (overlap_mx <= overlap_mn).any():
        return 0.0
    vol_overlap = float(np.prod(overlap_mx - overlap_mn))
    return vol_overlap


def find_revisit_pairs(metas: list[dict], min_overlap_m3: float = 100.0,
                       min_temporal_gap_frames: int = 200) -> list[tuple[int, int, float]]:
    """Return pairs (i, j, overlap_vol_m3) where:
       - bbox overlap volume > min_overlap_m3
       - window frame ranges are non-adjacent (gap > min_temporal_gap_frames)
    """
    pairs = []
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            mi, mj = metas[i], metas[j]
            # Skip temporally adjacent windows
            if abs(mi["win_start"] - mj["win_end"]) < min_temporal_gap_frames and \
               abs(mj["win_start"] - mi["win_end"]) < min_temporal_gap_frames:
                continue
            ov = bbox_overlap_volume(mi, mj)
            if ov > min_overlap_m3:
                pairs.append((i, j, ov))
    return pairs


# ── Synthetic Dirichlet posterior from KITTI-360 semantic labels ───────────
# Since we don't run our trained EDL head on KITTI-360 (no Cylinder3D
# unblock yet), we derive a synthetic Dirichlet posterior using:
#   alpha_v[gt_class] = 1 + κ * confidence
#   alpha_v[other] = 1
# This is consistent with the M1 evidential semantics: confidence-weighted
# evidence on the labelled class, uniform prior elsewhere.

def synth_dirichlet_per_voxel(xyz: np.ndarray, sem: np.ndarray, conf: np.ndarray,
                                num_classes: int = 19, voxel_size: float = 0.25,
                                kappa: float = 10.0,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate points to voxels and build per-voxel Dirichlet alpha.

    Returns (voxel_keys, alpha[V, C+1]) where C+1 includes the unknown channel.
    """
    # Voxel hash: floor(xyz / voxel_size) -> 3D int -> 1D key
    v = np.floor(xyz / voxel_size).astype(np.int64)
    keys = v[:, 0] * 1_000_003 + v[:, 1] * 1_009 + v[:, 2]
    # Per voxel, sum evidence weighted by confidence
    unique_keys, inv = np.unique(keys, return_inverse=True)
    V = unique_keys.shape[0]
    C_plus_1 = num_classes + 1
    alpha = np.ones((V, C_plus_1), dtype=np.float32)
    # Add kappa * conf to alpha[voxel, sem_class]
    # Skip points with invalid sem (out of range)
    valid = (sem >= 0) & (sem < num_classes)
    sem_v = sem[valid]
    conf_v = conf[valid] if conf is not None else np.ones_like(sem_v, dtype=np.float32)
    inv_v = inv[valid]
    # Use np.add.at for unbuffered scatter-add
    np.add.at(alpha, (inv_v, sem_v), kappa * conf_v.astype(np.float32))
    return unique_keys, alpha


def vacuity(alpha: np.ndarray) -> np.ndarray:
    return alpha.shape[-1] / alpha.sum(axis=-1)


def dissonance(alpha: np.ndarray) -> np.ndarray:
    """Sensoy 2018 dissonance approximation: 1 - p_top1 = mass not in winning class."""
    p = alpha / alpha.sum(axis=-1, keepdims=True).clip(min=1e-9)
    p_max = p.max(axis=-1)
    return 1.0 - p_max


def softmax_entropy(alpha: np.ndarray) -> np.ndarray:
    p = alpha / alpha.sum(axis=-1, keepdims=True).clip(min=1e-9)
    h = -(p * np.log(p.clip(min=1e-9))).sum(axis=-1)
    return h / np.log(alpha.shape[-1])  # normalise to [0, 1]


def histogram_descriptor(scalar: np.ndarray, n_bins: int = 16) -> np.ndarray:
    hist, _ = np.histogram(np.clip(scalar, 0.0, 1.0), bins=n_bins, range=(0.0, 1.0))
    s = hist.sum()
    return hist.astype(np.float64) / max(1, s)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = (a * b).sum()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(n / max(1e-9, d))


def m3_stale_pr(keys_a, alpha_a, keys_b, *, scalar_fn, threshold: float):
    scalar = scalar_fn(alpha_a)
    flagged_stale = scalar > threshold
    # Which keys in A are present in B?
    keys_b_sorted = np.sort(keys_b)
    overlap = np.isin(keys_a, keys_b_sorted)
    a_only = ~overlap
    n_flagged = int(flagged_stale.sum())
    n_correctly_flagged = int((flagged_stale & a_only).sum())
    n_a_only = int(a_only.sum())
    p = n_correctly_flagged / n_flagged if n_flagged > 0 else float("nan")
    r = n_correctly_flagged / n_a_only if n_a_only > 0 else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p) and (r == r) and (p + r) > 0 else float("nan")
    return {"threshold": threshold, "precision": p, "recall": r, "f1": f1,
            "n_flagged": n_flagged, "n_a_only": n_a_only, "n_overlap": int(overlap.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply-dir", type=Path, required=True,
                    help="e.g. .../data_3d_semantics/train/2013_05_28_drive_0000_sync/static/")
    ap.add_argument("--voxel-size", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=10.0,
                    help="Synthetic evidence weight per labelled point")
    ap.add_argument("--min-overlap-m3", type=float, default=100.0,
                    help="Min bbox overlap (m^3) to count as a revisit pair")
    ap.add_argument("--min-gap-frames", type=int, default=200,
                    help="Min temporal gap (frames) between revisit windows")
    ap.add_argument("--max-pairs", type=int, default=10,
                    help="Cap on number of revisit pairs to evaluate")
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/w3x_kitti360_revisit.json"))
    args = ap.parse_args()

    print(f"=== W3-X KITTI-360 revisit-pair M2 + M3 verification ===")
    print(f"PLY dir: {args.ply_dir}")
    print(f"voxel_size: {args.voxel_size} m, kappa: {args.kappa}")
    print()

    t0 = time.perf_counter()
    metas = scan_metadata(args.ply_dir, sample_step=1)
    print(f"  {len(metas)} PLY files indexed in {time.perf_counter() - t0:.1f}s")

    print(f"Finding revisit pairs (overlap > {args.min_overlap_m3} m^3, gap > {args.min_gap_frames} fr)...")
    pairs = find_revisit_pairs(metas, args.min_overlap_m3, args.min_gap_frames)
    pairs.sort(key=lambda t: -t[2])
    print(f"  found {len(pairs)} revisit pairs (top {args.max_pairs} kept)")
    pairs = pairs[:args.max_pairs]

    all_m2 = {"vacuity": [], "dissonance": [], "softmax_entropy": []}
    all_m3 = {"vacuity": [], "dissonance": []}

    for k, (i, j, ov) in enumerate(pairs):
        mi, mj = metas[i], metas[j]
        print(f"\n--- Pair {k+1}/{len(pairs)}: {mi['file']} ↔ {mj['file']} (overlap={ov:.0f} m^3) ---")
        d_a = load_ply(Path(mi["path"]))
        d_b = load_ply(Path(mj["path"]))
        xyz_a = np.stack([d_a["x"], d_a["y"], d_a["z"]], axis=-1)
        xyz_b = np.stack([d_b["x"], d_b["y"], d_b["z"]], axis=-1)
        sem_a = d_a.get("semantic") if "semantic" in d_a else d_a.get("label", np.zeros(xyz_a.shape[0], dtype=np.int32))
        sem_b = d_b.get("semantic") if "semantic" in d_b else d_b.get("label", np.zeros(xyz_b.shape[0], dtype=np.int32))
        conf_a = d_a.get("confidence", np.ones(xyz_a.shape[0], dtype=np.float32))
        conf_b = d_b.get("confidence", np.ones(xyz_b.shape[0], dtype=np.float32))
        # KITTI-360 semantic IDs are raw; map to 19-class
        # For W3-X preliminary we just clip to [0, 18]
        sem_a = np.clip(sem_a.astype(np.int32) - 1, 0, 18)  # subtract 1 to drop "unlabelled=0"
        sem_b = np.clip(sem_b.astype(np.int32) - 1, 0, 18)
        # Ensure conf in [0, 1]
        if conf_a.dtype == np.uint8:
            conf_a = conf_a.astype(np.float32) / 255.0
            conf_b = conf_b.astype(np.float32) / 255.0
        keys_a, alpha_a = synth_dirichlet_per_voxel(xyz_a, sem_a, conf_a,
                                                      voxel_size=args.voxel_size,
                                                      kappa=args.kappa)
        keys_b, alpha_b = synth_dirichlet_per_voxel(xyz_b, sem_b, conf_b,
                                                      voxel_size=args.voxel_size,
                                                      kappa=args.kappa)
        print(f"  voxels A: {alpha_a.shape[0]}, voxels B: {alpha_b.shape[0]}")

        # M2 descriptors
        for ch, fn in [("vacuity", vacuity), ("dissonance", dissonance),
                       ("softmax_entropy", softmax_entropy)]:
            s_a = fn(alpha_a); s_b = fn(alpha_b)
            h_a = histogram_descriptor(s_a); h_b = histogram_descriptor(s_b)
            sim = cosine_sim(h_a, h_b)
            all_m2[ch].append(sim)
            print(f"  M2 [{ch}]: cosine sim = {sim:.4f}")

        # M3 stale-voxel detection (vacuity-vs-dissonance threshold scalar)
        for ch, fn in [("vacuity", vacuity), ("dissonance", dissonance)]:
            r = m3_stale_pr(keys_a, alpha_a, keys_b, scalar_fn=fn, threshold=0.5)
            all_m3[ch].append(r)
            print(f"  M3 [{ch}] @ thr=0.5: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    # Aggregate
    print("\n=== Aggregate across {} revisit pairs ===".format(len(pairs)))
    summary = {"n_pairs": len(pairs), "M2": {}, "M3": {}}
    for ch in all_m2:
        sims = np.array(all_m2[ch])
        summary["M2"][ch] = {"mean": float(sims.mean()), "std": float(sims.std()),
                              "min": float(sims.min()), "max": float(sims.max())}
        print(f"  M2 [{ch}]: mean cosine sim = {sims.mean():.4f} ± {sims.std():.4f}")
    for ch in all_m3:
        f1s = np.array([r["f1"] for r in all_m3[ch] if r["f1"] == r["f1"]])
        ps = np.array([r["precision"] for r in all_m3[ch] if r["precision"] == r["precision"]])
        rs = np.array([r["recall"] for r in all_m3[ch] if r["recall"] == r["recall"]])
        summary["M3"][ch] = {"f1_mean": float(f1s.mean()) if f1s.size else float("nan"),
                              "f1_std": float(f1s.std()) if f1s.size else float("nan"),
                              "p_mean": float(ps.mean()) if ps.size else float("nan"),
                              "r_mean": float(rs.mean()) if rs.size else float("nan"),
                              "per_pair": all_m3[ch]}
        print(f"  M3 [{ch}]: mean F1 = {f1s.mean():.4f} (P={ps.mean():.4f}, R={rs.mean():.4f})")

    out = {
        "kitti360_dir": str(args.ply_dir),
        "voxel_size_m": args.voxel_size, "kappa": args.kappa,
        "min_overlap_m3": args.min_overlap_m3, "min_gap_frames": args.min_gap_frames,
        "n_pairs_total_found": len(pairs),
        "summary": summary,
        "pairs_metadata": [
            {"i": i, "j": j, "overlap_m3": ov,
             "file_a": metas[i]["file"], "file_b": metas[j]["file"],
             "n_points_a": metas[i]["n_points"], "n_points_b": metas[j]["n_points"]}
            for (i, j, ov) in pairs
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
