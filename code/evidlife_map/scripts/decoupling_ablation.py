#!/usr/bin/env python
"""W3-UVW Decoupling Ablation — tests the §I.B "one vacuity, three jobs" thesis.

Per R4 Devil's Advocate adjudication demand and R3 W1 dissonance-vs-vacuity:

Config 1: VACUITY-EVERYWHERE (paper's current; baseline)
  - M1 trained with EDL using u_v for OOD
  - M3 decay rate τ = f(u_v) per Eq. 11
  - M3 "stale" threshold on u_v
  - M2 descriptor entropy channel = vacuity histogram

Config 2: DISSONANCE-FOR-DECAY
  - M1: same EDL head (u_v still for OOD)
  - M3 decay rate τ = f(dissonance) (Sensoy 2018 §4)
  - M3 "stale" threshold on dissonance
  - M2: unchanged (vacuity histogram)

Config 3: SOFTMAX-ENTROPY-FOR-M2
  - M1: same EDL head (u_v still for OOD)
  - M3: unchanged (vacuity-driven decay)
  - M2 descriptor entropy channel = softmax-entropy histogram instead of vacuity

PASS CONDITION for §I.B integration thesis (R4):
  Config 1 must strictly Pareto-dominate Configs 2 and 3 on all reported
  metrics (M3 stale-voxel F1, M2 same-area similarity).

If config 1 does NOT strictly Pareto-dominate, §I.B must be reframed per
the round-1 editorial decision: "one vacuity is sufficient but not
uniquely load-bearing — system-engineering convenience, not scientific
contribution."

All three configs reuse the W3-M ckpt (PointNet2Lite_v2 + warm-restart
+ 600 fr/seq, best epoch 21, val_miou=0.2336). No retraining required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evidlife_map.data.semantic_kitti import SemanticKITTIDataset
from evidlife_map.m1_evidential.edl_head import vacuity_from_alpha
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator
from evidlife_map.m3_decay.conjugate_decay import decay_concentration
from scripts.train_pointnet2lite_v2_edl_multiseq import PointNetLiteV2EDL
from scripts.validate_rq4_lifelong_sim import _voxel_hash, build_session


def dissonance_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Sensoy 2018 §4 dissonance (Eq 11 in their paper) on Dirichlet α.

    diss(α) = Σ_k [ b_k · Σ_{j≠k} b_j · Bal(b_j, b_k) ] / max(1e-9, Σ_{j≠k} b_j)

    where b_k = (α_k - 1) / S (the belief, since α_k - 1 = e_k evidence)
    and Bal(b_i, b_j) = 1 - |b_i - b_j| / (b_i + b_j) if b_i + b_j > 0 else 0.

    Operates on the last dim; returns shape alpha.shape[:-1].

    Higher dissonance = evidence split among multiple competing classes.
    Lower dissonance = evidence concentrated on a single class (or no
    evidence at all, in which case vacuity is the relevant signal).
    """
    e = (alpha - 1.0).clamp_min(0.0)
    S = e.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    b = e / S  # belief (shares of evidence per class)
    # b_i + b_j and |b_i - b_j| pairwise
    b_i = b.unsqueeze(-1)  # (..., C, 1)
    b_j = b.unsqueeze(-2)  # (..., 1, C)
    pair_sum = b_i + b_j
    pair_diff = (b_i - b_j).abs()
    bal = torch.where(
        pair_sum > 1e-9,
        1.0 - pair_diff / pair_sum.clamp_min(1e-9),
        torch.zeros_like(pair_sum),
    )
    # Mask diagonal (i == j contributions removed)
    C = alpha.shape[-1]
    eye = torch.eye(C, dtype=torch.bool, device=alpha.device)
    bal = bal.masked_fill(eye, 0.0)
    # Σ_{j≠k} b_j · Bal(b_j, b_k) per k
    weighted = b_j * bal  # (..., C, C); element (k, j) = b_j * bal(b_j, b_k)
    sum_weighted_per_k = weighted.sum(dim=-1)  # sum over j, per k
    # Σ_{j≠k} b_j per k (denominator)
    sum_b_other = b.sum(dim=-1, keepdim=True) - b  # (..., C)
    sum_b_other = sum_b_other.clamp_min(1e-9)
    # Per-k contribution: b_k · sum_weighted_per_k / sum_b_other_per_k
    per_k = b * sum_weighted_per_k / sum_b_other
    return per_k.sum(dim=-1)


def softmax_entropy_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Entropy of the Dirichlet expected categorical p_k = α_k / S.

    Last-dim entropy, returns shape alpha.shape[:-1].

    Used as the Config 3 substitute for vacuity in the M2 descriptor
    entropy channel: H[p] vs h_vac in Eq. 7.
    """
    p = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    h = -(p * (p.clamp_min(1e-9)).log()).sum(dim=-1)
    return h


def stale_voxel_pr_with_scalar(
    acc_a: EvidenceAccumulator,
    acc_b: EvidenceAccumulator,
    *,
    scalar_fn,
    threshold: float,
    use_scalar_for_tau: bool,
    inter_session_gap_s: float = 600.0,
) -> dict:
    """Compute stale-voxel P/R using a configurable scalar.

    Parameters
    ----------
    scalar_fn : callable
        Takes alpha tensor (N, C+1) → returns (N,) scalar to threshold.
        E.g. vacuity_from_alpha or dissonance_from_alpha.
    threshold : float
        Voxels with scalar > threshold are "flagged stale."
    use_scalar_for_tau : bool
        If True, M3 decay rate τ uses this scalar (replacing vacuity in Eq 11).
        If False, decay still uses vacuity (control: only threshold changed).
    """
    alpha_a = acc_a._flat_alpha
    # Compute the scalar BEFORE decay (for the τ rate parameter)
    if use_scalar_for_tau:
        scalar_for_tau = scalar_fn(alpha_a)
        # τ(scalar) per Eq 11: linear interp from τ_min (at scalar=1) to τ_max (at scalar=0)
        # For vacuity: u_v ∈ [0,1]. For dissonance: also in [0,1] under our normalisation.
        # So we can reuse decay_concentration with the scalar as if it were vacuity.
        # We need a custom decay because decay_concentration computes vacuity internally.
        S = alpha_a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        u_substitute = scalar_for_tau.clamp(0.0, 1.0)
        tau_min_s, tau_max_s = 60.0, 3600.0
        tau = tau_min_s + (tau_max_s - tau_min_s) * (1.0 - u_substitute)
        decay_factor = torch.exp(-inter_session_gap_s / tau.unsqueeze(-1))
        alpha_a_decayed = (alpha_a - 1.0) * decay_factor + 1.0
    else:
        alpha_a_decayed = decay_concentration(
            alpha_a, inter_session_gap_s, tau_min_s=60.0, tau_max_s=3600.0
        )

    # Now compute the threshold-side scalar AFTER decay (paper's protocol)
    threshold_scalar = scalar_fn(alpha_a_decayed)
    flagged_stale = threshold_scalar > threshold

    # Ground truth: which session-A voxels are absent from session B?
    n_a = int(acc_a._sorted_keys.shape[0])
    if n_a == 0 or acc_b._sorted_keys.numel() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "n_flagged": 0, "n_a_only": 0, "n_a": n_a}
    pos = torch.searchsorted(acc_b._sorted_keys, acc_a._sorted_keys)
    pos_clamped = pos.clamp(max=int(acc_b._sorted_keys.shape[0]) - 1)
    overlap = (pos < acc_b._sorted_keys.shape[0]) & (
        acc_b._sorted_keys[pos_clamped] == acc_a._sorted_keys
    )
    a_only = ~overlap

    n_flagged = int(flagged_stale.sum())
    n_correctly_flagged = int((flagged_stale & a_only).sum())
    n_a_only = int(a_only.sum())
    p = (n_correctly_flagged / n_flagged) if n_flagged > 0 else float("nan")
    r = (n_correctly_flagged / n_a_only) if n_a_only > 0 else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else float("nan")
    return {
        "threshold": threshold,
        "precision": p, "recall": r, "f1": f1,
        "n_flagged": n_flagged, "n_a_only": n_a_only,
        "n_overlap": int(overlap.sum()),
        "use_scalar_for_tau": use_scalar_for_tau,
    }


def m2_descriptor(acc: EvidenceAccumulator, *, channel: str, n_bins: int = 16) -> torch.Tensor:
    """Build the M2 descriptor entropy channel using `channel` scalar.

    channel: 'vacuity' or 'softmax_entropy' or 'dissonance'
    Returns L1-normalised histogram on voxels in the lower-half scalar
    of the accumulator (paper's `h_vac` rule, transposed for "high-confidence
    subset" interpretation across all 3 scalar choices).
    """
    alpha = acc._flat_alpha
    if channel == "vacuity":
        s = vacuity_from_alpha(alpha)
    elif channel == "softmax_entropy":
        s = softmax_entropy_from_alpha(alpha)
        # Normalise to [0,1] by dividing by log(C+1) for fair histogram
        C_plus_1 = alpha.shape[-1]
        import math
        s = s / max(1e-9, math.log(C_plus_1))
    elif channel == "dissonance":
        s = dissonance_from_alpha(alpha)
    else:
        raise ValueError(channel)
    # Restrict to lower-half (high-confidence) subset
    median_s = s.median()
    mask = s <= median_s
    s_sel = s[mask]
    # Histogram into n_bins on [0, 1]
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=s_sel.device)
    hist = torch.histc(s_sel.clamp(0.0, 1.0), bins=n_bins, min=0.0, max=1.0)
    hist = hist / hist.sum().clamp_min(1e-9)
    return hist


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a * b).sum().item()
    den = (a.norm() * b.norm()).clamp_min(1e-9).item()
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=Path("weights/pointnet2lite_v2_warmrestart_bigger.pt"))
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames-a", type=int, default=50)
    ap.add_argument("--frames-b", type=int, default=50)
    ap.add_argument("--voxel-size", type=float, default=0.25)
    ap.add_argument("--gap-s", type=float, default=600.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/decoupling_ablation.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Load W3-M ckpt
    model = PointNetLiteV2EDL(in_channels=4, num_classes=19, k=16).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    print(f"Loaded W3-M ckpt: ep {state['epoch']}, val_mIoU {state['val_miou']:.4f}\n",
          flush=True)

    # Build sessions
    root = args.root / "dataset" if (args.root / "dataset" / "sequences").exists() else args.root
    ds = SemanticKITTIDataset(root=root, sequences=[args.seq], synthetic=False)
    print(f"seq {args.seq}: {len(ds)} frames total")
    print(f"Building Session A (frames 0-{args.frames_a-1})...", flush=True)
    acc_a = build_session(model, ds, range(0, args.frames_a), device, voxel_size=args.voxel_size)
    print(f"  voxels: {acc_a._sorted_keys.shape[0]}")
    print(f"Building Session B (frames {args.frames_a}-{args.frames_a+args.frames_b-1})...",
          flush=True)
    acc_b = build_session(model, ds, range(args.frames_a, args.frames_a + args.frames_b),
                          device, voxel_size=args.voxel_size)
    print(f"  voxels: {acc_b._sorted_keys.shape[0]}\n")

    # M3 axis: vacuity vs dissonance for τ + threshold
    print("=== M3 axis (vacuity vs dissonance for decay + stale threshold) ===")
    print()
    results = {"m3": [], "m2": {}}
    for scalar_name, scalar_fn in [
        ("vacuity", vacuity_from_alpha),
        ("dissonance", dissonance_from_alpha),
    ]:
        print(f"--- M3 with {scalar_name} ---")
        print(f"{'threshold':>10s}  {'P':>8s}  {'R':>8s}  {'F1':>8s}  "
              f"{'n_flag':>8s}  {'n_a_only':>10s}")
        for thr in [0.3, 0.5, 0.7]:
            r = stale_voxel_pr_with_scalar(
                acc_a, acc_b,
                scalar_fn=scalar_fn,
                threshold=thr,
                use_scalar_for_tau=True,
                inter_session_gap_s=args.gap_s,
            )
            print(f"{thr:>10.2f}  {r['precision']:>8.4f}  {r['recall']:>8.4f}  "
                  f"{r['f1']:>8.4f}  {r['n_flagged']:>8d}  {r['n_a_only']:>10d}")
            r["scalar"] = scalar_name
            results["m3"].append(r)
        print()

    # M2 axis: descriptor similarity between session A and session B (same area)
    print("=== M2 axis (descriptor entropy channel comparison) ===")
    print("Session A vs Session B descriptor similarity (higher = better match):")
    print(f"{'channel':>20s}  {'cosine sim':>12s}")
    for channel in ["vacuity", "softmax_entropy", "dissonance"]:
        d_a = m2_descriptor(acc_a, channel=channel)
        d_b = m2_descriptor(acc_b, channel=channel)
        sim = cosine_sim(d_a, d_b)
        print(f"{channel:>20s}  {sim:>12.4f}")
        results["m2"][channel] = {"cosine_sim_A_B": sim,
                                    "hist_A": d_a.cpu().tolist(),
                                    "hist_B": d_b.cpu().tolist()}

    # Pareto-dominance verdict
    print()
    print("=== Pareto-dominance verdict (R4 Decoupling Ablation pass condition) ===")
    print()
    # On M3: compare F1 at each threshold
    vac_f1 = [r["f1"] for r in results["m3"] if r["scalar"] == "vacuity"]
    dis_f1 = [r["f1"] for r in results["m3"] if r["scalar"] == "dissonance"]
    print(f"M3 F1 at threshold 0.3: vacuity = {vac_f1[0]:.4f}, dissonance = {dis_f1[0]:.4f}")
    print(f"M3 F1 at threshold 0.5: vacuity = {vac_f1[1]:.4f}, dissonance = {dis_f1[1]:.4f}")
    print(f"M3 F1 at threshold 0.7: vacuity = {vac_f1[2]:.4f}, dissonance = {dis_f1[2]:.4f}")
    print()
    vac_wins_all = all(vac_f1[i] > dis_f1[i] for i in range(3))
    dis_wins_all = all(dis_f1[i] > vac_f1[i] for i in range(3))
    print(f"M3: vacuity strictly Pareto-dominates dissonance? {vac_wins_all}")
    print(f"M3: dissonance strictly Pareto-dominates vacuity? {dis_wins_all}")
    print()
    print(f"M2 descriptor similarity (A vs B same area):")
    print(f"  vacuity        = {results['m2']['vacuity']['cosine_sim_A_B']:.4f}")
    print(f"  softmax_entropy= {results['m2']['softmax_entropy']['cosine_sim_A_B']:.4f}")
    print(f"  dissonance     = {results['m2']['dissonance']['cosine_sim_A_B']:.4f}")
    vac_m2 = results['m2']['vacuity']['cosine_sim_A_B']
    se_m2 = results['m2']['softmax_entropy']['cosine_sim_A_B']
    diss_m2 = results['m2']['dissonance']['cosine_sim_A_B']
    m2_vac_wins = vac_m2 > max(se_m2, diss_m2)
    print(f"M2: vacuity is the highest-similarity channel? {m2_vac_wins}")
    print()

    overall_vac_dominates = vac_wins_all and m2_vac_wins
    print("=== Verdict ===")
    if overall_vac_dominates:
        print("PASS: Vacuity strictly Pareto-dominates on M3 (all thresholds) AND M2.")
        print("  →  §I.B 'one vacuity, three jobs' integration thesis SURVIVES.")
    else:
        print("FAIL: Vacuity does NOT strictly Pareto-dominate.")
        print("  →  §I.B integration thesis must be REFRAMED in this revision per the")
        print("      round-1 editorial decision.")
    print()

    verdict = {
        "vacuity_pareto_dominates_M3": vac_wins_all,
        "vacuity_pareto_dominates_M2": m2_vac_wins,
        "overall_thesis_survives": overall_vac_dominates,
    }
    out = {"results": results, "verdict": verdict,
           "ckpt": str(args.ckpt), "ckpt_epoch": int(state["epoch"]),
           "ckpt_val_mIoU": float(state["val_miou"]),
           "seq": args.seq, "frames_a": args.frames_a, "frames_b": args.frames_b,
           "voxel_size_m": args.voxel_size, "inter_session_gap_s": args.gap_s,
           }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
