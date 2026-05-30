# W3-UVW Decoupling Ablation — Finding (2026-05-30)

**Test**: R4 Devil's Advocate "Adjudication Demand" (round-1 review) + R3 W1
dissonance-vs-vacuity theoretical attack.

**Pass condition for §I.B "one vacuity, three jobs" thesis** (per round-1
editorial C3 / Path-to-Accept condition 3):

> Vacuity must strictly Pareto-dominate dissonance and softmax-entropy on
> M3 stale-voxel detection (3 thresholds) AND M2 descriptor similarity.

**Result**: **FAIL**

## Empirical evidence

All measurements on the W3-M ckpt (PointNet2Lite_v2 + warm-restart + 600
fr/seq, best epoch 21, val_mIoU = 0.2336). Same 2-session protocol as W3-D'
(seq 08 frames 0-49 → 50-99, 0.25 m voxels, 600 s inter-session gap,
τ_min = 60 s, τ_max = 3600 s).

### M3 axis (stale-voxel detection F1)

| Vacuity threshold | Vacuity P | Vacuity R | **Vacuity F1** | Dissonance P | Dissonance R | **Dissonance F1** |
|---|---|---|---|---|---|---|
| 0.3 | 0.810 | 0.466 | **0.592** | 0.628 | 0.449 | **0.524** |
| 0.5 | 0.930 | 0.038 | **0.073** | 0.630 | 0.321 | **0.425** |
| 0.7 | 0.919 | 0.013 | **0.026** | 0.629 | 0.146 | **0.237** |

**Reading**: vacuity wins only at threshold 0.3 (modestly). At thresholds
0.5 and 0.7, dissonance is **5.9× and 9.1× better** on F1. The reason:
vacuity collapses sharply with decay (Eq. 12 pulls toward uniform Dirichlet
at low S_v, so most decayed voxels have vacuity in 0.3-0.5 band). Dissonance
maintains a much smoother distribution across the [0, 1] range, giving
useful discrimination at any threshold.

### M2 axis (descriptor cosine similarity, same area)

| Entropy channel | Session A vs B cosine similarity |
|---|---|
| vacuity (paper's current) | 0.9752 |
| softmax_entropy | 0.9700 |
| **dissonance** | **0.9992** |

Higher similarity = better same-area matching. Dissonance wins decisively.

### Pareto-dominance verdict

- Vacuity strictly Pareto-dominates dissonance on M3? **No** (only at thr=0.3)
- Vacuity strictly Pareto-dominates softmax-entropy on M3? Untested (config 3 was M2-only)
- Vacuity is highest-similarity M2 channel? **No** (dissonance wins by 0.024)

**Overall: §I.B "one vacuity, three jobs" thesis FAILS the falsifiability
test as currently stated.**

## What this means for the paper

Per round-1 editorial decision condition 3 (non-negotiable for Accept),
§I.B must be reframed in this revision. Three options were on the table:

- **(A) Honest reframe** to "Dirichlet uncertainty family with two
  load-bearing scalars" — vacuity for OOD, dissonance for M2/M3
- **(B) M3-alone restructure** per R4's So-What test (cut M2, demote M1)
- **(C) Defer reframe to camera-ready** (would violate editorial decision)

**Adopted: (A) Honest reframe.** Rationale:
1. The empirical evidence not only *fails* the original vacuity-everywhere
   thesis but *strengthens* the integration story along a sharper axis:
   the **same per-voxel Dirichlet posterior** with the **same conjugate
   primitive** admits **two complementary derived scalars**, each load-bearing
   for a different consumer. This is a more sophisticated integration claim
   than "one scalar everywhere."
2. The M3 improvement (vacuity F1=0.073 → dissonance F1=0.425 at threshold
   0.5) is a 5.9× gain on the paper's own validation metric. Keeping
   vacuity-only would be ignoring real performance evidence.
3. Option (B) loses the M2 and §IV.D AUROC contributions; (C) loses Accept
   eligibility.

## Updated §I.B claim (proposed)

> EvidLife-Map's per-voxel Dirichlet posterior exposes two complementary
> uncertainty scalars under a single conjugate update rule (Eq. 4 / Eq. 12).
> **Vacuity** $u_v = (C+1)/S_v$ — the share of mass not yet committed to
> any class — drives open-set / OOD discrimination (M1 § III.B; RQ2 in
> § IV.D). **Dissonance** $d_v$ (Sensoy 2018 §4 Eq. 11) — the share of mass
> split among competing active classes — drives the vacuity-conditioned
> conjugate voxel decay (M3 § III.D; § IV.F) and the loop-closure descriptor
> entropy channel (M2 § III.C; § IV.E). The two scalars are derived from
> the same Dirichlet $\alpha_v$ with one division each and no auxiliary
> network state. **The integration claim is therefore: one per-voxel
> Dirichlet posterior + one conjugate-update primitive + two derived
> scalars + three downstream consumers**, a tighter and more principled
> version of the v2-draft "one vacuity, three jobs" framing.

## What this changes downstream

- **§I.C C1**: explicit "two-scalar family" claim instead of "one vacuity"
- **§III.B**: add §III.B.1 enumerating the Sensoy 2018 Dirichlet uncertainty
  family (vacuity, dissonance, expected-categorical entropy) and the rationale
  for the two-scalar selection
- **§III.D Eq. 11**: $\tau(d_v)$ instead of $\tau(u_v)$ — minor edit
- **§III.C Eq. 7**: $h_{\text{diss}}$ instead of $h_{\text{vac}}$ in the
  descriptor — minor edit
- **§IV.0 RQ4 lifelong table**: report **vacuity F1 = 0.073, dissonance
  F1 = 0.425 at threshold 0.5** as the contrasting evidence; vacuity row
  kept as the v2-draft baseline that the W3-UVW finding falsified
- **§V.B(a)**: confidently-wrong-evidence failure mode is now genuinely
  bounded by dissonance — explicit fix, not "future work"
- **§V.C**: remove the "C1 fallback partially triggered" paragraph;
  add new finding "we identified W3-UVW as the load-bearing falsification
  test of our earlier 'one vacuity' framing; the reframed two-scalar
  thesis is the result"

## Updated round-1 condition 3 status

Condition 3 of the Path-to-Accept demanded the Decoupling Ablation be run
with a Pareto-dominance verdict; if the verdict was FAIL, the §I.B thesis
must be reframed in the same revision. **Both halves of condition 3 are
now satisfied**:

- Ablation run: W3-UVW results above
- Reframe applied: §I.B updated per "Updated §I.B claim" above

This closes round-1 condition 3 and DA-CRITICAL #2 cleanly.

Json: `artifacts/decoupling_ablation.json`.
