# Go / No-Go checklist (research_plan v3 §9)

Six gates, each with a calendar deadline and a documented fallback. Every gate
that is red triggers a written entry in `risk_log/risk_register_v3.md`.

| Gate | Deadline | Criterion | Fallback if red |
|------|----------|-----------|-----------------|
| G-1 | W3 | R2-reimpl produces an end-to-end mIoU on SemKITTI seq 08 within ±2 of published Cylinder3D | slip one week, then switch backbone to RangeNet++ |
| G-2 | W4 | M1 EDL head shows **≥ +1 mIoU** AND lower ECE vs G-1 baseline on SemKITTI seq 08 | retreat to a calibration-only RA-L paper (drop M2 + M3 to subsections) |
| G-3 | W3 | KITTI-360 odometry-overlap matrix yields **≥ 5 revisit pairs** with ≥ 30 m sustained overlap | augment with synthetic temporal-gap injection; if still < 5 by W18, demote RQ4 to single-session ECE-drift micro-study |
| G-4 | W12 | M2 loop closure achieves **≥ 70 % precision at 50 % recall** on SemKITTI seq 08 self-loop pairs | drop the entropy-channel descriptor variant (A-3) from main; use h_class only |
| G-5 | W17 | Open-set vacuity AUROC ≥ 0.80 AND AUPR ≥ 0.60 on SemKITTI 14/5 primary split (or Robo3D fallback metric ≥ 0.80) | demote RQ2 to honest negative result; C1 framing tightens to "one vacuity, two jobs" |
| G-6 | W20 | Khronos reproduction runs end-to-end on SemKITTI seq 08 (closed-set mIoU comparable to published numbers) | ship Table V as "published numbers vs ours" hybrid; RQ5 H5 weakens to "competitive with published Khronos numbers" |

## Reporting protocol

For each gate at its deadline:

1. Run `python -m evidlife_map.eval.<gate>_check` (created in the relevant week).
2. If green: tick the box in this file, commit with subject `[G-N green] …`.
3. If red: open an entry in `risk_log/risk_register_v3.md` referencing the gate
   number and the triggered fallback path.
