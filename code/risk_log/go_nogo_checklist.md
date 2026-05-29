# Go / No-Go checklist

Mirror of `evidlife_map/docs/go_nogo_checklist.md`. Updated in lock-step.

| Gate | Deadline | Criterion | Trigger condition |
|------|----------|-----------|-------------------|
| G-1  | W3 | R2-reimpl end-to-end mIoU on SK seq 08 within ±2 of Cylinder3D publication | slip 1 week; W4 still red → switch backbone to RangeNet++ |
| G-2  | W4 | M1 EDL head ≥ +1 mIoU over G-1 baseline on SK seq 08 | red → §0 fallback (calibration-only RA-L) |
| G-3  | W3 | ≥ 5 KITTI-360 revisit pairs with ≥ 30 m sustained overlap | red → synthetic injection (R-13); W18 still red → demote RQ4 |
| G-4  | W12 | M2 loop closure ≥ 70 % precision @ 50 % recall on SK seq 08 self-loop | red → drop A-3 entropy descriptor; use h_class only |
| G-5  | W17 | Open-set AUROC ≥ 0.80 AND AUPR ≥ 0.60 (or Robo3D fallback ≥ 0.80) | red → demote RQ2 to negative result; C1 → "one vacuity, two jobs" |
| G-6  | W20 | Khronos end-to-end run on SK seq 08 reproduces published mIoU | red → published-numbers Table V hybrid (R-17) |

Gates are checked at the deadline date with a single command
(`python -m evidlife_map.eval.<gate>_check`) per gate, added in the relevant week.
