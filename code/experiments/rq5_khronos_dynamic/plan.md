# RQ5 — Dynamic-scene head-to-head vs Khronos

## Hypothesis (research_plan v3 §2 H5)

On SemKITTI dynamic frames (high `moving-*` GT fraction in seq 00, 04, 05, 07):

* Dynamic-object mIoU within **3** of Khronos.
* Static-region recall within **1 pp** of Khronos.
* End-to-end latency ≥ **5 Hz** on Jetson Orin NX (Khronos is not designed for
  Orin NX deployment).

## R-17 (highest-risk reproduction)

* G-6 at W20: Khronos reproduction runs end-to-end on SemKITTI seq 08.
* If red: ship Table V as "published numbers vs ours" hybrid; H5 weakens to
  "competitive with published Khronos numbers".

## Dynamic-frame selection

`select_dynamic_frames.py` (W18 deliverable) walks SemKITTI seq 00/04/05/07
and emits every frame in which ≥ 5 % of GT points carry a `moving-*` label.
Expected yield: 1500–2000 frames.

## Khronos reproduction notes

See `khronos_reproduction_notes.md` for the R-17 mitigation checklist.
