# Khronos reproduction notes (R-17 mitigation)

## Goal

Land an end-to-end Khronos run on SemKITTI seq 08 by **W20** (G-6 gate),
producing a closed-set mIoU comparable to the published Khronos numbers.

## Risk register

| Sub-risk                     | Likelihood | Detection                          | Action                                      |
|------------------------------|------------|------------------------------------|---------------------------------------------|
| Khronos targets RGB-D + IMU  | high       | upstream README                    | port SemKITTI LiDAR-only adapter (W18)      |
| TESSE-specific path planning | medium     | upstream config                    | swap to GT trajectory mode                  |
| Mesh extraction crashes      | medium     | run on a 10-frame slice first      | open issue upstream                         |
| arXiv code != published code | medium     | diff against arXiv 2402.13817 sup. | document and pick whichever produces numbers|
| Dependency breakage          | high       | docker build log                   | pin GCC 11 + Eigen 3.4 + ROS Noetic         |

## Fallback (R-17)

If by W20 the end-to-end run still fails:

1. Read the published dynamic-object F1 numbers from Khronos RSS-24 Table III.
2. Build `risk_log/khronos_published_numbers.json` with the cells we will cite.
3. Mark `KhronosAdapter(use_published_numbers_fallback=True)` in the RQ5 config.
4. Add the §V disclaimer:

   > Comparison numbers for Khronos in Table V are taken from the original
   > paper (Schmid et al., RSS 2024); a fully reproduced LiDAR-only Khronos
   > run on SemKITTI is left to future work because of the sensor-stack
   > mismatch with our deployment platform.

5. Pre-register this fallback path in W17 so reviewers cannot argue we
   retroactively adjusted the experimental protocol.

## Communications

* W17: email Khronos authors (MIT-SPARK) about SemKITTI port intent.
* W18: open a public discussion thread on Khronos GitHub issue tracker.
* W19: if author reply received, integrate their guidance.
