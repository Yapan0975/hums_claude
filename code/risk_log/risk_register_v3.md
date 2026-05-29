# Risk register (v3)

Live mirror of research_plan v3 §6 with execution-time annotations.

Status legend: `open` (active) · `monitor` (low-prob, periodic check) ·
`triggered` (mitigation engaged) · `retired` (no longer applicable).

| ID    | Severity | Status | Risk                                           | Mitigation                                                       | Trigger to fall back |
|-------|----------|--------|------------------------------------------------|------------------------------------------------------------------|----------------------|
| R-2   | M        | monitor| Khronos lifelong extension lands first         | weekly arXiv monitor; sharpen our "vacuity, three jobs" framing  | sharpen claim        |
| R-3   | M        | monitor| GS-LIVO ships semantic extension               | differentiate on representation + Jetson-deployable lifelong     | sharpen claim        |
| R-4   | H        | open   | EDL head fails to beat ConvBKI in mIoU         | calibration / OOD-AUROC axis is independent                      | RA-L calibration-only|
| R-6   | M        | open   | KITTI-360 revisit overlap insufficient         | W3 pre-compute matrix; synthetic injection backup                | demote RQ4 to ECE-drift micro-study |
| R-7   | M        | open   | Voxel-size confound contaminates mIoU vs ConvBKI| A-6 ablation (drop-first stretch)                                | report at fixed 0.25 m  |
| R-8   | L        | monitor| Cylinder3D licence blocks code release         | RangeNet++ (BSD) as backup                                       | switch backbone      |
| R-9   | H        | open   | 8-page IROS overflow                            | aggressively push tables to supplementary                         | drop A-6; cut RQ5 detail |
| R-10  | M        | open   | Reviewer demands real-robot deployment         | Jetson Orin NX benchmark + 30-s demo video                        | desk-reject defence  |
| R-11  | H        | open   | RTX 4060 16 GB VRAM insufficient                | batch=1, freeze backbone, host-offload submaps                    | swap to SalsaNext    |
| R-12  | H        | open   | ConvBKI reimpl fails to reproduce 77.7 % mIoU  | W2 + W3 allocated; pin CUDA + PyTorch                             | quoted-numbers cell  |
| R-13  | M        | open   | KITTI-360 multi-session revisit too short      | pre-compute matrix; synthetic injection                           | AUROC of stale ranking|
| R-14  | M        | open   | EDL head training unstable                     | Sensoy 2018 KL anneal; grad-clip; warm-start                      | switch to PostNet    |
| R-15  | H        | open   | Open-set 14/5 split methodologically contested | pre-register split; nuScenes reverse cross-check                  | report mean ± std over splits |
| R-16  | M        | open   | SemKITTI lacks taxonomic diversity for "unknown" | 2-NN feature check at W5; Robo3D fallback                       | switch to Robo3D fallback |
| R-17  | H        | open   | Khronos reproduction fails for RQ5 (G-6)        | W17 author contact; W18-W20 bring-up; published-numbers fallback | quoted-numbers Table V |

## Per-gate triggers

* G-2 red ⇒ R-4 triggered ⇒ §0 fallback (calibration-only RA-L paper).
* G-3 red after W18 ⇒ R-13 triggered ⇒ demote RQ4.
* G-4 red ⇒ R-9 triggered (descriptor variant pruned).
* G-5 red ⇒ R-15 / R-16 triggered ⇒ Robo3D fallback within the same week.
* G-6 red ⇒ R-17 triggered ⇒ published-numbers Table V hybrid.

## Weekly review cadence

* Every Monday: walk this table, update `Status`, record any new evidence in
  the corresponding row.
* Every gate (W3, W4, W12, W17, W20): replay the trigger column and confirm
  no mitigation has silently failed.
