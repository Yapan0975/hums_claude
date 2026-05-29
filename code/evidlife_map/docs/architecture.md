# Architecture

ASCII overview of the EvidLife-Map P1 skeleton. Paper §III.A gives the same
diagram in prose; this file is the engineering version.

```
                        +-------------------+
                        |  LiDAR (10 Hz)    |
                        +---------+---------+
                                  |
                                  v
                +-----------------+-----------------+
                |  Cylinder3D backbone (frozen)     |  m1_evidential.edl_head
                |    last linear -> EDL evidence    |  (paper Eq 1)
                +-----------------+-----------------+
                                  |
                                  v
   +------------------------------+-----------------------------+
   |   M1  EvidenceAccumulator (paper Eq 4)                     |
   |   per-voxel α ∈ R^{C+1}, vacuity u_v = (C+1)/S_v (Eq 2)    |
   +-----------+----------------------------+-------------------+
               |                            |
        (M2 reads u_v)               (M3 reads u_v)
               |                            |
               v                            v
   +-----------+-----------+   +------------+----------------+
   |  M2  SubmapDescriptor |   |  M3  conjugate_decay_step   |
   |     (Eq 7)            |   |     (Eq 11, Eq 12)          |
   |  + matcher (Eq 8)     |   |  τ(u_v) sweep, α-1 pull     |
   |  + consistency (Eq 9) |   +------------+----------------+
   +-----------+-----------+                |
               |                            |
               +------------+---------------+
                            |
                            v
              +-------------+---------------+
              |   Downstream consumers      |
              |   * RQ2 vacuity-as-OOD      |
              |   * RQ3 traversability      |
              |   * RQ4 lifelong eval       |
              |   * RQ5 dynamic recall      |
              +-----------------------------+
```

## Module ↔ paper-equation cross-reference

| Module                            | Paper §    | Equation     |
|-----------------------------------|------------|--------------|
| `m1_evidential.edl_head`          | §III.B     | Eq 1, 2, 3   |
| `m1_evidential.loss`              | §III.B     | Eq 5         |
| `m1_evidential.fusion`            | §III.B     | Eq 4         |
| `m1_evidential.calibration`       | §III.B     | Eq 6         |
| `m2_loop_closure.descriptor`      | §III.C     | Eq 7         |
| `m2_loop_closure.matcher`         | §III.C     | Eq 8         |
| `m2_loop_closure.consistency`     | §III.C     | Eq 9         |
| `m3_decay.voxel_age`              | §III.D     | Eq 10        |
| `m3_decay.decay_rate`             | §III.D     | Eq 11        |
| `m3_decay.conjugate_decay`        | §III.D     | Eq 12        |

## Cross-module coupling (paper §III.E)

The integrated claim "one vacuity, three jobs" is realised by computing
``u_v`` exactly once (in `m1_evidential.edl_head.vacuity_from_alpha`) and
reading it from three places:

1. RQ2 open-set head (`eval.auroc.vacuity_auroc`).
2. M2 descriptor (`m2_loop_closure.descriptor.build_submap_descriptor` —
   the ``h_vac`` channel and the confident-voxel median filter).
3. M3 decay clock (`m3_decay.decay_rate.tau_from_vacuity`).

This is the "zero additional per-voxel runtime cost" property paper §III.E
ascribes to the design.
