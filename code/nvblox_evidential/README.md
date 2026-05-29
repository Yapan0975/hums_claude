# nvblox_evidential

C++ / CUDA plug-in implementing the `EvidentialLayer` over the upstream nvblox
layer cake (paper §III.F). This is the production implementation of M1 / M3;
the pure-Python reference in `../evidlife_map/evidlife_map/core/voxel_map.py`
is the test-time fallback.

## Compatibility

* **Upstream nvblox:** tag `v0.2.0` or later (compatible with the Isaac ROS 2
  release used in our docker image).
* **CUDA:** 12.1 (matches the workstation Dockerfile and the Jetson Orin NX
  L4T 35.x runtime).
* **C++:** C++20; gcc 11 / clang 15.
* **Python bindings:** pybind11 ≥ 2.11; expose `EvidentialLayer`, the
  `EvidentialIntegrator`, and the serialiser to `core.nvblox_bridge`.

## Build

```bash
cd nvblox_evidential
cmake -B build -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CUDA_ARCHITECTURES=86\;87  # 86 = 4060, 87 = Orin NX
cmake --build build -j
```

The resulting `_nvblox_evidential.so` is auto-loaded by
`evidlife_map.core.nvblox_bridge` when present on `PYTHONPATH`.

## File map

| Header                                                | Implements           | Paper §  |
|-------------------------------------------------------|----------------------|----------|
| `include/nvblox_evidential/evidential_voxel.h`        | per-voxel record     | §III.A   |
| `include/nvblox_evidential/evidential_layer.h`        | `BlockLayer` subclass | §III.F  |
| `include/nvblox_evidential/evidential_integrator.h`   | LiDAR splat → α      | §III.B   |
| `include/nvblox_evidential/evidential_io.h`           | serialiser           | §IV repro |
| `src/evidential_layer.cu`                             | CUDA kernel for α update | §III.B Eq 4 |
| `src/evidential_integrator.cu`                        | LiDAR ray-cast       | §III.F   |

## Memory accounting (paper §III.F)

`sizeof(EvidentialVoxel)` at `C+1 = 20`:

```
float    alpha[20]              =   80 B
uint32_t last_update_ts         =    4 B
uint16_t pose_idx               =    2 B
uint16_t _padding               =    2 B
-----------------------------------------
total                           =   88 B per voxel
```

At 512 voxels per 8×8×8 block, one block is `88 B × 512 = 44 032 B ≈ 44 kB`,
matching the figure in paper §III.B implementation paragraph and the T1.5
nvblox deep-read §9.
