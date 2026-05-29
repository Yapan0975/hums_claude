# S-BKI wrapper

**Upstream:** `https://github.com/ganlumomo/BKISemanticMapping` (RA-L 2020).
**Paper:** Gan et al., "Bayesian Spatial Kernel Smoothing for Scalable Dense
Semantic Mapping", RA-L 2020 (deep-read at `精读_T1.6_S-BKI.html`).

Low-risk reproduction (paper_tables.md Table III row 4): small C++ codebase,
mature, ~2 h build on a 4060 workstation.

## Bring-up

```
git clone https://github.com/ganlumomo/BKISemanticMapping external/sbki
cd external/sbki && mkdir build && cd build && cmake .. && make -j
```

The wrapper invokes the upstream binary via subprocess on a sequence of
``.bin`` files; we postprocess its output mesh into our voxel-aligned tensor.
