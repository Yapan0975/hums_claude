# W2-1 Cylinder3D Integration · STATUS

## Done (commit-worthy)
- spconv-cu126 2.3.8 + 9 deps (numba/llvmlite/pccm/cumm-cu126/ccimport/pybind11/fire/portalocker/ninja) installed on 5090
- numpy 1.24.4 for numba compat (keeps system scipy 1.8 happy)
- spconv.pytorch.SubMConv3d + SparseConvTensor verified on RTX 5090 sm_120
- Cylinder3D code cloned to ~/Documents/yping/mapping/Cylinder3D/
- spconv 1.x → 2.x import patches (`import spconv` → `import spconv.pytorch as spconv`)
- spconv 1.x → 2.x feature assignment patches (36 `.features = ...` → `.replace_feature(...)`)
- spconv 1.x ckpt weight permutation (D,H,W,in,out) → (out,D,H,W,in) for 48 conv layers
- torch_scatter replaced with native torch.scatter_reduce_
- Cylinder3D model build + load_state_dict: 0 missing, 0 unexpected, 55.85 M params
- 9-dim feature voxelization (dxyz_pol + xyz_pol + xy_cart + intensity)
- Inference wrapper at scripts/cyl3d_inference.py

## Blocking remaining issue
spconv 2.x enforces same indice_key → same kernel size across ALL conv layers in network.
Cylinder3D `ResContextBlock` etc share keys but use kernels (1,3,3), (3,1,3), etc.
Fix requires rewriting segmentator_3d_asymm_spconv.py to give each conv unique indice_key — ~50 keys to refactor.

## W3 options
1. Refactor segmentator_3d_asymm_spconv.py (~3 hours engineering)
2. Build spconv 1.x from source against torch 2.11+cu130 (~2-4 hours)
3. Use PVKD (Cylinder3D lightweight successor, possibly already spconv 2.x compatible)
4. Use Cylinder3D's TorchSparse port if any

## Files modified (revertable backups in .bak)
- ~/Documents/yping/mapping/Cylinder3D/network/{cylinder_fea_generator,segmentator_3d_asymm_spconv,cylinder_spconv_3d}.py

---

## 2026-05-29 Update — Hit Deeper Blocker

After resolving the indice_key kernel-size constraint (15 patches applied),
inference now runs without spconv 2.x API errors. BUT output logits are ALL NaN.

### Diagnosis
- Model loads structurally OK (0 missing keys after permute(4,0,1,2,3))
- Forward pass executes (dense output shape (1, 20, 480, 360, 32) correct)
- Per-voxel logits = NaN across ALL 46k non-zero voxels
- BatchNorm running stats look reasonable from ckpt inspection

### Root cause
spconv 1.x → 2.x changed both:
1. Weight shape layout: (kD,kH,kW,in,out) → (out,kD,kH,kW,in) ✓ patched
2. Internal kernel index iteration order: (D,H,W) vs (W,H,D) ✗ NOT patched

The kernel index order affects weight value semantics — same shape, different
mapping of voxel-offsets to weight elements. A pure permute() can't fix this;
needs either:
- Source-build spconv 1.x for our torch 2.11+cu130 (2-4 h, may fail vs new
  PyTorch internals)
- Use PVKD (Cylinder3D spconv-2.x compatible port; reported in CVPR 2022)
- Use a different LiDAR semantic backbone (e.g. WaffleIron, MinkUNet)

### Patches in place (resume from here)
1. spconv 2.x imports: `import spconv.pytorch as spconv`
2. .features = X → .replace_feature(X) [36 sites]
3. indice_key per kernel shape [15 sites]
4. torch_scatter.scatter_max → torch.scatter_reduce_ [1 site]
5. 9-dim voxel input (dxyz_pol + xyz_pol + xy_cart + intensity)
6. correct volume bounds (max z=2, min z=-4 per official semantickitti.yaml)

All patches in ~/Documents/yping/mapping/Cylinder3D/network/*.py.bak preserved.

## Recommendation for W3
Skip spconv 1.x source build (risky). Two cleaner paths:
- (i) Train Cylinder3D from scratch on spconv 2.x using PVKD codebase or
      forked Cylinder3D-spconv2 repos on GitHub
- (ii) Adopt a simpler backbone (PointNet++/RandLA-Net) that has clean
      torch-only inference; sacrifice 5-10 mIoU but full reproducibility
