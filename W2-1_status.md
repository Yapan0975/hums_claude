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
