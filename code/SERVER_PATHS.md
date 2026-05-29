# Server workspace paths (2026-05-29)

## SSH
- Alias:      `ssh evidlife-server`  (= server@100.64.0.5 via Tailscale)
- Identity:   `~/.ssh/id_evidlife_server`

## Code repo on server
- Path:       `~/Documents/yping/mapping/code/`
- Python pkg: `~/Documents/yping/mapping/code/evidlife_map/evidlife_map/`
- Tests:      `~/Documents/yping/mapping/code/evidlife_map/tests/`
- Scripts:    `~/Documents/yping/mapping/code/evidlife_map/scripts/`

## Datasets on server
- SemanticKITTI: `/data/shared/SemanticKITTI/dataset/sequences/`
  - Currently only seq 08 first 100 frames (200 MB)
  - Full seq 08 (9.4 GB) pending: laptop->server bandwidth ~2 MB/s

## GPU allocation
- GPU 0: 21 GB free (others using; avoid)
- GPU 1: 25 GB free
- GPU 2: 28 GB free
- GPU 3: 31 GB free (preferred for W1)

## Quick re-smoke
ssh evidlife-server "cd ~/Documents/yping/mapping/code/evidlife_map && \
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python3 scripts/smoke_test_real_kitti.py \
  --root /data/shared/SemanticKITTI --seq 08 --frames 5 --device cuda"
