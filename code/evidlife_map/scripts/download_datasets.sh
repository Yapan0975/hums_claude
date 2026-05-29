#!/usr/bin/env bash
# W0b — Public dataset download script.
#
# Run this on your *local machine* (not in chat / not in cloud) so the large
# downloads go to your physical disk. Uses aria2c for multi-stream + resume.
#
# Total disk: ~85 GB (SemanticKITTI 80 + KITTI-360 100 of which we keep ~30 GB
# subset + SemanticSpray 10 GB + Robo3D-SemanticKITTI scripts).
# Recommended drive: D:\datasets (must have ≥ 250 GB free for headroom).
#
# Usage:
#   bash scripts/download_datasets.sh /d/datasets
#
# Or on Windows PowerShell:
#   wsl bash scripts/download_datasets.sh /mnt/d/datasets
#
# Prerequisites:
#   - aria2 (apt install aria2  OR  scoop install aria2  OR  brew install aria2)
#   - ~85 GB disk space on the target drive
#
# Datasets:
#   1. SemanticKITTI (~80 GB)         — RQ1, RQ2, RQ5 main
#   2. KITTI-360 subset (~30 GB)      — RQ4 lifelong  (skipping image frames)
#   3. SemanticSpray (~10 GB)         — RQ3 traversability + RQ2 cross-check
#   4. Robo3D scripts (kilobytes)     — RQ2 fallback (deterministic transforms)
#
set -euo pipefail

DATASET_ROOT="${1:-D:/datasets}"
echo "==> Dataset target root: ${DATASET_ROOT}"
mkdir -p "${DATASET_ROOT}"

# Helper: only download if file is missing or size mismatch.
download() {
    local url="$1"; local out_dir="$2"; local out_name="$3"
    mkdir -p "${out_dir}"
    if [ -f "${out_dir}/${out_name}" ] && [ -s "${out_dir}/${out_name}" ]; then
        echo "  SKIP ${out_name} (exists, $(du -h "${out_dir}/${out_name}" | cut -f1))"
        return 0
    fi
    echo "  DL   ${url} -> ${out_dir}/${out_name}"
    aria2c -x 8 -s 8 --continue=true --dir="${out_dir}" --out="${out_name}" "${url}"
}

# ============================================================
# 1. SemanticKITTI  — http://www.semantic-kitti.org/dataset.html
# Splits:
#   train: seq 00..07, 09, 10  (RQ1 train)
#   val:   seq 08              (RQ1 main eval, RQ2 open-set, G-1 G-2 G-5 gate)
#   test:  seq 11..21          (no labels; only for submission)
#
# For W1 G-1, ONLY seq 08 is strictly needed (~4 GB). The rest can stream in.
# ============================================================
echo ""
echo "==> [1/4] SemanticKITTI"
SK_ROOT="${DATASET_ROOT}/SemanticKITTI"
mkdir -p "${SK_ROOT}/sequences"

# Strategy: download via the official mirror at semantic-kitti.org.
# Note: the website redirects to several .zip/.tar parts. The script below
# fetches the "labels and calib" archive (~600 MB) which contains all label
# files and pose data — minimal first step.
download "http://www.semantic-kitti.org/assets/data_odometry_labels.zip" \
         "${SK_ROOT}" "data_odometry_labels.zip"

cat <<'NOTE'

  -----------------------------------------------------------------
  The full SemanticKITTI Velodyne .bin files (~80 GB) come from the
  official KITTI odometry server, which requires CAPTCHA registration.

  Manual step required:
    1. Go to https://www.cvlibs.net/datasets/kitti/eval_odometry.php
    2. Register (free); accept CC-BY-NC-SA terms.
    3. Download "velodyne laser data (80 GB)" → put .zip into
       ${SK_ROOT}/data_odometry_velodyne.zip
    4. Re-run this script — it will detect & unzip.

  G-1 (W3) MINIMUM: you only need seq 08 (~4 GB) to bring up R2-reimpl.
  -----------------------------------------------------------------
NOTE

if [ -f "${SK_ROOT}/data_odometry_velodyne.zip" ]; then
    echo "  EXTRACT data_odometry_velodyne.zip"
    unzip -q -n "${SK_ROOT}/data_odometry_velodyne.zip" -d "${SK_ROOT}"
fi
if [ -f "${SK_ROOT}/data_odometry_labels.zip" ]; then
    echo "  EXTRACT data_odometry_labels.zip"
    unzip -q -n "${SK_ROOT}/data_odometry_labels.zip" -d "${SK_ROOT}"
fi


# ============================================================
# 2. KITTI-360 (poses only for W1; full LiDAR streams ~100 GB later)
# https://www.cvlibs.net/datasets/kitti-360/
# ============================================================
echo ""
echo "==> [2/4] KITTI-360 (W1 minimal: poses only, ~50 MB)"
K360_ROOT="${DATASET_ROOT}/KITTI-360"
mkdir -p "${K360_ROOT}/data_poses"
download "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc7e564354e9418/data_poses.zip" \
         "${K360_ROOT}" "data_poses.zip"
if [ -f "${K360_ROOT}/data_poses.zip" ]; then
    unzip -q -n "${K360_ROOT}/data_poses.zip" -d "${K360_ROOT}"
fi
cat <<'NOTE'

  KITTI-360 LiDAR streams (~100 GB) needed at W8 only (M2 loop closure).
  For W3 G-3 (revisit matrix), only poses are needed — already covered.
  -----------------------------------------------------------------
NOTE


# ============================================================
# 3. SemanticSpray (wet-road)  — https://semantic-spray-dataset.github.io/
# ============================================================
echo ""
echo "==> [3/4] SemanticSpray (~10 GB)"
SS_ROOT="${DATASET_ROOT}/SemanticSpray"
mkdir -p "${SS_ROOT}"
cat <<'NOTE'

  SemanticSpray is distributed via HuggingFace at:
    https://huggingface.co/datasets/aldipiroli/SemanticSprayDataset

  Recommended (requires `huggingface-cli login`):
    huggingface-cli download --repo-type dataset aldipiroli/SemanticSprayDataset \
        --local-dir ${SS_ROOT}/

  Or git-lfs clone:
    git lfs clone https://huggingface.co/datasets/aldipiroli/SemanticSprayDataset ${SS_ROOT}/
  -----------------------------------------------------------------
NOTE


# ============================================================
# 4. Robo3D scripts (deterministic LiDAR corruption — RQ2 fallback)
# https://github.com/ldkong1205/Robo3D
# ============================================================
echo ""
echo "==> [4/4] Robo3D scripts (kilobytes; RQ2 fallback path per v3 R-15)"
R3D_ROOT="${DATASET_ROOT}/Robo3D"
if [ ! -d "${R3D_ROOT}/.git" ]; then
    git clone --depth=1 https://github.com/ldkong1205/Robo3D.git "${R3D_ROOT}"
else
    echo "  SKIP (already cloned)"
fi


# ============================================================
# Summary
# ============================================================
echo ""
echo "==> Disk usage:"
du -sh "${DATASET_ROOT}"/* 2>/dev/null | sort -h
echo ""
echo "==> Next steps:"
cat <<'NEXT'

  1. Verify SemanticKITTI seq 08 is present:
     ls ${DATASET_ROOT}/SemanticKITTI/sequences/08/velodyne/ | wc -l   # expect 4071
     ls ${DATASET_ROOT}/SemanticKITTI/sequences/08/labels/   | wc -l   # expect 4071

  2. Update configs/evidlife_kitti.yaml:
     dataset_root: ${DATASET_ROOT}/SemanticKITTI

  3. Run W1 smoke test:
     cd D:/_7_sci/semantic_mapping/_new_paper/code/evidlife_map
     python -m pytest tests/ -q
     python -m scripts.smoke_test_real_kitti --root ${DATASET_ROOT}/SemanticKITTI --seq 08 --frames 5

  4. When ready, run R2-reimpl baseline on seq 08 for G-1:
     bash scripts/eval_rq1.sh --baseline r2_reimpl --seq 08

NEXT
