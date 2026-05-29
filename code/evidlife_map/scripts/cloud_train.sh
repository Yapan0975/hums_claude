#!/usr/bin/env bash
# Cloud training launcher (plan v4 §4.6 deliverable).
#
# Run this script ON THE CLOUD GPU INSTANCE, after rsyncing the repo over.
#
# Workflow:
#   1. From local 5060 (D:\_7_sci):
#        rsync -avzh --exclude='*.pdf' --exclude='_h5' --exclude='__pycache__' \
#          semantic_mapping/_new_paper/code/ \
#          user@cloud-host:~/evidlife/code/
#        rsync -avzh D:/datasets/SemanticKITTI/sequences/{08} \
#          user@cloud-host:~/data/SemanticKITTI/sequences/
#
#   2. SSH into cloud instance.
#
#   3. Run this script:
#        bash scripts/cloud_train.sh <experiment_name> [--no-shutdown]
#
# Supported experiments (from plan v4 §4.4 + research_plan v3 §4):
#   rq1_convbki_reproduce      ConvBKI on SemKITTI seq 08 (W2 sanity)
#   rq1_main                   Full RQ1 6-system × 3-dataset eval (W18)
#   rq2_openset                Open-set vacuity AUROC/AUPR on SemKITTI val (W17)
#   rq4_lifelong               KITTI-360 multi-session fusion eval (W15)
#   rq5_khronos_dynamic        SemKITTI dynamic split vs Khronos (W20)
#   ablation_a5                ± Dirichlet inter-session fusion (W18-26)
#   ablation_a7                ± open-set channel 20-D vs 19-D (W18-26)
#
# Cost budget (plan v4 §4.6):
#   $80 total cap. Each call appends to ~/cost_ledger.txt.
set -euo pipefail

EXP_NAME="${1:-}"
NO_SHUTDOWN="${2:-}"
LEDGER="${HOME}/cost_ledger.txt"
ARTIFACT_ROOT="${HOME}/artifacts"

if [ -z "${EXP_NAME}" ]; then
    echo "Usage: $0 <experiment_name> [--no-shutdown]"
    echo ""
    echo "Supported experiments:"
    sed -n '/Supported experiments/,/Cost budget/p' "$0" | tail -n +2 | head -n -1
    exit 1
fi

# ----- Environment bring-up -----
echo "==> [cloud_train] Bring up environment for: ${EXP_NAME}"
nvidia-smi | head -12
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU: ${GPU_NAME}, ${GPU_MEM_TOTAL_MB} MB total"

# Detect provider for cost tracking (best-effort)
PROVIDER="unknown"
if [ -f /etc/autodl-info ]; then PROVIDER="autoDL"; fi
if [ -f /etc/vast.ai_info ]; then PROVIDER="Vast.ai"; fi
if [ -f /etc/runpod_info ]; then PROVIDER="RunPod"; fi

# ----- Hourly cost estimate (auto-detect by GPU name) -----
HR_COST_USD="0.00"
case "${GPU_NAME}" in
    *A100*)         HR_COST_USD="1.50" ;;
    *4090*)         HR_COST_USD="0.40" ;;
    *3090*)         HR_COST_USD="0.30" ;;
    *T4*)           HR_COST_USD="0.20" ;;
    *)              HR_COST_USD="0.50" ;;  # conservative default
esac
echo "  Estimated cost: \$${HR_COST_USD}/hr on ${PROVIDER}"

# Budget check
SPENT_USD=$(awk -F'\t' '{sum+=$3} END {printf "%.2f", sum}' "${LEDGER}" 2>/dev/null || echo "0.00")
REMAINING_USD=$(awk -v s="${SPENT_USD}" 'BEGIN {printf "%.2f", 80.00 - s}')
echo "  Budget: \$${SPENT_USD} spent / \$80.00 cap / \$${REMAINING_USD} remaining"
if (( $(echo "${REMAINING_USD} < 5.00" | bc -l) )); then
    echo "  ERROR: Budget remaining < \$5; refuse to start."
    exit 2
fi

# ----- Run the experiment -----
mkdir -p "${ARTIFACT_ROOT}/${EXP_NAME}"
START_TS=$(date +%s)
START_DATE=$(date -Iseconds)

set +e
case "${EXP_NAME}" in
    rq1_convbki_reproduce)
        python -m scripts.eval_rq1 --baseline convbki --seq 08 --output "${ARTIFACT_ROOT}/${EXP_NAME}/"
        ;;
    rq1_main)
        python -m scripts.eval_rq1 --all-baselines --datasets semkitti,nuscenes --output "${ARTIFACT_ROOT}/${EXP_NAME}/"
        ;;
    rq2_openset)
        python -m scripts.eval_rq2_openset --split primary --output "${ARTIFACT_ROOT}/${EXP_NAME}/"
        ;;
    rq4_lifelong)
        python -m scripts.eval_rq4_lifelong --revisit-matrix artifacts/revisit_matrix.json --output "${ARTIFACT_ROOT}/${EXP_NAME}/"
        ;;
    rq5_khronos_dynamic)
        python -m scripts.eval_rq5_khronos --output "${ARTIFACT_ROOT}/${EXP_NAME}/"
        ;;
    ablation_a5|ablation_a7)
        VAR="${EXP_NAME#ablation_}"
        python -m scripts.run_ablation --variable "${VAR}" --output "${ARTIFACT_ROOT}/${EXP_NAME}/"
        ;;
    *)
        echo "ERROR: unknown experiment ${EXP_NAME}"
        exit 1
        ;;
esac
EXIT_CODE=$?
set -e

END_TS=$(date +%s)
END_DATE=$(date -Iseconds)
ELAPSED_S=$((END_TS - START_TS))
ELAPSED_HR=$(awk -v s="${ELAPSED_S}" 'BEGIN {printf "%.3f", s / 3600.0}')
COST_USD=$(awk -v hr="${ELAPSED_HR}" -v rate="${HR_COST_USD}" 'BEGIN {printf "%.2f", hr * rate}')

# Append to ledger (tab-separated: date, experiment, cost, hours, exit_code)
mkdir -p "$(dirname "${LEDGER}")"
printf "%s\t%s\t%.2f\t%.3f\t%d\n" \
    "${END_DATE}" "${EXP_NAME}" "${COST_USD}" "${ELAPSED_HR}" "${EXIT_CODE}" \
    >> "${LEDGER}"

echo ""
echo "==> [cloud_train] Done."
echo "  Started:  ${START_DATE}"
echo "  Ended:    ${END_DATE}"
echo "  Elapsed:  ${ELAPSED_HR} hr"
echo "  Cost:     \$${COST_USD}"
echo "  Exit:     ${EXIT_CODE}"
echo ""
echo "==> [cloud_train] Pulling artifacts back to local (if rsync configured)..."
# Stub: local user runs the rsync from their workstation.
echo "  (No rsync configured; artifacts at ${ARTIFACT_ROOT}/${EXP_NAME}/)"
echo "  From local: rsync -avzh user@cloud-host:${ARTIFACT_ROOT}/${EXP_NAME}/ ./experiments/runs/"

# Auto-shutdown unless --no-shutdown
if [ "${NO_SHUTDOWN}" = "--no-shutdown" ]; then
    echo ""
    echo "==> [cloud_train] --no-shutdown: leaving instance running."
else
    echo ""
    echo "==> [cloud_train] Shutting down instance in 60s (Ctrl-C to abort)..."
    sleep 60
    sudo shutdown -h now || echo "  (shutdown not permitted; abort manually)"
fi

exit "${EXIT_CODE}"
