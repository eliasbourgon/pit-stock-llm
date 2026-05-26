#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_qwen3_rlvr.sh  —  Launch 3 Qwen3-4B RLVR training jobs (one per reward)
#
# Usage:
#   bash submit_qwen3_rlvr.sh [binary|pnl|gaussian|all]
#   Default: all (launches 3 independent jobs)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TARGET="${1:-all}"

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
NUM_GPUS=1
CPU_CORES=12
MEMORY="40G"

# ── Shared params ─────────────────────────────────────────────────────────────
MODEL_PATH="/home/bourgon/models/qwen3-4b"
DATA_PATH="data/merged_data.parquet"
DATA_OFFSET=0
SAVE_STEPS=100
WANDB_PROJECT="rlvr-earnings-qwen3"

# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# ── W&B API key — lue depuis .env dans le répertoire du projet ────────────────
WANDB_API_KEY=$(grep WANDB_API_KEY /home/bourgon/pit-stock-llm/.env | cut -d '=' -f2)
if [ -z "${WANDB_API_KEY}" ]; then
  echo "ERREUR : WANDB_API_KEY introuvable dans /home/bourgon/pit-stock-llm/.env"
  exit 1
fi

submit_job() {
    local REWARD="$1"
    local SCRIPT="qwen3_rlvr_${REWARD}.py"
    local OUTPUT_DIR="checkpoints/qwen3-${REWARD}-v1"
    local JOB_NAME="pit-qwen3-${REWARD}-${TIMESTAMP}"

    RUN_CMD="cd /home/bourgon/pit-stock-llm && mkdir -p ${OUTPUT_DIR} && \
  pip install -q --upgrade trl peft bitsandbytes vllm wandb && \
  export WANDB_API_KEY=${WANDB_API_KEY} && \
  wandb login ${WANDB_API_KEY} && \
  export HF_HOME=/home/bourgon/.cache/huggingface && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  python -u ${SCRIPT} \
  --model_path   ${MODEL_PATH} \
  --data_path    ${DATA_PATH} \
  --output_dir   ${OUTPUT_DIR} \
  --data_offset  ${DATA_OFFSET} \
  --save_steps   ${SAVE_STEPS} \
  --wandb \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run_name qwen3-${REWARD}-v1"

    echo "Submitting [${REWARD}] → ${JOB_NAME}"

    runai submit "${JOB_NAME}" \
      --project     "${PROJECT}" \
      --image       "${IMAGE}" \
      --gpu         "${NUM_GPUS}" \
      --cpu         "${CPU_CORES}" \
      --memory      "${MEMORY}" \
      --run-as-uid  "$(id -u)" \
      --run-as-gid  "$(id -g)" \
      -e USER="$(whoami)" -e HOME="/home/bourgon" -e PYTHONUNBUFFERED=1 \
      --pvc         "${PVC_HOME}:/home/bourgon" \
      --working-dir "/tmp" \
      --command -- bash -c "${RUN_CMD}"

    echo "  Logs   : runai logs ${JOB_NAME} -f"
    echo "  Stop   : runai delete job ${JOB_NAME}"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
echo "Model  : ${MODEL_PATH}"
echo "Data   : ${DATA_PATH} (offset=${DATA_OFFSET})"
echo "Save   : every ${SAVE_STEPS} steps (all checkpoints kept)"
echo "W&B    : ${WANDB_PROJECT}"
echo "─────────────────────────────────────────────────────────────────────────"

case "${TARGET}" in
  binary)   submit_job binary   ;;
  pnl)      submit_job pnl      ;;
  gaussian) submit_job gaussian ;;
  all)
    submit_job binary
    submit_job pnl
    submit_job gaussian
    ;;
  *)
    echo "Unknown target: ${TARGET}. Use binary | pnl | gaussian | all"
    exit 1
    ;;
esac
