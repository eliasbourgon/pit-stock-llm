#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_eval.sh  —  Majority-vote @ 4 evaluation: baseline vs RLVR checkpoint
#
# Usage:
#   bash submit_eval.sh
#   bash submit_eval.sh --checkpoint=checkpoints/pit-2019-rlvr-ddp-v3/checkpoint-400
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=3
CPU_CORES=24
MEMORY="80G"

# ── Params ────────────────────────────────────────────────────────────────────
BASE_MODEL="Diamegs/PIT-4B-FT-201912"
RLVR_CHECKPOINT="checkpoints/pit-2019-rlvr-ddp-v3/checkpoint-600"
DATA_PATH="data/merged_data.parquet"
OUTPUT_DIR="results/eval_majority4"
DATA_OFFSET=2000
N_EVAL=200

# ── Args ──────────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --checkpoint=*) RLVR_CHECKPOINT="${arg#*=}" ;;
    --offset=*)     DATA_OFFSET="${arg#*=}" ;;
    --n-eval=*)     N_EVAL="${arg#*=}" ;;
    *) echo "Usage: bash submit_eval.sh [--checkpoint=...] [--offset=N] [--n-eval=N]"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-eval-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && mkdir -p ${OUTPUT_DIR} && \
  pip install -q scikit-learn matplotlib seaborn && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  torchrun --nproc_per_node=${NUM_GPUS} --master_port=29501 eval.py \
  --base_model      ${BASE_MODEL} \
  --rlvr_checkpoint ${RLVR_CHECKPOINT} \
  --data_path       ${DATA_PATH} \
  --output_dir      ${OUTPUT_DIR} \
  --data_offset     ${DATA_OFFSET} \
  --n_eval          ${N_EVAL}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job        : ${JOB_NAME}"
echo "Base model : ${BASE_MODEL}"
echo "Checkpoint : ${RLVR_CHECKPOINT}"
echo "Eval set   : ${N_EVAL} samples (offset=${DATA_OFFSET})"
echo "Output     : ${OUTPUT_DIR}"
echo "─────────────────────────────────────────────────────────────────────────"

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
  --pvc         "${PVC_SCRATCH}:/scratch" \
  --working-dir "/tmp" \
  --command -- bash -c "${RUN_CMD}"

echo ""
echo "Logs   : runai logs ${JOB_NAME} -f"
echo "Status : runai describe job ${JOB_NAME}"
echo "Stop   : runai delete job ${JOB_NAME}"
