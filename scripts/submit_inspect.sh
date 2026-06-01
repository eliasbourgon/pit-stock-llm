#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_inspect.sh  —  Qualitative generation inspector (single GPU)
#
# Usage:
#   bash submit_inspect.sh
#   bash submit_inspect.sh --checkpoint=checkpoints/pit-2019-rlvr-ddp-v3/checkpoint-600
#   bash submit_inspect.sh --checkpoint=... --n-samples=10 --rlvr-only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=1
CPU_CORES=8
MEMORY="32G"

# ── Params ────────────────────────────────────────────────────────────────────
BASE_MODEL="Diamegs/PIT-4B-FT-201912"
RLVR_CHECKPOINT="checkpoints/pit-2019-rlvr-ddp-v3/checkpoint-600"
DATA_PATH="data/merged_data.parquet"
DATA_OFFSET=2000
N_SAMPLES=5
NUM_VOTES=2
RLVR_ONLY=""

# ── Args ──────────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --checkpoint=*) RLVR_CHECKPOINT="${arg#*=}" ;;
    --offset=*)     DATA_OFFSET="${arg#*=}" ;;
    --n-samples=*)  N_SAMPLES="${arg#*=}" ;;
    --num-votes=*)  NUM_VOTES="${arg#*=}" ;;
    --rlvr-only)    RLVR_ONLY="--rlvr_only" ;;
    *) echo "Usage: bash submit_inspect.sh [--checkpoint=...] [--offset=N] [--n-samples=N] [--num-votes=N] [--rlvr-only]"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-inspect-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  python src/evaluation/inspect_generations.py \
  --base_model      ${BASE_MODEL} \
  --rlvr_checkpoint ${RLVR_CHECKPOINT} \
  --data_path       ${DATA_PATH} \
  --data_offset     ${DATA_OFFSET} \
  --n_samples       ${N_SAMPLES} \
  --num_votes       ${NUM_VOTES} \
  ${RLVR_ONLY}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job        : ${JOB_NAME}"
echo "Base model : ${BASE_MODEL}"
echo "Checkpoint : ${RLVR_CHECKPOINT}"
echo "Samples    : ${N_SAMPLES}  |  votes/sample: ${NUM_VOTES}  |  offset: ${DATA_OFFSET}"
echo "RLVR only  : ${RLVR_ONLY:-no}"
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