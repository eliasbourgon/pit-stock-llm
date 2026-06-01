#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_rlvr.sh  —  Step 4: GRPO fine-tuning
#
# Usage:
#   bash run_rlvr.sh                              # run complet
#   bash run_rlvr.sh --test                       # smoke test (10 samples, 1 step)
#   bash run_rlvr.sh --model=Diamegs/PIT-4B-FT-201312 --output=checkpoints/pit-2013
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=1
CPU_CORES=16
MEMORY="40G"

# ── Params ────────────────────────────────────────────────────────────────────
MODEL_NAME="Diamegs/PIT-4B-FT-201912"
OUTPUT_DIR="checkpoints/pit-2019-rlvr"
DATA_PATH="data/merged_data.parquet"

# ── Args ──────────────────────────────────────────────────────────────────────
TEST_FLAG=""
N_TEST=10
for arg in "$@"; do
  case $arg in
    --test)      TEST_FLAG="--test" ;;
    --n-test=*)  N_TEST="${arg#*=}" ;;
    --model=*)   MODEL_NAME="${arg#*=}" ;;
    --output=*)  OUTPUT_DIR="${arg#*=}" ;;
    *) echo "Usage: bash run_rlvr.sh [--test] [--n-test=N] [--model=...] [--output=...]"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-rlvr${TEST_FLAG:+-test}-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && \
  pip install -q --upgrade trl peft bitsandbytes && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  CUDA_VISIBLE_DEVICES=0 python -u rlvr_pipeline.py \
  --model_name ${MODEL_NAME} \
  --data_path  ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR}"
[ -n "$TEST_FLAG" ] && RUN_CMD="${RUN_CMD} --test --n_test ${N_TEST}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job    : ${JOB_NAME}"
echo "Modèle : ${MODEL_NAME}"
echo "Output : ${OUTPUT_DIR}"
echo "Test   : ${TEST_FLAG:-non}"
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
