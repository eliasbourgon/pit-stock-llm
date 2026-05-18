#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_rlvr_ddp.sh  —  GRPO fine-tuning on 3 GPUs (DDP via torchrun)
#
# Usage:
#   bash submit_rlvr_ddp.sh                              # full run (1 epoch)
#   bash submit_rlvr_ddp.sh --test                       # 1 step — mesure temps
#   bash submit_rlvr_ddp.sh --model=Diamegs/PIT-4B-FT-201312 --output=checkpoints/pit-2013-ddp
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
MODEL_NAME="Diamegs/PIT-4B-FT-201912"
OUTPUT_DIR="checkpoints/pit-2019-rlvr-ddp"
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
    *) echo "Usage: bash submit_rlvr_ddp.sh [--test] [--n-test=N] [--model=...] [--output=...]"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-rlvr-ddp${TEST_FLAG:+-test}-${TIMESTAMP}"

# torchrun spawns NUM_GPUS processes et injecte RANK / LOCAL_RANK / WORLD_SIZE.
# Pas de CUDA_VISIBLE_DEVICES — torchrun assigne les GPUs automatiquement.
RUN_CMD="cd /home/bourgon/pit-stock-llm && \
  pip install -q --upgrade trl peft bitsandbytes && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 rlvr_pipeline_ddp.py \
  --model_name ${MODEL_NAME} \
  --data_path  ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR}"
[ -n "$TEST_FLAG" ] && RUN_CMD="${RUN_CMD} --test --n_test ${N_TEST}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job    : ${JOB_NAME}"
echo "Model  : ${MODEL_NAME}"
echo "Output : ${OUTPUT_DIR}"
echo "GPUs   : ${NUM_GPUS}"
echo "Test   : ${TEST_FLAG:-non}"
if [ -n "$TEST_FLAG" ]; then
  echo ""
  echo "  Speed test: 1 step sur ${NUM_GPUS} GPUs."
  echo "  Comparer avec submit_rlvr_fast.sh --test pour voir le gain DDP."
fi
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
