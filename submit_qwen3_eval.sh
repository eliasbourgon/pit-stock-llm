#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_qwen3_eval.sh  —  Zero-shot eval of Qwen3-4B on earnings call prediction
#
# Usage:
#   bash submit_qwen3_eval.sh
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
MODEL_LOCAL_DIR="/home/bourgon/models/qwen3-4b"
DATA_PATH="data/merged_data.parquet"
OUTPUT_DIR="results/qwen3_eval"
DATA_OFFSET=2000
N_EVAL=100

# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-qwen3-eval-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && mkdir -p ${OUTPUT_DIR} && \
  pip install -q vllm scikit-learn matplotlib huggingface_hub && \
  export HF_HOME=/home/bourgon/.cache/huggingface && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  python -u qwen3_eval.py \
  --model_local_dir      ${MODEL_LOCAL_DIR} \
  --data_path            ${DATA_PATH} \
  --output_dir           ${OUTPUT_DIR} \
  --data_offset          ${DATA_OFFSET} \
  --n_eval               ${N_EVAL} \
  --tensor_parallel_size ${NUM_GPUS}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job    : ${JOB_NAME}"
echo "Model  : Qwen/Qwen3-4B → ${MODEL_LOCAL_DIR}"
echo "Eval   : ${N_EVAL} samples (offset=${DATA_OFFSET})"
echo "Output : ${OUTPUT_DIR}"
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
