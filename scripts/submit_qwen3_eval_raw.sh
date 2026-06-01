#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_qwen3_eval_raw.sh — Qwen3-4B eval on RAW (non-summarized) earnings calls
#
# Step 1: pre_process.py on raw parquet → merged_data_raw.parquet
# Step 2: qwen3_eval.py with larger context window
#
# Usage:
#   bash submit_qwen3_eval_raw.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=2
CPU_CORES=24
MEMORY="80G"

# ── Params ────────────────────────────────────────────────────────────────────
MODEL_LOCAL_DIR="/home/bourgon/models/qwen3-4b"
RAW_PARQUET="data/Predictors/sm-calls_with_connectors.parquet"
RETURNS_CSV="data/Targets/monthly_crsp.csv"
MERGED_RAW="data/merged_data_raw.parquet"
OUTPUT_DIR="results/qwen3_eval_raw"
DATA_OFFSET=2000
N_EVAL=100
MAX_PROMPT_CHARS=20000

# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-qwen3-eval-raw-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && mkdir -p ${OUTPUT_DIR} && \
  pip install -q vllm scikit-learn matplotlib huggingface_hub fastparquet && \
  export HF_HOME=/home/bourgon/.cache/huggingface && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  echo '--- Step 1: merging raw transcripts with returns ---' && \
  python pre_process.py \
    --input   ${RAW_PARQUET} \
    --returns ${RETURNS_CSV} \
    --output  ${MERGED_RAW} && \
  echo '--- Step 2: Qwen3 eval on raw transcripts ---' && \
  python -u qwen3_eval.py \
    --model_local_dir      ${MODEL_LOCAL_DIR} \
    --data_path            ${MERGED_RAW} \
    --output_dir           ${OUTPUT_DIR} \
    --data_offset          ${DATA_OFFSET} \
    --n_eval               ${N_EVAL} \
    --max_prompt_chars     ${MAX_PROMPT_CHARS} \
    --tensor_parallel_size ${NUM_GPUS}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job    : ${JOB_NAME}"
echo "Model  : ${MODEL_LOCAL_DIR}"
echo "Data   : ${RAW_PARQUET} (raw, no summarization)"
echo "Eval   : ${N_EVAL} samples (offset=${DATA_OFFSET})"
echo "Chars  : ${MAX_PROMPT_CHARS} per transcript"
echo "Output : ${OUTPUT_DIR}"
echo "─────────────────────────────────────────────────────────────────────────"

ENCODED=$(printf '%s' "${RUN_CMD}" | base64 | tr -d '\n')

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
  --command -- bash -c "echo ${ENCODED} | base64 -d | bash"

echo ""
echo "Logs   : runai logs ${JOB_NAME} -f"
echo "Status : runai describe job ${JOB_NAME}"
echo "Stop   : runai delete job ${JOB_NAME}"
