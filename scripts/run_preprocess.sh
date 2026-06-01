#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_preprocess.sh  —  Step 2: merge earnings calls with CRSP returns
#
# Usage:
#   bash run_preprocess.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=0
CPU_CORES=8
MEMORY="16G"

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT="data/Predictors/sm-calls_summarized_post2018.parquet"
RETURNS="data/Targets/monthly_crsp.csv"
OUTPUT="data/merged_data.parquet"

# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-preprocess-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && python -u src/preprocessing/pre_process.py \
  --input   ${INPUT} \
  --returns ${RETURNS} \
  --output  ${OUTPUT}"

echo "Job    : ${JOB_NAME}"
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
