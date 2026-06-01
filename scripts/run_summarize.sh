#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_summarize.sh  —  Step 1: summarize earnings calls with vLLM
#
# Usage:
#   bash run_summarize.sh              # run complet
#   bash run_summarize.sh --test       # smoke test (10 samples)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=3
CPU_CORES=16
MEMORY="40G"

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT="data/Predictors/sm-calls_with_connectors.parquet"
OUTPUT="data/Predictors/sm-calls_summarized_post2018.parquet"

# ── Args ──────────────────────────────────────────────────────────────────────
TEST_FLAG=""
N_TEST=10
for arg in "$@"; do
  case $arg in
    --test)      TEST_FLAG="--test" ;;
    --n-test=*)  N_TEST="${arg#*=}" ;;
    *) echo "Usage: bash run_summarize.sh [--test] [--n-test=N]"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-summarize${TEST_FLAG:+-test}-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && python -u preprocess_summarize.py \
  --input  ${INPUT} \
  --output ${OUTPUT} \
  ${TEST_FLAG}"
[ -n "$TEST_FLAG" ] && RUN_CMD="${RUN_CMD} --n_test ${N_TEST}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job    : ${JOB_NAME}"
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
