#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_baseline.sh  —  Step 3: zero-shot evaluation before fine-tuning
#
# Usage:
#   bash run_baseline.sh                              # run complet
#   bash run_baseline.sh --test                       # smoke test (10 samples)
#   bash run_baseline.sh --model=Diamegs/PIT-4B-FT-201312
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Cluster ───────────────────────────────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
NUM_GPUS=1
CPU_CORES=8
MEMORY="40G"

# ── Params ────────────────────────────────────────────────────────────────────
MODEL_NAME="Diamegs/PIT-4B-FT-201912"
DATA_PATH="data/merged_data.parquet"
OUTPUT_CSV="results/baseline_results.csv"

# ── Args ──────────────────────────────────────────────────────────────────────
TEST_FLAG=""
N_TEST=10
for arg in "$@"; do
  case $arg in
    --test)      TEST_FLAG="--test" ;;
    --n-test=*)  N_TEST="${arg#*=}" ;;
    --model=*)   MODEL_NAME="${arg#*=}" ;;
    *) echo "Usage: bash run_baseline.sh [--test] [--n-test=N] [--model=...]"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="pit-baseline${TEST_FLAG:+-test}-${TIMESTAMP}"

RUN_CMD="cd /home/bourgon/pit-stock-llm && mkdir -p results && \
  pip install -q matplotlib seaborn scikit-learn && \
  python -u src/evaluation/baseline.py \
  --model_name ${MODEL_NAME} \
  --data_path  ${DATA_PATH} \
  --output_csv ${OUTPUT_CSV}"
[ -n "$TEST_FLAG" ] && RUN_CMD="${RUN_CMD} --n_test ${N_TEST}"

# ─────────────────────────────────────────────────────────────────────────────
echo "Job    : ${JOB_NAME}"
echo "Modèle : ${MODEL_NAME}"
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
