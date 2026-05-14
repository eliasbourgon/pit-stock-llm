#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
# Soumet le job de fine-tuning RLVR complet sur RunAI avec 3x A100
#
# Usage :
#   bash run_pipeline.sh                         # run complet
#   bash run_pipeline.sh --test                  # smoke test (10 samples)
#   bash run_pipeline.sh --skip-summarize        # skip step 1 (déjà fait)
#   bash run_pipeline.sh --test --skip-summarize --skip-preprocess
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paramètres cluster (à adapter) ───────────────────────────────────────────
PROJECT="sfi-sm-bourgon"
IMAGE="ayushkumartarun/course-cs-552-standard:v1"
# home  → code + scripts  (~/pit-stock-llm)
# scratch → datasets + checkpoints (gros fichiers)
PVC_HOME="home"
PVC_SCRATCH="sfi-sm-scratch"
WORKING_DIR="/workspace"

NUM_GPUS=3
CPU_CORES=16
MEMORY="40G"

# ── Paramètres pipeline ───────────────────────────────────────────────────────
MODEL_NAME="Diamegs/PIT-4B-FT-201912"    # checkpoint PIT à fine-tuner
OUTPUT_DIR="checkpoints/pit-2019-rlvr"

RAW_PARQUET="sm-calls_with_connectors.parquet"
SUMMARIZED_PARQUET="sm-calls_summarized_post2018.parquet"
RETURNS_CSV="data/Targets/monthly_crsp.csv"
MERGED_PARQUET="data/merged_data.parquet"

# ── Parsing des arguments ─────────────────────────────────────────────────────
TEST_FLAG=""
SKIP_SUMMARIZE=""
SKIP_PREPROCESS=""
SKIP_BASELINE=""
N_TEST=10

for arg in "$@"; do
  case $arg in
    --test)             TEST_FLAG="--test" ;;
    --n-test=*)         N_TEST="${arg#*=}" ;;
    --skip-summarize)   SKIP_SUMMARIZE="--skip_summarize" ;;
    --skip-preprocess)  SKIP_PREPROCESS="--skip_preprocess" ;;
    --skip-baseline)    SKIP_BASELINE="--skip_baseline" ;;
    --model=*)          MODEL_NAME="${arg#*=}" ;;
    --output=*)         OUTPUT_DIR="${arg#*=}" ;;
    *)
      echo "Argument inconnu : $arg"
      echo "Usage: bash run_pipeline.sh [--test] [--n-test=N] [--skip-summarize] [--skip-preprocess] [--skip-baseline] [--model=...] [--output=...]"
      exit 1
      ;;
  esac
done

# ── Nom du job (unique par run) ───────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [ -n "$TEST_FLAG" ]; then
  JOB_NAME="pit-pipeline-test-${TIMESTAMP}"
else
  JOB_NAME="pit-pipeline-${TIMESTAMP}"
fi

# ── Commande exécutée dans le container ──────────────────────────────────────
RUN_CMD="cd pit-stock-llm && python submit_job.py \
  --model_name ${MODEL_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --raw_parquet ${RAW_PARQUET} \
  --summarized_parquet ${SUMMARIZED_PARQUET} \
  --returns_csv ${RETURNS_CSV} \
  --merged_parquet ${MERGED_PARQUET} \
  ${TEST_FLAG} \
  ${SKIP_SUMMARIZE} \
  ${SKIP_PREPROCESS} \
  ${SKIP_BASELINE}"

# Ajout de --n_test seulement si test mode
if [ -n "$TEST_FLAG" ]; then
  RUN_CMD="${RUN_CMD} --n_test ${N_TEST}"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo "Job         : ${JOB_NAME}"
echo "Modèle      : ${MODEL_NAME}"
echo "Output      : ${OUTPUT_DIR}"
echo "Test mode   : ${TEST_FLAG:-non}"
echo "─────────────────────────────────────────────────────────────────────────"

runai submit "${JOB_NAME}" \
  --project     "${PROJECT}" \
  --image       "${IMAGE}" \
  --gpu         "${NUM_GPUS}" \
  --cpu         "${CPU_CORES}" \
  --memory      "${MEMORY}" \
  --pvc         "${PVC_HOME}:/workspace" \
  --pvc         "${PVC_SCRATCH}:/scratch" \
  --working-dir "${WORKING_DIR}" \
  --command -- bash -c "${RUN_CMD}"

echo ""
echo "Job soumis  : ${JOB_NAME}"
echo "Logs        : runai logs ${JOB_NAME} -f"
echo "Status      : runai describe job ${JOB_NAME}"
echo "Annuler     : runai delete job ${JOB_NAME}"
