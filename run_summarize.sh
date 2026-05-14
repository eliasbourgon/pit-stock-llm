#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_summarize.sh
# Soumet le job de preprocessing/summarization sur RunAI avec 3x A100 40GB
# Usage : bash run_summarize.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Paramètres à adapter ─────────────────────────────────────────────────────
JOB_NAME="ec-summarize-post2018"
PROJECT="<ton-projet-runai>"          # ex: "quant-research"
IMAGE="nvcr.io/nvidia/pytorch:24.01-py3"  # image avec PyTorch + CUDA 12
PVC_NAME="<ton-pvc>"                  # PVC où sont tes données
PVC_MOUNT="/workspace"                # point de montage dans le container
WORKING_DIR="/workspace"              # là où sont les .parquet et le .py

NUM_GPUS=3
GPU_TYPE="A100"                       # label GPU si configuré dans ton cluster
CPU_CORES=16
MEMORY="64G"

# ── Commande à exécuter dans le container ────────────────────────────────────
RUN_CMD="python preprocess_summarize.py"

# ─────────────────────────────────────────────────────────────────────────────
# Soumission RunAI
# ─────────────────────────────────────────────────────────────────────────────
runai submit "${JOB_NAME}" \
  --project "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${NUM_GPUS}" \
  --cpu "${CPU_CORES}" \
  --memory "${MEMORY}" \
  --pvc "${PVC_NAME}:${PVC_MOUNT}" \
  --working-dir "${WORKING_DIR}" \
  --command -- bash -c "${RUN_CMD}"

echo ""
echo "Job soumis : ${JOB_NAME}"
echo "Suivi      : runai logs ${JOB_NAME} -f"
echo "Status     : runai describe job ${JOB_NAME}"