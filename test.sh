#!/bin/bash
set -e
set -x

# Define paths relative to this script (assumed to be running from 'code/' directory)
# data: ../tcdata
# output: ../prediction_result/result.csv
DATA_DIR="../tcdata"
USER_DATA_DIR="../user_data"
PRED_RESULT_DIR="../prediction_result"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

source .venv/bin/activate

echo "=================================================="
echo "Current Directory: $(pwd)"
echo "Data Directory: ${DATA_DIR}"
echo "User Data Directory: ${USER_DATA_DIR}"
echo "=================================================="

# Ensure directories exist
mkdir -p ${USER_DATA_DIR}/processed
mkdir -p ${USER_DATA_DIR}/model_data
mkdir -p ${USER_DATA_DIR}/logs
mkdir -p ${PRED_RESULT_DIR}

# -----------------------------------------------------------------------------
# 1. Data Preparation
# -----------------------------------------------------------------------------
log ">> Step 1: Preparing Data..."
# make data
python -m tc_news_rec.scripts.prepare_data

# -----------------------------------------------------------------------------
# 2. Training
# -----------------------------------------------------------------------------
log ">> Step 2: Training Model..."
# make train
python tc_news_rec/scripts/train.py

# -----------------------------------------------------------------------------
# 3. Prediction
# -----------------------------------------------------------------------------
log ">> Step 3: Generating Predictions..."
# make predict
python tc_news_rec/scripts/predict.py

echo "Pipeline completed successfully."
