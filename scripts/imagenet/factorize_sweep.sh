#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TIMM_FUSED_ATTN=0

MODEL_NAME="${MODEL_NAME:?MODEL_NAME not set}"
MODE="${MODE:?MODE not set}"
SEED="${SEED:-0}"
IMAGENET_ROOT="${IMAGENET_ROOT:?IMAGENET_ROOT not set}"
TRAIN_DIR="${IMAGENET_ROOT}/train"
VAL_DIR="${IMAGENET_ROOT}/val"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:--1}"

BATCH_SIZE_EVAL=512
BATCH_SIZE_CACHE=512
# for resnext101, smaller cache bs
if [[ "${MODEL_NAME}" == "resnext101_32x8d" ]]; then
  BATCH_SIZE_CACHE=256
fi

echo "=== ImageNet ${MODEL_NAME}, seed ${SEED}, mode ${MODE} ==="
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_NAME}" \
  --results_dir "${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain/${MODE}/seed-${SEED}" \
  --mode "${MODE}" \
  --seed "${SEED}" \
  --train_dir "${TRAIN_DIR}" \
  --val_dir "${VAL_DIR}" \
  --batch_size_eval "${BATCH_SIZE_EVAL}" \
  --batch_size_cache "${BATCH_SIZE_CACHE}" \
  --eval_subset_size "${EVAL_SUBSET_SIZE}"
