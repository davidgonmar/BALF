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
IMAGENET_TRAIN_DIR="${IMAGENET_ROOT}/train"
IMAGENET_VAL_DIR="${IMAGENET_ROOT}/val"
IMAGENETC_ROOT="${IMAGENETC_ROOT:?IMAGENETC_ROOT not set}"
SUBSET="${SUBSET:?SUBSET not set}"

BATCH_SIZE_EVAL=512
BATCH_SIZE_CACHE=512
if [[ "${MODEL_NAME}" == "resnext101_32x8d" ]]; then
  BATCH_SIZE_CACHE=256
fi

OUT_DIR="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetc/${SUBSET}/${MODE}"

echo "=== ImageNet-C ${MODEL_NAME}, seed ${SEED}, mode ${MODE}, subset ${SUBSET} ==="
python "${SCRIPT_DIR}/corrupted_sweep.py" \
  --model_name "${MODEL_NAME}" \
  --results_json "${OUT_DIR}/results.json" \
  --imagenetc_root "${IMAGENETC_ROOT}" \
  --imagenetc_subset "${SUBSET}" \
  --train_dir "${IMAGENET_TRAIN_DIR}" \
  --val_dir "${IMAGENET_VAL_DIR}" \
  --mode "${MODE}" \
  --seed "${SEED}" \
  --batch_size_eval "${BATCH_SIZE_EVAL}" \
  --batch_size_cache "${BATCH_SIZE_CACHE}"
