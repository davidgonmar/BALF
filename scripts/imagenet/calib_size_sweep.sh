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
IMAGENET_ROOT="${IMAGENET_ROOT:?IMAGENET_ROOT not set}"

RESULTS_DIR="${ROOT_DIR}/results/imagenet/calib_size_sweep_${MODEL_NAME}_${MODE}"
BATCH_SIZE_EVAL=512
BATCH_SIZE_CACHE=512
EVAL_SUBSET_SIZE=-1
RATIOS=(0.4 0.5 0.7 0.8)

OUTPUT_DIR="${ROOT_DIR}/results/imagenet/plots"
mkdir -p "${OUTPUT_DIR}"

echo "=== ImageNet calib-size sweep for ${MODEL_NAME}, mode ${MODE} ==="
python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name "${MODEL_NAME}" \
  --results_dir "${RESULTS_DIR}" \
  --train_dir "${IMAGENET_ROOT}/train" \
  --val_dir "${IMAGENET_ROOT}/val" \
  --mode "${MODE}" \
  --ratios "${RATIOS[@]}" \
  --batch_size_eval "${BATCH_SIZE_EVAL}" \
  --batch_size_cache "${BATCH_SIZE_CACHE}" \
  --eval_subset_size "${EVAL_SUBSET_SIZE}"

echo "=== Plotting ${MODEL_NAME} calib-size sweep ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${RESULTS_DIR}" \
  --model_name "${MODEL_NAME}" \
  --out "${OUTPUT_DIR}/calib_size_sweep_${MODEL_NAME}_${MODE}.pdf"
