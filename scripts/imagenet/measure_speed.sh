#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

MODEL_NAME="${1:?model name not set}"
GPU_TAG="${2:?gpu tag not set}"
BATCH_SIZE_CACHE="${3:?cache batch size not set}"
shift 3
BATCH_SIZES=("$@")
if [ "${#BATCH_SIZES[@]}" -eq 0 ]; then
  echo "No batch sizes provided" >&2
  exit 1
fi

case "${GPU_TAG}" in
  a100) GPU_LABEL="A100" ;;
  rtx2080ti) GPU_LABEL="RTX 2080 Ti" ;;
  *) GPU_LABEL="${GPU_TAG}" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGENET_ROOT="${IMAGENET_ROOT:?IMAGENET_ROOT not set}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

CALIB_SIZE=8192
OUTPUT_TAG="${GPU_TAG}"
RESULTS_DIR="${ROOT_DIR}/results/imagenet/measure_speed/${GPU_TAG}"

echo "=== ${MODEL_NAME} on ${GPU_LABEL} ==="
echo "Batch sizes: ${BATCH_SIZES[*]}"
python "${SCRIPT_DIR}/measure_speed.py" \
  --model_name "${MODEL_NAME}" \
  --results_dir "${RESULTS_DIR}" \
  --train_dir "${IMAGENET_ROOT}/train" \
  --seed 0 \
  --calib_size ${CALIB_SIZE} \
  --batch_size_cache "${BATCH_SIZE_CACHE}" \
  --batch_sizes "${BATCH_SIZES[@]}" \
  --gpu_label "${GPU_LABEL}" \
  --output_tag "${OUTPUT_TAG}"
