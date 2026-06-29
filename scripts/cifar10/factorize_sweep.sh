#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
MODEL_NAME="${MODEL_NAME:?MODEL_NAME not set}"
MODE="${MODE:?MODE not set}"
SEED="${SEED:-0}"
CIFAR10_ROOT="${CIFAR10_ROOT:?CIFAR10_ROOT not set}"

case "${MODEL_NAME}" in
  resnet20|resnet56) ;;
  *) echo "Unsupported CIFAR-10 model: ${MODEL_NAME}" >&2; exit 2 ;;
esac

PRETRAINED_PATH="${ROOT_DIR}/results/cifar10/${MODEL_NAME}/base/model.pth"

echo "=== CIFAR-10 ${MODEL_NAME}, seed ${SEED}, mode ${MODE} ==="
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name "${MODEL_NAME}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  --results_dir "${ROOT_DIR}/results/cifar10/${MODEL_NAME}/factorized_posttrain/${MODE}/seed-${SEED}" \
  --data_root "${CIFAR10_ROOT}" \
  --mode "${MODE}" \
  --seed "${SEED}"
