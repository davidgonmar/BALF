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
CIFAR10C_ROOT="${CIFAR10C_ROOT:?CIFAR10C_ROOT not set}"

PRETRAINED_PATH="${ROOT_DIR}/results/cifar10/${MODEL_NAME}/base/model.pth"
OUT_DIR="${ROOT_DIR}/results/cifar10/${MODEL_NAME}/factorized_posttrain_cifar10c/${MODE}"

python "${SCRIPT_DIR}/corrupted_sweep.py" \
  --model_name "${MODEL_NAME}" \
  --pretrained_path "${PRETRAINED_PATH}" \
  --results_json "${OUT_DIR}/results.json" \
  --cifar10c_root "${CIFAR10C_ROOT}" \
  --data_root "${CIFAR10_ROOT}" \
  --mode "${MODE}" \
  --seed "${SEED}"
