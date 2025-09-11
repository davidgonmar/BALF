#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CIFAR10C_ROOT="${ROOT_DIR}/CIFAR-10-C"  # adjust if your CIFAR-10-C .npy files live elsewhere
OUT_DIR="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain_cifar10c"


python "${SCRIPT_DIR}/corrupted_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_json "${OUT_DIR}/params_auto/results.json" \
  --cifar10c_root "${CIFAR10C_ROOT}" \
  --mode params_auto \
  --seed 0


python "${SCRIPT_DIR}/corrupted_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_json "${OUT_DIR}/flops_auto/results.json" \
  --cifar10c_root "${CIFAR10C_ROOT}" \
  --mode flops_auto \
  --seed 0
