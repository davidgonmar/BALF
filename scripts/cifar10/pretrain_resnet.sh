#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CIFAR10_ROOT="${CIFAR10_ROOT:?CIFAR10_ROOT not set}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT_DIR}/results/cifar10/resnet20/base" "${ROOT_DIR}/results/cifar10/resnet56/base"

python "${SCRIPT_DIR}/pretrain_resnet.py" \
  --data-root "${CIFAR10_ROOT}" \
  --weights-out "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --epochs-out  "${ROOT_DIR}/results/cifar10/resnet20/base/epochs.json" \
  --model resnet20 \
  --epochs 200 \
  --batch-size 128 \
  --val-batch-size 1024 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --milestones 100 150 \
  --gamma 0.1 \
  --num-workers 8 \
  --seed 0

python "${SCRIPT_DIR}/pretrain_resnet.py" \
  --data-root "${CIFAR10_ROOT}" \
  --weights-out "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --epochs-out  "${ROOT_DIR}/results/cifar10/resnet56/base/epochs.json" \
  --model resnet56 \
  --epochs 200 \
  --batch-size 128 \
  --val-batch-size 1024 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --milestones 100 150 \
  --gamma 0.1 \
  --num-workers 8 \
  --seed 0
