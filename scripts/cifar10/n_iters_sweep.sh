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


python "${SCRIPT_DIR}/n_iters_sweep.py" \
  --model resnet20 \
  --pretrained "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --model resnet56 \
  --pretrained "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/n_iters_sweep" \
  --data_root "${CIFAR10_ROOT}" \
  --seed 0


python "${SCRIPT_DIR}/plot_n_iters_sweep.py" \
  --summary "${ROOT_DIR}/results/cifar10/n_iters_sweep/summary.json" \
  --out_usage "${ROOT_DIR}/results/cifar10/plots/n_iters_sweep_usage.pdf" \
  --out_accuracy "${ROOT_DIR}/results/cifar10/plots/n_iters_sweep_accuracy.pdf"
  
echo "Plots saved in ${ROOT_DIR}/results/cifar10/plots/n_iters_sweep.pdf"
