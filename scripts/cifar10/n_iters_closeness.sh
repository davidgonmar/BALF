#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


python "${SCRIPT_DIR}/n_iters_closeness.py" \
  --model resnet20 \
  --pretrained "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --model resnet56 \
  --pretrained "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/n_iters_closeness" \
  --seed 0

python "${SCRIPT_DIR}/plot_n_iters_closeness.py" \
  --results_dir "${ROOT_DIR}/results/cifar10/n_iters_closeness" \
  --out "${ROOT_DIR}/results/cifar10/plots/n_iters_closeness.pdf"
  
echo "Plots saved in ${ROOT_DIR}/results/cifar10/plots/n_iters_closeness.pdf."