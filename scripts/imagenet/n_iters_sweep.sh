#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${SCRIPT_DIR}/n_iters_sweep.py" \
  --model_name resnet18 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet18_n_iters_sweep" \
  --seed 0

python "${SCRIPT_DIR}/n_iters_sweep_imagenet.py" \
  --model_name resnet50 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet50_n_iters_sweep" \
  --seed 0

python "${SCRIPT_DIR}/plot_n_iters_sweep.py" \
  --results_dir "${ROOT_DIR}/results/imagenet" \
  --out "${ROOT_DIR}/results/imagenet/plots/n_iters_sweep.pdf"

echo "Plots saved in ${ROOT_DIR}/results/imagenet/plots/n_iters_sweep.pdf."