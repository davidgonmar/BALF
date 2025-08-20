#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"






python "${SCRIPT_DIR}/factorize_sweep_resnet.py" \
  --model_name resnet18 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val"