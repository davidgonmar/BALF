#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"



  python "${SCRIPT_DIR}/factorize_sweep_resnet.py" \
  --model_name mobilenet_v2 \
  --results_dir "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/flops_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --force_recache

