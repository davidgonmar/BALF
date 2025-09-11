#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


  python "${SCRIPT_DIR}/factorize_sweep_resnet.py" \
    --model_name deit_b_16 \
    --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/flops_auto" \
    --mode flops_auto \
    --seed 0 \
    --train_dir "${ROOT_DIR}/imagenet-calib" \
    --val_dir "${ROOT_DIR}/imagenet-val" \
    --batch_size_eval 128 \
    --batch_size_cache 64 \
    --cache_file "${ROOT_DIR}/deit_b_16_imagenet_cache.pt"

  python "${SCRIPT_DIR}/factorize_sweep_resnet.py" \
    --model_name deit_b_16 \
    --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/params_auto" \
    --mode params_auto \
    --seed 0 \
    --train_dir "${ROOT_DIR}/imagenet-calib" \
    --val_dir "${ROOT_DIR}/imagenet-val" \
    --batch_size_eval 128 \
    --batch_size_cache 64 \
    --cache_file "${ROOT_DIR}/deit_b_16_imagenet_cache.pt"

  python "${SCRIPT_DIR}/factorize_sweep_resnet.py" \
    --model_name deit_b_16 \
    --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/energy" \
    --mode energy \
    --seed 0 \
    --train_dir "${ROOT_DIR}/imagenet-calib" \
    --val_dir "${ROOT_DIR}/imagenet-val" \
    --batch_size_eval 128 \
    --batch_size_cache 64 \
    --cache_file "${ROOT_DIR}/deit_b_16_imagenet_cache.pt"


  python "${SCRIPT_DIR}/factorize_sweep_resnet.py" \
    --model_name deit_b_16 \
    --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/energy_act_aware" \
    --mode energy_act_aware \
    --seed 0 \
    --train_dir "${ROOT_DIR}/imagenet-calib" \
    --val_dir "${ROOT_DIR}/imagenet-val" \
    --batch_size_eval 128 \
    --batch_size_cache 64 \
    --cache_file "${ROOT_DIR}/deit_b_16_imagenet_cache.pt"
