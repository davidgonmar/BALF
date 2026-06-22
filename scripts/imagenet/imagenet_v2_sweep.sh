#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_NAME="resnet18"
IMAGENET_TRAIN_DIR="${ROOT_DIR}/imagenet-calib"
IMAGENET_VAL_DIR="${ROOT_DIR}/imagenet-val"
IMAGENETV2_ROOT="${ROOT_DIR}/ImageNet-V2"
SEED="0"

OUT_DIR="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetv2_all"

if false; then
  echo "=== Running ${MODEL_NAME} on ImageNet-V2 all variants (params_auto) ==="
  python "${SCRIPT_DIR}/imagenet_v2_sweep.py" \
    --model_name "${MODEL_NAME}" \
    --results_json "${OUT_DIR}/params_auto/results.json" \
    --imagenetv2_root "${IMAGENETV2_ROOT}" \
    --train_dir "${IMAGENET_TRAIN_DIR}" \
    --val_dir "${IMAGENET_VAL_DIR}" \
    --mode params_auto \
    --seed "${SEED}"

  echo "=== Running ${MODEL_NAME} on ImageNet-V2 all variants (flops_auto) ==="
  python "${SCRIPT_DIR}/imagenet_v2_sweep.py" \
    --model_name "${MODEL_NAME}" \
    --results_json "${OUT_DIR}/flops_auto/results.json" \
    --imagenetv2_root "${IMAGENETV2_ROOT}" \
    --train_dir "${IMAGENET_TRAIN_DIR}" \
    --val_dir "${IMAGENET_VAL_DIR}" \
    --mode flops_auto \
    --seed "${SEED}"
fi

FIG_OUT_DIR="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/figs_appendix_imagenetv2"
PARAMS_JSON="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetv2_all/params_auto/results.json"
FLOPS_JSON="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetv2_all/flops_auto/results.json"

echo "=== Plotting ${MODEL_NAME} on ImageNet-V2 (all variants) ==="
python "${SCRIPT_DIR}/show_imagenet_v2_sweep_results.py" \
  --model_name "${MODEL_NAME}" \
  --params_auto_json "${PARAMS_JSON}" \
  --flops_auto_json "${FLOPS_JSON}" \
  --out_dir "${FIG_OUT_DIR}"