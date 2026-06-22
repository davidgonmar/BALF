#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_NAME="resnet18"
IMAGENET_TRAIN_DIR="${ROOT_DIR}/imagenet-calib"
IMAGENET_VAL_DIR="${ROOT_DIR}/imagenet-val"
IMAGENETC_ROOT="${ROOT_DIR}/ImageNet-C"
SEED="0"

SUBSETS=(noise weather blur digital extra)

for SUBSET in "${SUBSETS[@]}"; do

  if false; then
    echo "=== Running ${MODEL_NAME} on ImageNet-C subset: ${SUBSET} (params_auto) ==="
    OUT_DIR="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetc/${SUBSET}"

    python "${SCRIPT_DIR}/corrupted_sweep.py" \
      --model_name "${MODEL_NAME}" \
      --results_json "${OUT_DIR}/params_auto/results.json" \
      --imagenetc_root "${IMAGENETC_ROOT}" \
      --imagenetc_subset "${SUBSET}" \
      --train_dir "${IMAGENET_TRAIN_DIR}" \
      --val_dir "${IMAGENET_VAL_DIR}" \
      --mode params_auto \
      --seed "${SEED}"

    echo "=== Running ${MODEL_NAME} on ImageNet-C subset: ${SUBSET} (flops_auto) ==="
    python "${SCRIPT_DIR}/corrupted_sweep.py" \
      --model_name "${MODEL_NAME}" \
      --results_json "${OUT_DIR}/flops_auto/results.json" \
      --imagenetc_root "${IMAGENETC_ROOT}" \
      --imagenetc_subset "${SUBSET}" \
      --train_dir "${IMAGENET_TRAIN_DIR}" \
      --val_dir "${IMAGENET_VAL_DIR}" \
      --mode flops_auto \
      --seed "${SEED}"
  fi

  FIG_OUT_DIR="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/figs_appendix_imagenetc/${SUBSET}"
  PARAMS_JSON="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetc/${SUBSET}/params_auto/results.json"
  FLOPS_JSON="${ROOT_DIR}/results/imagenet/${MODEL_NAME}/factorized_posttrain_imagenetc/${SUBSET}/flops_auto/results.json"

  echo "=== Plotting ${MODEL_NAME} on ImageNet-C subset: ${SUBSET} ==="
  python "${SCRIPT_DIR}/show_corrupted_sweep_results.py" \
    --model_name "${MODEL_NAME}" \
    --imagenetc_subset "${SUBSET}" \
    --params_auto_json "${PARAMS_JSON}" \
    --flops_auto_json "${FLOPS_JSON}" \
    --out_dir "${FIG_OUT_DIR}"
done
