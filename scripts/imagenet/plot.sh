#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to the single-figure plotting script
PLOT_SCRIPT="${SCRIPT_DIR}/plot_acc_vs_complexity.py"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/imagenet/plots"
mkdir -p "${OUTPUT_DIR}"

echo "=== Plotting ResNet18 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet18 \
  --flops_json "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/resnet18//factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

echo "=== Plotting MobileNetV2 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name mobilenet_v2 \
  --flops_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

# resnext50_32x4d
echo "=== Plotting ResNeXt50_32x4d (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnext50_32x4d \
  --flops_json "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

# resnext101_32x8d
echo "=== Plotting ResNeXt101_32x8d (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnext101_32x8d \
  --flops_json "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

# VIT
echo "=== Plotting ViT-B_16 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name vit_b_16 \
  --flops_json "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"
  
echo "Plots saved in ${OUTPUT_DIR}."


# DeiT
echo "=== Plotting DeiT-B_16 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name deit_b_16 \
  --flops_json "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"
  