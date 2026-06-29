#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Path to the single-figure plotting script
PLOT_SCRIPT="${SCRIPT_DIR}/plot_acc_vs_complexity.py"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/imagenet/plots"
mkdir -p "${OUTPUT_DIR}"

results_path() {
  local model_name="$1"
  local mode="$2"
  echo "${ROOT_DIR}/results/imagenet/${model_name}/factorized_posttrain/${mode}"
}

echo "=== Plotting ResNet18 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet18 \
  --flops_json "$(results_path resnet18 flops_auto)" \
  --params_json "$(results_path resnet18 params_auto)" \
  --energy_json "$(results_path resnet18 energy)" \
  --energy_act_aware_json "$(results_path resnet18 energy_act_aware)" \
  --uniform_json "$(results_path resnet18 uniform)" \
  --uniform_act_aware_json "$(results_path resnet18 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

# resnet50
echo "=== Plotting ResNet50 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet50 \
  --flops_json "$(results_path resnet50 flops_auto)" \
  --params_json "$(results_path resnet50 params_auto)" \
  --energy_json "$(results_path resnet50 energy)" \
  --energy_act_aware_json "$(results_path resnet50 energy_act_aware)" \
  --uniform_json "$(results_path resnet50 uniform)" \
  --uniform_act_aware_json "$(results_path resnet50 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

echo "=== Plotting MobileNetV2 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name mobilenet_v2 \
  --flops_json "$(results_path mobilenet_v2 flops_auto)" \
  --params_json "$(results_path mobilenet_v2 params_auto)" \
  --energy_json "$(results_path mobilenet_v2 energy)" \
  --energy_act_aware_json "$(results_path mobilenet_v2 energy_act_aware)" \
  --uniform_json "$(results_path mobilenet_v2 uniform)" \
  --uniform_act_aware_json "$(results_path mobilenet_v2 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

# resnext50_32x4d
echo "=== Plotting ResNeXt50_32x4d (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnext50_32x4d \
  --flops_json "$(results_path resnext50_32x4d flops_auto)" \
  --params_json "$(results_path resnext50_32x4d params_auto)" \
  --energy_json "$(results_path resnext50_32x4d energy)" \
  --energy_act_aware_json "$(results_path resnext50_32x4d energy_act_aware)" \
  --uniform_json "$(results_path resnext50_32x4d uniform)" \
  --uniform_act_aware_json "$(results_path resnext50_32x4d uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

# resnext101_32x8d
echo "=== Plotting ResNeXt101_32x8d (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnext101_32x8d \
  --flops_json "$(results_path resnext101_32x8d flops_auto)" \
  --params_json "$(results_path resnext101_32x8d params_auto)" \
  --energy_json "$(results_path resnext101_32x8d energy)" \
  --energy_act_aware_json "$(results_path resnext101_32x8d energy_act_aware)" \
  --uniform_json "$(results_path resnext101_32x8d uniform)" \
  --uniform_act_aware_json "$(results_path resnext101_32x8d uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

# ViT
echo "=== Plotting ViT-B/16 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name vit_b_16 \
  --flops_json "$(results_path vit_b_16 flops_auto)" \
  --params_json "$(results_path vit_b_16 params_auto)" \
  --energy_json "$(results_path vit_b_16 energy)" \
  --energy_act_aware_json "$(results_path vit_b_16 energy_act_aware)" \
  --uniform_json "$(results_path vit_b_16 uniform)" \
  --uniform_act_aware_json "$(results_path vit_b_16 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

# DeiT
echo "=== Plotting DeiT-B/16 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name deit_b_16 \
  --flops_json "$(results_path deit_b_16 flops_auto)" \
  --params_json "$(results_path deit_b_16 params_auto)" \
  --energy_json "$(results_path deit_b_16 energy)" \
  --energy_act_aware_json "$(results_path deit_b_16 energy_act_aware)" \
  --uniform_json "$(results_path deit_b_16 uniform)" \
  --uniform_act_aware_json "$(results_path deit_b_16 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
