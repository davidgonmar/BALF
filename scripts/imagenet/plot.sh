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
  --model_name mobilenetv2 \
  --flops_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
