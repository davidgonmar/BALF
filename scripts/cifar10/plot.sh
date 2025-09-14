#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to the single-figure plotting script
PLOT_SCRIPT="${SCRIPT_DIR}/plot_acc_vs_complexity.py"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"
mkdir -p "${OUTPUT_DIR}"

echo "=== Plotting ResNet20 ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet20 \
  --flops_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

echo "=== Plotting ResNet56 ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet56 \
  --flops_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
