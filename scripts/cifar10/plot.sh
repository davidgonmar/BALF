#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Path to the single-figure plotting script
PLOT_SCRIPT="${SCRIPT_DIR}/plot.py"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"
mkdir -p "${OUTPUT_DIR}"

echo "=== Plotting ResNet20 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet20 \
  --flops_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

echo "=== Plotting ResNet56 (single figure) ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet56 \
  --flops_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/flops_auto/results.json" \
  --params_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/params_auto/results.json" \
  --energy_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy/results.json" \
  --energy_act_aware_json "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy_act_aware/results.json" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
