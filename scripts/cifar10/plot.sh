#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Path to the single-figure plotting script
PLOT_SCRIPT="${SCRIPT_DIR}/plot_acc_vs_complexity.py"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"
mkdir -p "${OUTPUT_DIR}"

results_path() {
  local model_name="$1"
  local mode="$2"
  echo "${ROOT_DIR}/results/cifar10/${model_name}/factorized_posttrain/${mode}"
}

echo "=== Plotting ResNet20 ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet20 \
  --flops_json "$(results_path resnet20 flops_auto)" \
  --params_json "$(results_path resnet20 params_auto)" \
  --energy_json "$(results_path resnet20 energy)" \
  --energy_act_aware_json "$(results_path resnet20 energy_act_aware)" \
  --uniform_json "$(results_path resnet20 uniform)" \
  --uniform_act_aware_json "$(results_path resnet20 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

echo "=== Plotting ResNet56 ==="
python "${PLOT_SCRIPT}" \
  --model_name resnet56 \
  --flops_json "$(results_path resnet56 flops_auto)" \
  --params_json "$(results_path resnet56 params_auto)" \
  --energy_json "$(results_path resnet56 energy)" \
  --energy_act_aware_json "$(results_path resnet56 energy_act_aware)" \
  --uniform_json "$(results_path resnet56 uniform)" \
  --uniform_act_aware_json "$(results_path resnet56 uniform_act_aware)" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
