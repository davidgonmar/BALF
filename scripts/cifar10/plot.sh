#!/usr/bin/env bash
set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths to JSON results
RES20_FLOPS="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/flops_auto/results.json"
RES20_PARAMS="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/params_auto/results.json"
RES20_ENERGY="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy/results.json"
RES20_ENERGY_ACT_AWARE="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy_act_aware/results.json"

RES56_FLOPS="${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/flops_auto/results.json"
RES56_PARAMS="${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/params_auto/results.json"
RES56_ENERGY="${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy/results.json"
RES56_ENERGY_ACT_AWARE="${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy_act_aware/results.json"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"
mkdir -p "${OUTPUT_DIR}"

# Run the plotting script
python "${SCRIPT_DIR}/plot.py" \
  --res20_flops "${RES20_FLOPS}" \
  --res20_params "${RES20_PARAMS}" \
  --res20_energy "${RES20_ENERGY}" \
  --res20_energy_act_aware "${RES20_ENERGY_ACT_AWARE}" \
  --res56_flops "${RES56_FLOPS}" \
  --res56_params "${RES56_PARAMS}" \
  --res56_energy "${RES56_ENERGY}" \
  --res56_energy_act_aware "${RES56_ENERGY_ACT_AWARE}" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
