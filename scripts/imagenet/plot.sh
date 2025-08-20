#!/usr/bin/env bash
set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths to JSON results
RES18_FLOPS="${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/flops_auto/results.json"
RES18_PARAMS="${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/params_auto/results.json"
RES18_ENERGY="${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/energy/results.json"
RES18_ENERGY_ACT_AWARE="${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/energy_act_aware/results.json"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/imagenet/plots"
mkdir -p "${OUTPUT_DIR}"

# Run the plotting script
python "${SCRIPT_DIR}/plot.py" \
  --res18_flops "${RES18_FLOPS}" \
  --res18_params "${RES18_PARAMS}" \
  --res18_energy "${RES18_ENERGY}" \
  --res18_energy_act_aware "${RES18_ENERGY_ACT_AWARE}" \
  --output_dir "${OUTPUT_DIR}"

echo "Plots saved in ${OUTPUT_DIR}."
