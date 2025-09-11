#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


  # plot

PLOT_SCRIPT="${SCRIPT_DIR}/plot_calib_size_sweep.py"
# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/imagenet/plots"
mkdir -p "${OUTPUT_DIR}"
echo "=== Plotting ResNet18 Calib Size Sweep (single figure) ==="
python "${PLOT_SCRIPT}" \
  --results_dir "${ROOT_DIR}/results/imagenet/calib_size_sweep_035_resnet18_params"\
  --out "${OUTPUT_DIR}/resnet18_calib_size_sweep_params.pdf" \
