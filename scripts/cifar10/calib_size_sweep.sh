#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"

mkdir -p "${OUTPUT_DIR}"

# first run the sweep
python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet20 \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet20_params" \
  --mode params_auto \
  --ratios 0.4 0.5 0.7 0.8 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \


echo "=== Plotting ResNet20 Calib Size Sweep (single figure) ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet20_params" \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet20_params.pdf" \
