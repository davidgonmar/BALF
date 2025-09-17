#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"

mkdir -p "${OUTPUT_DIR}"

# First run the sweep on ResNet20
python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet20 \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet20_params" \
  --mode params_auto \
  --ratios 0.4 0.5 0.7 0.8 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \

# Then run the sweep on ResNet56
python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet56 \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet56_params" \
  --mode params_auto \
  --ratios 0.4 0.5 0.7 0.8 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \


# Plot the results for ResNet20
echo "=== Plotting ResNet20 Calib Size Sweep ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet20_params" \
  --model_name resnet20 \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet20_params.pdf" \

# Plot the results for ResNet56
echo "=== Plotting ResNet56 Calib Size Sweep (single figure) ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet56_params" \
  --model_name resnet56 \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet56_params.pdf" \

  
echo "=== All done! Plots saved to ${OUTPUT_DIR} ==="