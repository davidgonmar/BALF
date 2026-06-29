#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CIFAR10_ROOT="${CIFAR10_ROOT:?CIFAR10_ROOT not set}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots"

mkdir -p "${OUTPUT_DIR}"
MODE=params_auto

python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet20 \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet20_${MODE}" \
  --mode "${MODE}" \
  --ratios 0.4 0.5 0.7 0.8 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --data_root "${CIFAR10_ROOT}"

python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet56 \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet56_${MODE}" \
  --mode "${MODE}" \
  --ratios 0.4 0.5 0.7 0.8 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --data_root "${CIFAR10_ROOT}"


echo "=== Plotting ResNet20 Calib Size Sweep ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet20_${MODE}" \
  --model_name resnet20 \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet20_${MODE}.pdf"

echo "=== Plotting ResNet56 Calib Size Sweep ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/cifar10/calib_size_sweep_resnet56_${MODE}" \
  --model_name resnet56 \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet56_${MODE}.pdf"

  
echo "=== All done! Plots saved to ${OUTPUT_DIR} ==="
