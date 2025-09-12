#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Output directory for plots
OUTPUT_DIR="${ROOT_DIR}/results/imagenet/plots"

mkdir -p "${OUTPUT_DIR}"


:<< 'COMMENT'
# first run the sweep
python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet18 \
  --results_dir "${ROOT_DIR}/results/imagenet/calib_size_sweep_resnet18_params" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --mode params_auto \
  --ratios 0.4 0.5 0.7 0.8 \
  --eval_subset_size 5000 \


echo "=== Plotting ResNet18 Calib Size Sweep (single figure) ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/imagenet/calib_size_sweep_resnet18_params" \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet18_params.pdf" \
COMMENT

# ResNet50
python "${SCRIPT_DIR}/calib_size_sweep.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/calib_size_sweep_resnet50_params" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --mode params_auto \
  --ratios 0.4 0.5 0.7 0.8 \
  --eval_subset_size 5000 \

echo "=== Plotting ResNet50 Calib Size Sweep (single figure) ==="
python "${SCRIPT_DIR}/../plot_calib_size_sweep.py" \
  --results_dir "${ROOT_DIR}/results/imagenet/calib_size_sweep_resnet50_params" \
  --out "${OUTPUT_DIR}/calib_size_sweep_resnet50_params.pdf" \
  