#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf


# First, do the different runs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

CIFAR10_ROOT="${CIFAR10_ROOT:?CIFAR10_ROOT not set}"
CIFAR10C_ROOT="${CIFAR10C_ROOT:?CIFAR10C_ROOT not set}"
OUT_DIR="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain_cifar10c"


: <<'MULTILINE_COMMENT'
python "${SCRIPT_DIR}/corrupted_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_json "${OUT_DIR}/params_auto/results.json" \
  --cifar10c_root "${CIFAR10C_ROOT}" \
  --data_root "${CIFAR10_ROOT}" \
  --mode params_auto \
  --seed 0

python "${SCRIPT_DIR}/corrupted_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_json "${OUT_DIR}/flops_auto/results.json" \
  --cifar10c_root "${CIFAR10C_ROOT}" \
  --data_root "${CIFAR10_ROOT}" \
  --mode flops_auto \
  --seed 0
MULTILINE_COMMENT


# Then, plot the results
OUT_DIR="${ROOT_DIR}/results/cifar10/resnet20/figs_appendix_cifar10c"
PARAMS_JSON="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain_cifar10c/params_auto/results.json"
FLOPS_JSON="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain_cifar10c/flops_auto/results.json"

python "${ROOT_DIR}/scripts/show_corrupted_sweep_results.py" \
  --params_auto_json "${PARAMS_JSON}" \
  --flops_auto_json "${FLOPS_JSON}" \
  --out_dir "${OUT_DIR}" \
  --dataset_name "CIFAR-10-C" \
  --ylabel "Accuracy"
