#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT_DIR="${ROOT_DIR}/results/cifar10/resnet20/figs_appendix_cifar10c"
PARAMS_JSON="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain_cifar10c/params_auto/results.json"
FLOPS_JSON="${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain_cifar10c/flops_auto/results.json"

python "${SCRIPT_DIR}/show_corrupted_sweep_results.py" \
  --params_auto_json "${PARAMS_JSON}" \
  --flops_auto_json "${FLOPS_JSON}" \
  --out_dir "${OUT_DIR}"