#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

SEED="${SEED:-0}"
CIFAR10_ROOT="${CIFAR10_ROOT:?CIFAR10_ROOT not set}"
BALF_CACHE_ROOT="${BALF_CACHE_ROOT:?BALF_CACHE_ROOT not set}"

python "${SCRIPT_DIR}/compare_svd_vs_actaware.py" \
  --results_dir "${ROOT_DIR}/results/compare_svd_vs_actaware" \
  --data_root "${CIFAR10_ROOT}" \
  --tmp_dir "${BALF_CACHE_ROOT}/compare_svd_vs_actaware_seed-${SEED}" \
  --seed "${SEED}"
