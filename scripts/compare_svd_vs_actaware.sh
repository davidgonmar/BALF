#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/compare_svd_vs_actaware.py" \
  --results_dir "${ROOT_DIR}/results/compare_svd_vs_actaware" \