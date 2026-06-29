#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

TABLE_SCRIPT="${SCRIPT_DIR}/../create_tables.py"
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/tables"
mkdir -p "${OUTPUT_DIR}"

echo "=== Creating ResNet20 table ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain" \
  --ratios 0.4,0.5,0.7,0.8 \
  --decimals 2 \
  > "${OUTPUT_DIR}/resnet20_table.tex"
