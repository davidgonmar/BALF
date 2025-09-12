#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TABLE_SCRIPT="${SCRIPT_DIR}/../create_tables.py"
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/tables"
mkdir -p "${OUTPUT_DIR}"

echo "=== Creating ResNet20 table (row-grouped) ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain" \
  --ratios 0.4,0.5,0.7,0.8 \
  --decimals 2 \
  > "${OUTPUT_DIR}/resnet20_table.tex"
