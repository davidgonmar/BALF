#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TABLE_SCRIPT="${SCRIPT_DIR}/../create_tables.py"
OUTPUT_DIR="${ROOT_DIR}/results/imagenet/tables"
mkdir -p "${OUTPUT_DIR}"

echo "=== Creating ResNet50 table (row-grouped) ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/imagenet/resnet50/factorized_posttrain" \
  --ratios 0.4,0.5,0.7,0.8 \
  --decimals 2 \
  > "${OUTPUT_DIR}/resnet50_table.tex"

echo "=== Creating DeiT-B_16 table (row-grouped) ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain" \
  --ratios 0.5,0.6,0.7,0.8 \
  --decimals 2 \
  > "${OUTPUT_DIR}/deit_b_16_table.tex"

echo "=== Creating MobileNetV2 table (row-grouped) ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain" \
  --ratios 0.7,0.75,0.8,0.9,0.97 \
  --decimals 2 \
  > "${OUTPUT_DIR}/mobilenetv2_table.tex"
