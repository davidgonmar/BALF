#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root of the project (two levels up)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Output directory for the four plots
OUTPUT_DIR="${ROOT_DIR}/results/cifar10/plots/reg_factorized"
mkdir -p "${OUTPUT_DIR}"

# Invoke the Python plotting driver
python "${SCRIPT_DIR}/plot_reg_debug.py" \
  --root-dir "${ROOT_DIR}/results/cifar10" \
  --output-dir "${OUTPUT_DIR}"

echo "All plots saved under ${OUTPUT_DIR}"
