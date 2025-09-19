#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"



python "${SCRIPT_DIR}/measure_compression_time.py" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --out "${ROOT_DIR}/results/imagenet/timings/timings.json" \
  --seed 0 \
  --batch_size 64 \
  --calib_size 8192 \

# =========================================================
# LaTeX table
# =========================================================
echo "=== Rendering LaTeX table"
python "${SCRIPT_DIR}/measure_compression_time_table.py" \
  --in_json "${ROOT_DIR}/results/imagenet/timings/timings.json" \
  --out_tex "${ROOT_DIR}/results/imagenet/timings/timings.tex"