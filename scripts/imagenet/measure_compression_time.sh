#!/usr/bin/env bash
set -euo pipefail

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate balf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGENET_ROOT="${IMAGENET_ROOT:?IMAGENET_ROOT not set}"
SEED="${SEED:?SEED not set}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

OUT_DIR="${ROOT_DIR}/results/imagenet/timings"
mkdir -p "${OUT_DIR}"

# use same calib_size as in our main experiments
python "${SCRIPT_DIR}/measure_compression_time.py" \
  --train_dir "${IMAGENET_ROOT}/train" \
  --out "${OUT_DIR}/timings_seed${SEED}.json" \
  --seed "${SEED}" \
  --batch_size 64 \
  --calib_size 8192
