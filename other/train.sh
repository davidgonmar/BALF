#!/usr/bin/env bash
set -e           # exit on the first error
RESULTS_DIR=./results
mkdir -p "$RESULTS_DIR"   # make sure the directory exists

for REG in 0.0005 0.002 0.003 0.004 0.001   0.005; do
  echo "==> Running with --reg_weight ${REG}"
  python train.py \
    --reg_weight "${REG}" \
    --save_path "${RESULTS_DIR}/cifar10_resnet20_hoyer_finetuned_reg${REG}.pth" \
    --log_path  "${RESULTS_DIR}/cifar10_resnet20_hoyer_reg${REG}.json" \
    --reg_state_path "${RESULTS_DIR}/reg_state${REG}.pth"
done