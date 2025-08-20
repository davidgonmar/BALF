#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_train_regularized_resnet.sh [PARALLELISM]
PARALLELISM="${1:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure the top-level result directories exist
mkdir -p "${ROOT_DIR}/results/cifar10/resnet20/reg_train" \
         "${ROOT_DIR}/results/cifar10/resnet56/reg_train"

MODELS=(resnet20 resnet56)
RWS=(0.001 0.003 0.005)
SEFS=(0.01 0.003 0.005 )
SES=(300 200)

CMDS=()
#  --pretrained-path \"$PRETRAIN\" \
for MODEL in "${MODELS[@]}"; do
  PRETRAIN="${ROOT_DIR}/results/cifar10/${MODEL}/base/model.pth"
  for RW in "${RWS[@]}"; do
    for SEF in "${SEFS[@]}"; do
      for SE in "${SES[@]}"; do
        OUTDIR="${ROOT_DIR}/results/cifar10/${MODEL}/reg_train/rw${RW}_sef${SEF}_se${SE}"
        CMDS+=(
          "mkdir -p \"$OUTDIR\" && python \"$SCRIPT_DIR/train_regularized_resnet.py\" \
            --data-root \"$ROOT_DIR/data\" \
            --out-dir \"$OUTDIR\" \
            --model \"$MODEL\" \
            --pretrained-path \"$PRETRAIN\" \
            --epochs 100 \
            --batch-size 128 \
            --val-batch-size 1024 \
            --lr 0.001 \
            --momentum 0.9 \
            --weight-decay 0.0001 \
            --milestones 60 \
            --gamma 0.1 \
            --reg-weight \"$RW\" \
            --shrink-energy-frac \"$SEF\" \
            --shrink-every \"$SE\" \
            --num-workers 8 \
            --seed 0"
        )
      done
    done
  done
done

# Run in parallel; xargs returns non-zero if any job fails
printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD
