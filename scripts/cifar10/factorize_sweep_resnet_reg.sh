#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_factorize_sweep_on_regularized_resnet.sh [PARALLELISM]
PARALLELISM="${1:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Models and factorization modes
MODELS=(resnet20 resnet56)
MODES=(params_auto flops_auto)

# Regularization hyperparameter grid (must match training script)
RWS=(0.001 0.003 0.005)
SEFS=(0.01 0.003 0.005)
SES=(300 200)

CMDS=()
for MODEL in "${MODELS[@]}"; do
  for RW in "${RWS[@]}"; do
    for SEF in "${SEFS[@]}"; do
      for SE in "${SES[@]}"; do
        # Base directory for this regularized model checkpoint
        MODEL_DIR="${ROOT_DIR}/results/cifar10/${MODEL}/reg_train/rw${RW}_sef${SEF}_se${SE}"
        PRETRAIN="${MODEL_DIR}/model.pth"

        if [ ! -f "${PRETRAIN}" ]; then
          echo "[WARNING] Missing model file: ${PRETRAIN}, skipping." >&2
          continue
        fi

        # Root for posttrain factorization on regularized model
        CONFIG_DIR="${ROOT_DIR}/results/cifar10/${MODEL}/factorized_posttrain_reg/rw${RW}_sef${SEF}_se${SE}"
        mkdir -p "${CONFIG_DIR}"

        for MODE in "${MODES[@]}"; do
          RESULTS_DIR="${CONFIG_DIR}/${MODE}"
          mkdir -p "${RESULTS_DIR}"

          CMDS+=(
            "python \"${SCRIPT_DIR}/factorize_sweep_resnet.py\" \
              --model_name ${MODEL} \
              --pretrained_path \"${PRETRAIN}\" \
              --results_dir \"${RESULTS_DIR}\" \
              --mode ${MODE} \
              --seed 0"
          )
        done
      done
    done
  done

done

# Execute all factorization jobs in parallel
printf '%s\n' "${CMDS[@]}" | xargs -P "${PARALLELISM}" -I CMD bash -c CMD
