#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


# First run the sweep on ResNet20
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0


python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy" \
  --mode energy \
  --seed 0

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet20 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet20/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet20/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0

# Then run the sweep on ResNet56
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet56 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet56 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet56 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy" \
  --mode energy \
  --seed 0

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet56 \
  --pretrained_path "${ROOT_DIR}/results/cifar10/resnet56/base/model.pth" \
  --results_dir "${ROOT_DIR}/results/cifar10/resnet56/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0
