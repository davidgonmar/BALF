#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


# calib size variable
EVAL_SUBSET_SIZE=5000


:<< 'COMMENT'
# =========================================================
# ResNet18
# =========================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet18 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
  --batch_size_eval 256

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet18 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
  --batch_size_eval 512

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet18 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
  --batch_size_eval 256

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet18 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
  --batch_size_eval 256
COMMENT


:<< 'COMMENT'
# =========================================================
# ResNet50
# =========================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet50/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE}

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet50/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE}

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet50/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE}

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet50/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE}
COMMENT



# =========================================================
# ResNeXt50_32x4d
# =========================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext50_32x4d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext50_32x4d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
: << 'COMMENT'
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext50_32x4d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext50_32x4d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext50_32x4d/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
COMMENT

: << 'COMMENT'
# ========================================================
# ResNeXt101_32x8d
# =========================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext101_32x8d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 64 \
  --batch_size_cache 32 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
  
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext101_32x8d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 64 \
  --batch_size_cache 32 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext101_32x8d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 64 \
  --batch_size_cache 32 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name resnext101_32x8d \
  --results_dir "${ROOT_DIR}/results/imagenet/resnext101_32x8d/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 64 \
  --batch_size_cache 32 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
COMMENT


:<< 'COMMENT'
# ==================================================
# MobileNetV2
# ==================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name mobilenet_v2 \
  --results_dir "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 256 \
  --batch_size_cache 128 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name mobilenet_v2 \
  --results_dir "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 256 \
  --batch_size_cache 128 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name mobilenet_v2 \
  --results_dir "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name mobilenet_v2 \
  --results_dir "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
COMMENT



:<< 'COMMENT'
# ========================================================
# ViT-B/16
# =========================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
  
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \


python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name vit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/vit_b_16/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
COMMENT



:<< 'COMMENT'
# ========================================================
# DeiT-B/16
# =========================================================
python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name deit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/flops_auto" \
  --mode flops_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name deit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/params_auto" \
  --mode params_auto \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name deit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/energy" \
  --mode energy \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \

python "${SCRIPT_DIR}/factorize_sweep.py" \
  --model_name deit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/deit_b_16/factorized_posttrain/energy_act_aware" \
  --mode energy_act_aware \
  --seed 0 \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --val_dir "${ROOT_DIR}/imagenet-val" \
  --batch_size_eval 128 \
  --batch_size_cache 64 \
  --eval_subset_size ${EVAL_SUBSET_SIZE} \
COMMENT

