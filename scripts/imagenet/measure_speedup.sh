SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


BATCH_SIZES=(8 16 32)
CALIB_SIZE=8192
BATCH_SIZE_CACHE=128
ITERS=10
WARMUP=3

# =========================================================
# ResNet50
# =========================================================
echo "=== Measuring speedup on ResNet50"
python "${SCRIPT_DIR}/measure_speedup.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/resnet50/speedup_flops_auto" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --seed 0 \
  --calib_size ${CALIB_SIZE} \
  --batch_size_cache ${BATCH_SIZE_CACHE} \
  --batch_sizes "${BATCH_SIZES[@]}" \
  --iters ${ITERS} \
  --warmup ${WARMUP}

# =========================================================
# ViT-B/16
# =========================================================
echo "=== Measuring speedup on ViT-B/16"
python "${SCRIPT_DIR}/measure_speedup.py" \
  --model_name vit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/vit_b_16/speedup_flops_auto" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --seed 0 \
  --calib_size ${CALIB_SIZE} \
  --batch_size_cache ${BATCH_SIZE_CACHE} \
  --batch_sizes "${BATCH_SIZES[@]}" \
  --iters ${ITERS} \
  --warmup ${WARMUP}
