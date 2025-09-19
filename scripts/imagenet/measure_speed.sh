SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"


BATCH_SIZES=(8 16 32)
CALIB_SIZE=8192
BATCH_SIZE_CACHE=128

# =========================================================
# ResNet50
# =========================================================
echo "=== ResNet-50 ==="
python "${SCRIPT_DIR}/measure_speed.py" \
  --model_name resnet50 \
  --results_dir "${ROOT_DIR}/results/imagenet/measure_speed" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --seed 0 \
  --calib_size ${CALIB_SIZE} \
  --batch_size_cache ${BATCH_SIZE_CACHE} \
  --batch_sizes "${BATCH_SIZES[@]}" \

# =========================================================
# ViT-B/16
# =========================================================
echo "=== ViT-B/16 ==="
python "${SCRIPT_DIR}/measure_speed.py" \
  --model_name vit_b_16 \
  --results_dir "${ROOT_DIR}/results/imagenet/measure_speed" \
  --train_dir "${ROOT_DIR}/imagenet-calib" \
  --seed 0 \
  --calib_size ${CALIB_SIZE} \
  --batch_size_cache ${BATCH_SIZE_CACHE} \
  --batch_sizes "${BATCH_SIZES[@]}" \