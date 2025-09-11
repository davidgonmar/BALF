echo "=== Creating ResNet18 table (row-grouped) ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/imagenet/resnet18/factorized_posttrain" \
  --ratios 0.4,0.5,0.7,0.8 \
  --decimals 2 \
  > "${OUTPUT_DIR}/resnet18_table.tex"

echo "=== Creating MobileNetV2 table (row-grouped) ==="
python "${TABLE_SCRIPT}" \
  "${ROOT_DIR}/results/imagenet/mobilenet_v2/factorized_posttrain" \
  --ratios 0.7,0.8,0.9,0.97 \
  --decimals 2 \
  > "${OUTPUT_DIR}/mobilenetv2_table.tex"

echo "Tables saved in ${OUTPUT_DIR}."
