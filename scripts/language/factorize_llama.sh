#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_NAME="meta-llama/llama-2-7b-hf"
CACHE_PATH="$DIR/cache/llama7b_wikitext.pt"
RESULTS_DIR="$DIR/results/llama7b_flops05"
RATIO=0.8
SEED=42
python "$DIR/factorize_llama.py" \
  --model_name "$MODEL_NAME" \
  --cache_path "$CACHE_PATH" \
  --results_dir "$RESULTS_DIR" \
  --mode params_auto \
  --ratio_to_keep "$RATIO" \
  --seed "$SEED"