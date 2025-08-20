#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_NAME="meta-llama/llama-2-7b-hf"
CACHE_OUT="$DIR/cache/llama7b_wikitext.pt"
SEED=42
mkdir -p "$(dirname "$CACHE_OUT")"
python "$DIR/collect_activations_llama.py" \
  --model_name "$MODEL_NAME" \
  --dataset wikitext2 \
  --cache_out "$CACHE_OUT" \
  --seed "$SEED"
