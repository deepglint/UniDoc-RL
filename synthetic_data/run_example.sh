#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/generate_cot_data.py" \
    --data-path "slidevqa.json" \
    --output-path "output_file" \
    --num-workers 40 \
    --batch-size 4 \
    --max-steps 3 \
    --image-topk 3 \
    --crop-folder "crop_image_dir" \
    --enable-crop \
    --search-engine-url "${SEARCH_ENGINE_URL:-http://127.0.0.1:8200/search}" \
    --layout-parser-url "${LAYOUT_PARSER_URL:-http://127.0.0.1:30000}" \
    --llm-engine-url "${LLM_ENGINE_URL:-http://127.0.0.1:9009/v1}" \
    --vllm-engine-url "${VLLM_ENGINE_URL:-http://127.0.0.1:9000/v1}"
