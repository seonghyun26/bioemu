#!/usr/bin/env bash
set -euo pipefail

# Move to project root (opes/)
cd "$(dirname "$0")/.."

method="tica"
molecule="1fme"
date_str="${1:-$(date +'%Y%m%d_%H%M%S')}-short"
echo "$date_str"

python main.py \
    --config-name "$method-$molecule" \
    date="$date_str" \
    +tags=['debug','cuda-check'] \
    opes.max_seed=0 \
    opes.step=10000 \
    start_gpu=0 \
    ++gpu=1

wait

python scripts/summarize_cuda_perf.py \
    --method "$method" \
    --molecule "$molecule" \
    --date "$date_str" \
    --max-seed 0


