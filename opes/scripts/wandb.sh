cd ../

GPU=${1:-0}

python main.py \
    --config-name tda \
    step=1_000 \
    gpu=$GPU