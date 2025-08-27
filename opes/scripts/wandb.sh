cd ../

GPU=${1:-3}

python main.py \
    --config-name tda \
    step=10_000 \
    gpu=$GPU