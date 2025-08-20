cd ../

GPU=${1:-4}

python main.py \
    --config-name tda \
    step=100_000 \
    gpu=$GPU