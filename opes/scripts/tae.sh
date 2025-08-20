cd ../

GPU=${1:-6}

python main.py \
    --config-name tae \
    step=100_000 \
    gpu=$GPU