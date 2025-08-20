cd ../

GPU=${1:-7}

python main.py \
    --config-name vde \
    step=100_000 \
    gpu=$GPU