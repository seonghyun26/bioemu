cd ../

GPU=${1:-7}

python main.py \
    --config-name vde \
    step=100_000_000 \
    gpu=$GPU &