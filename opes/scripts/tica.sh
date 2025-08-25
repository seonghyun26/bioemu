cd ../

GPU=${1:-5}

python main.py \
    --config-name tica \
    step=100_000_000 \
    gpu=$GPU &