cd ../

GPU=${1:-5}

python main.py \
    --config-name tica \
    step=100_000 \
    gpu=$GPU