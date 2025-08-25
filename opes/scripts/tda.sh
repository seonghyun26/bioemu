cd ../

GPU=${1:-4}

python main.py \
    --config-name tda \
    step=600_000_000 \
    gpu=$GPU &