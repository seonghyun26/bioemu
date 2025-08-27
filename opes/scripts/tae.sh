cd ../

GPU=${1:-6}
SIGMA=${2:-0.2}

python main.py \
    --config-name tae \
    step=100_000_000 \
    gpu=$GPU \
    sigma=$SIGMA &