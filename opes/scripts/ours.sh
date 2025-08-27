cd ../

GPU=${1:-7}
SIGMA=${2:-0.2}

date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name ours \
    step=100_000_000 \
    date=$date \
    gpu=$GPU \
    sigma=$SIGMA \

python analysis_opes.py \
    --date=$date
