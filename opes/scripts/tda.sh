cd ../

GPU=${1:-4}
sigma=${2:-0.1}

date=$(date +%m%d_%H%M%S)
# date="0822_152426"
echo $date

python main.py \
    --config-name tda \
    step=200_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU &

python analysis_opes.py \
    --config-name tda \
    date=$date
